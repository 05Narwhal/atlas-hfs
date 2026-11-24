// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! The classic HFS catalog B-tree: keys, folder, file and thread records,
//! folder listings and lookups by name.
//!
//! The catalog is keyed by (parent id, name), but the name order of classic
//! HFS is a Mac OS Roman relational table this crate does not carry. Every
//! lookup here seeks to `(parent id, "")` instead: the empty name sorts
//! before every other under any table, so a descent that compares parent ids
//! and, on equal ids, treats the target as the smallest name, lands on the
//! first record of that parent. Named lookups then scan the folder
//! (design `m6-hfs.md`, decisions 3 and 4).

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use super::encoding::Encoding;
use super::extents::{
    EXTENT_RECORD_LEN, HfsExtentRecord, parse_extent_record, write_extent_record,
};
use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};
use atlas_fs_core::fs::{DirBatch, DirEntry, DirProgress, DirStream, NodeId, NodeKind};
use atlas_hfsplus::btree::{BTree, LeafCursor, Record};
use atlas_hfsplus::bytes::{array, be_u16, be_u32, bytes, u8_at};
use atlas_hfsplus::catalog::entry_flags;
use atlas_hfsplus::volume_header::hfs_time;

/// The parent id of the root folder.
pub const ROOT_PARENT_ID: u32 = 1;

/// The root folder's id.
pub const ROOT_FOLDER_ID: u32 = 2;

/// The first id given to a file or folder a user creates.
pub const FIRST_USER_ID: u32 = 16;

/// Longest name, in bytes, a key or a thread can hold.
pub const MAX_NAME_LEN: usize = 31;

/// Size of a folder record.
pub const FOLDER_RECORD_LEN: usize = 70;

/// Size of a file record.
pub const FILE_RECORD_LEN: usize = 102;

/// Size of a thread record.
pub const THREAD_RECORD_LEN: usize = 46;

/// Length of an index node's key, not counting its length byte: the key
/// field of an index record is padded to this.
pub const INDEX_KEY_LEN: usize = 37;

/// Most entries a [`ParentMap`] holds before it starts over.
pub const PARENT_MAP_CAP: usize = 1_000_000;

const STRUCTURE: &str = "catalog B-tree";

/// Bytes of a key before the name: reserved byte, parent id, name length.
const KEY_HEADER_LEN: usize = 6;

/// `cdrType` values.
const FOLDER_RECORD: u8 = 1;
const FILE_RECORD: u8 = 2;
const FOLDER_THREAD: u8 = 3;
const FILE_THREAD: u8 = 4;

/// Offset of the thread's parent id and of its name length byte.
const THREAD_PARENT_AT: usize = 10;
const THREAD_NAME_AT: usize = 14;

/// Offset of the Finder flags inside `FInfo` and `DInfo`.
const FINDER_FLAGS_AT: usize = 8;

fn corrupt(location: Location, kind: CorruptKind) -> Error {
    Error::Corrupt(Corruption::new(STRUCTURE, location, kind))
}

/// Length as `u64` for error values.
fn len64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

/// The location of the record `cursor` handed out last.
fn record_location(cursor: &LeafCursor) -> Location {
    cursor
        .position()
        .map_or(Location::Unspecified, |(node, record)| Location::Record {
            node,
            record,
        })
}

fn need(data: &[u8], len: usize) -> Result<(), CorruptKind> {
    if data.len() < len {
        return Err(CorruptKind::Truncated {
            needed: len64(len),
            available: len64(data.len()),
        });
    }
    Ok(())
}

fn zero_id(id: u32) -> Result<u32, CorruptKind> {
    if id == 0 {
        return Err(CorruptKind::InvalidValue {
            field: "catalog node ID",
            value: 0,
        });
    }
    Ok(id)
}

/// The type byte of a record, checked against the four known values.
fn record_type(data: &[u8]) -> Result<u8, CorruptKind> {
    let ty = u8_at(data, 0).ok_or(CorruptKind::Truncated {
        needed: 1,
        available: 0,
    })?;
    match ty {
        FOLDER_RECORD | FILE_RECORD | FOLDER_THREAD | FILE_THREAD => Ok(ty),
        _ => Err(CorruptKind::InvalidValue {
            field: "catalog record type",
            value: u64::from(ty),
        }),
    }
}

fn expect_type(data: &[u8], want: u8) -> Result<(), CorruptKind> {
    let ty = u8_at(data, 0).ok_or(CorruptKind::Truncated {
        needed: 1,
        available: 0,
    })?;
    if ty != want {
        return Err(CorruptKind::InvalidValue {
            field: "catalog record type",
            value: u64::from(ty),
        });
    }
    Ok(())
}

fn is_thread(ty: u8) -> bool {
    matches!(ty, FOLDER_THREAD | FILE_THREAD)
}

/// A catalog key read in place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogKey<'a> {
    /// The parent folder's id.
    pub parent_id: u32,
    /// The name as stored, in the volume's Mac OS script encoding.
    pub name: &'a [u8],
}

impl<'a> CatalogKey<'a> {
    /// Parses a key from the bytes after the key length byte: a reserved
    /// byte, the parent id, the name length (at most [`MAX_NAME_LEN`]) and
    /// the name. `None` when the key is too short or the name is too long
    /// or runs past the key. Bytes after the name are ignored (index keys
    /// are padded to [`INDEX_KEY_LEN`]).
    pub fn parse(key: &'a [u8]) -> Option<Self> {
        parse_key(key).ok()
    }
}

fn parse_key(key: &[u8]) -> Result<CatalogKey<'_>, CorruptKind> {
    let truncated = || CorruptKind::Truncated {
        needed: len64(KEY_HEADER_LEN),
        available: len64(key.len()),
    };
    let parent_id = be_u32(key, 1).ok_or_else(truncated)?;
    let len = usize::from(u8_at(key, 5).ok_or_else(truncated)?);
    if len > MAX_NAME_LEN {
        return Err(CorruptKind::OutOfRange {
            field: "key name length",
            value: len64(len),
            limit_name: "longest name",
            limit: len64(MAX_NAME_LEN),
        });
    }
    let name = bytes(key, KEY_HEADER_LEN, len).ok_or(CorruptKind::Truncated {
        needed: len64(KEY_HEADER_LEN.saturating_add(len)),
        available: len64(key.len()),
    })?;
    Ok(CatalogKey { parent_id, name })
}

/// A leaf record's key as it is stored: the key length byte, a reserved
/// byte, the parent id, the name length and the name. A name longer than
/// [`MAX_NAME_LEN`] is cut to that length; callers validate names first.
pub fn encode_key(parent_id: u32, name: &[u8]) -> Vec<u8> {
    let name = name.get(..MAX_NAME_LEN).unwrap_or(name);
    let mut out = Vec::with_capacity(KEY_HEADER_LEN.saturating_add(1).saturating_add(name.len()));
    out.push(u8::try_from(KEY_HEADER_LEN.saturating_add(name.len())).unwrap_or(u8::MAX));
    out.push(0);
    out.extend_from_slice(&parent_id.to_be_bytes());
    out.push(u8::try_from(name.len()).unwrap_or(0));
    out.extend_from_slice(name);
    out
}

/// Appends `record` as 12 bytes.
fn push_extents(out: &mut Vec<u8>, record: &HfsExtentRecord) {
    let mut buf = [0u8; EXTENT_RECORD_LEN];
    if write_extent_record(record, &mut buf).is_some() {
        out.extend_from_slice(&buf);
    }
}

/// A folder record (`cdrDirRec`, type 1, [`FOLDER_RECORD_LEN`] bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FolderRecord {
    /// `dirFlags`.
    pub flags: u16,
    /// `dirVal`: entries inside.
    pub valence: u16,
    /// `dirDirID`: the folder's id.
    pub folder_id: u32,
    /// `dirCrDat`, raw HFS seconds (0 is unset).
    pub created: u32,
    /// `dirMdDat`.
    pub modified: u32,
    /// `dirBkDat`.
    pub backed_up: u32,
    /// `dirUsrInfo` (`DInfo`): the Finder flags are the `u16` at +8.
    pub finder_info: [u8; 16],
    /// `dirFndrInfo` (`DXInfo`).
    pub ext_finder_info: [u8; 16],
    /// `dirResrv`, kept so a rewrite preserves it.
    pub reserved: [u8; 16],
}

impl FolderRecord {
    /// Parses a folder record from the record data. Bytes past the 70 of the
    /// record (a pad byte) are ignored.
    pub fn parse(d: &[u8]) -> Result<Self, CorruptKind> {
        expect_type(d, FOLDER_RECORD)?;
        need(d, FOLDER_RECORD_LEN)?;
        let record = (|| {
            Some(Self {
                flags: be_u16(d, 2)?,
                valence: be_u16(d, 4)?,
                folder_id: be_u32(d, 6)?,
                created: be_u32(d, 10)?,
                modified: be_u32(d, 14)?,
                backed_up: be_u32(d, 18)?,
                finder_info: array(d, 22)?,
                ext_finder_info: array(d, 38)?,
                reserved: array(d, 54)?,
            })
        })()
        .ok_or(CorruptKind::Truncated {
            needed: len64(FOLDER_RECORD_LEN),
            available: len64(d.len()),
        })?;
        zero_id(record.folder_id)?;
        Ok(record)
    }

    /// The record's 70 bytes, the inverse of [`parse`](Self::parse).
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(FOLDER_RECORD_LEN);
        out.extend_from_slice(&[FOLDER_RECORD, 0]);
        out.extend_from_slice(&self.flags.to_be_bytes());
        out.extend_from_slice(&self.valence.to_be_bytes());
        out.extend_from_slice(&self.folder_id.to_be_bytes());
        out.extend_from_slice(&self.created.to_be_bytes());
        out.extend_from_slice(&self.modified.to_be_bytes());
        out.extend_from_slice(&self.backed_up.to_be_bytes());
        out.extend_from_slice(&self.finder_info);
        out.extend_from_slice(&self.ext_finder_info);
        out.extend_from_slice(&self.reserved);
        out
    }

    /// The Finder flags of `DInfo`.
    pub fn finder_flags(&self) -> u16 {
        be_u16(&self.finder_info, FINDER_FLAGS_AT).unwrap_or(0)
    }
}

/// A file record (`cdrFilRec`, type 2, [`FILE_RECORD_LEN`] bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileRecord {
    /// `filFlags`; bit 0 is locked.
    pub flags: u8,
    /// `filTyp`, unused (0).
    pub file_kind: u8,
    /// `filUsrWds` (`FInfo`): type at +0, creator at +4, Finder flags at +8.
    pub finder_info: [u8; 16],
    /// `filFlNum`: the file's id.
    pub file_id: u32,
    /// `filStBlk`, unused.
    pub data_start_block: u16,
    /// `filLgLen`: data fork logical size.
    pub data_logical: u32,
    /// `filPyLen`: data fork physical size.
    pub data_physical: u32,
    /// `filRStBlk`, unused.
    pub rsrc_start_block: u16,
    /// `filRLgLen`: resource fork logical size.
    pub rsrc_logical: u32,
    /// `filRPyLen`: resource fork physical size.
    pub rsrc_physical: u32,
    /// `filCrDat`, raw HFS seconds (0 is unset).
    pub created: u32,
    /// `filMdDat`.
    pub modified: u32,
    /// `filBkDat`.
    pub backed_up: u32,
    /// `filFndrInfo` (`FXInfo`): the script byte is at +8.
    pub ext_finder_info: [u8; 16],
    /// `filClpSize`.
    pub clump_size: u16,
    /// `filExtRec`: the data fork's first three extents.
    pub data_extents: HfsExtentRecord,
    /// `filRExtRec`: the resource fork's first three extents.
    pub rsrc_extents: HfsExtentRecord,
    /// `filResrv`.
    pub reserved: u32,
}

impl FileRecord {
    /// Parses a file record from the record data. Bytes past the 102 of the
    /// record are ignored.
    pub fn parse(d: &[u8]) -> Result<Self, CorruptKind> {
        expect_type(d, FILE_RECORD)?;
        need(d, FILE_RECORD_LEN)?;
        let record = (|| {
            Some(Self {
                flags: u8_at(d, 2)?,
                file_kind: u8_at(d, 3)?,
                finder_info: array(d, 4)?,
                file_id: be_u32(d, 20)?,
                data_start_block: be_u16(d, 24)?,
                data_logical: be_u32(d, 26)?,
                data_physical: be_u32(d, 30)?,
                rsrc_start_block: be_u16(d, 34)?,
                rsrc_logical: be_u32(d, 36)?,
                rsrc_physical: be_u32(d, 40)?,
                created: be_u32(d, 44)?,
                modified: be_u32(d, 48)?,
                backed_up: be_u32(d, 52)?,
                ext_finder_info: array(d, 56)?,
                clump_size: be_u16(d, 72)?,
                data_extents: parse_extent_record(d.get(74..)?)?,
                rsrc_extents: parse_extent_record(d.get(86..)?)?,
                reserved: be_u32(d, 98)?,
            })
        })()
        .ok_or(CorruptKind::Truncated {
            needed: len64(FILE_RECORD_LEN),
            available: len64(d.len()),
        })?;
        zero_id(record.file_id)?;
        Ok(record)
    }

    /// The record's 102 bytes, the inverse of [`parse`](Self::parse).
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(FILE_RECORD_LEN);
        out.extend_from_slice(&[FILE_RECORD, 0, self.flags, self.file_kind]);
        out.extend_from_slice(&self.finder_info);
        out.extend_from_slice(&self.file_id.to_be_bytes());
        out.extend_from_slice(&self.data_start_block.to_be_bytes());
        out.extend_from_slice(&self.data_logical.to_be_bytes());
        out.extend_from_slice(&self.data_physical.to_be_bytes());
        out.extend_from_slice(&self.rsrc_start_block.to_be_bytes());
        out.extend_from_slice(&self.rsrc_logical.to_be_bytes());
        out.extend_from_slice(&self.rsrc_physical.to_be_bytes());
        out.extend_from_slice(&self.created.to_be_bytes());
        out.extend_from_slice(&self.modified.to_be_bytes());
        out.extend_from_slice(&self.backed_up.to_be_bytes());
        out.extend_from_slice(&self.ext_finder_info);
        out.extend_from_slice(&self.clump_size.to_be_bytes());
        push_extents(&mut out, &self.data_extents);
        push_extents(&mut out, &self.rsrc_extents);
        out.extend_from_slice(&self.reserved.to_be_bytes());
        out
    }

    /// The Finder file type (`FInfo.fdType`), four zero bytes when unset.
    pub fn file_type(&self) -> [u8; 4] {
        array(&self.finder_info, 0).unwrap_or([0; 4])
    }

    /// The Finder creator code (`FInfo.fdCreator`).
    pub fn creator(&self) -> [u8; 4] {
        array(&self.finder_info, 4).unwrap_or([0; 4])
    }

    /// The Finder flags of `FInfo`.
    pub fn finder_flags(&self) -> u16 {
        be_u16(&self.finder_info, FINDER_FLAGS_AT).unwrap_or(0)
    }

    /// True when the file's locked flag is set.
    pub fn locked(&self) -> bool {
        self.flags & 1 != 0
    }
}

/// Which kind of node a thread record belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadKind {
    /// A folder thread (type 3).
    Folder,
    /// A file thread (type 4).
    File,
}

/// A thread record: the parent and name of a node, keyed by `(its own id, "")`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadRecord {
    /// Folder or file thread.
    pub kind: ThreadKind,
    /// `thdResrv`, kept so a rewrite preserves it.
    pub reserved: [u8; 8],
    /// The node's parent folder.
    pub parent_id: u32,
    /// The node's name as stored.
    pub name: Vec<u8>,
}

impl ThreadRecord {
    /// Parses a thread record (type 3 or 4). The name must lie inside the
    /// data and be at most [`MAX_NAME_LEN`] bytes; the record may be shorter
    /// than [`THREAD_RECORD_LEN`] when the name is.
    pub fn parse(d: &[u8]) -> Result<Self, CorruptKind> {
        let kind = match record_type(d)? {
            FOLDER_THREAD => ThreadKind::Folder,
            FILE_THREAD => ThreadKind::File,
            other => {
                return Err(CorruptKind::InvalidValue {
                    field: "catalog record type",
                    value: u64::from(other),
                });
            }
        };
        let short = CorruptKind::Truncated {
            needed: len64(THREAD_NAME_AT.saturating_add(1)),
            available: len64(d.len()),
        };
        let reserved = array(d, 2).ok_or(short.clone())?;
        let parent_id = be_u32(d, THREAD_PARENT_AT).ok_or(short.clone())?;
        let len = usize::from(u8_at(d, THREAD_NAME_AT).ok_or(short)?);
        if len > MAX_NAME_LEN {
            return Err(CorruptKind::OutOfRange {
                field: "thread name length",
                value: len64(len),
                limit_name: "longest name",
                limit: len64(MAX_NAME_LEN),
            });
        }
        let name_at = THREAD_NAME_AT.saturating_add(1);
        let name = bytes(d, name_at, len).ok_or(CorruptKind::Truncated {
            needed: len64(name_at.saturating_add(len)),
            available: len64(d.len()),
        })?;
        Ok(Self {
            kind,
            reserved,
            parent_id,
            name: name.to_vec(),
        })
    }

    /// The record's [`THREAD_RECORD_LEN`] bytes, the inverse of
    /// [`parse`](Self::parse). A name longer than [`MAX_NAME_LEN`] is cut.
    pub fn encode(&self) -> Vec<u8> {
        self.encode_as(self.kind)
    }

    fn encode_as(&self, kind: ThreadKind) -> Vec<u8> {
        let name = self.name.get(..MAX_NAME_LEN).unwrap_or(&self.name);
        let ty = match kind {
            ThreadKind::Folder => FOLDER_THREAD,
            ThreadKind::File => FILE_THREAD,
        };
        let mut out = Vec::with_capacity(THREAD_RECORD_LEN);
        out.extend_from_slice(&[ty, 0]);
        out.extend_from_slice(&self.reserved);
        out.extend_from_slice(&self.parent_id.to_be_bytes());
        out.push(u8::try_from(name.len()).unwrap_or(0));
        out.extend_from_slice(name);
        out.resize(THREAD_RECORD_LEN, 0);
        out
    }
}

/// One catalog leaf record's data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CatalogRecord {
    /// A folder.
    Folder(FolderRecord),
    /// A file.
    File(FileRecord),
    /// A folder's thread.
    FolderThread(ThreadRecord),
    /// A file's thread.
    FileThread(ThreadRecord),
}

impl CatalogRecord {
    /// Parses record data of any of the four types. An unknown type byte, a
    /// record shorter than its type needs and a name longer than 31 bytes in
    /// a thread are `Corrupt`.
    pub fn parse(d: &[u8]) -> Result<Self, CorruptKind> {
        match record_type(d)? {
            FOLDER_RECORD => FolderRecord::parse(d).map(Self::Folder),
            FILE_RECORD => FileRecord::parse(d).map(Self::File),
            FOLDER_THREAD => ThreadRecord::parse(d).map(Self::FolderThread),
            _ => ThreadRecord::parse(d).map(Self::FileThread),
        }
    }

    /// The record's bytes, the inverse of [`parse`](Self::parse). The type
    /// byte follows the variant.
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Self::Folder(f) => f.encode(),
            Self::File(f) => f.encode(),
            Self::FolderThread(t) => t.encode_as(ThreadKind::Folder),
            Self::FileThread(t) => t.encode_as(ThreadKind::File),
        }
    }

    /// The folder's or file's id; `None` for a thread.
    pub fn id(&self) -> Option<u32> {
        match self {
            Self::Folder(f) => Some(f.folder_id),
            Self::File(f) => Some(f.file_id),
            Self::FolderThread(_) | Self::FileThread(_) => None,
        }
    }

    /// The Finder flags of a folder or file; `None` for a thread.
    pub fn finder_flags(&self) -> Option<u16> {
        match self {
            Self::Folder(f) => Some(f.finder_flags()),
            Self::File(f) => Some(f.finder_flags()),
            Self::FolderThread(_) | Self::FileThread(_) => None,
        }
    }
}

/// CNID to parent id, filled by listings, for files that have no thread
/// record and so cannot be found by id from nothing. Capped at
/// [`PARENT_MAP_CAP`] entries (or the cap given to
/// [`with_cap`](Self::with_cap)): when it is full, it is cleared, then the
/// new entry goes in.
#[derive(Debug)]
pub struct ParentMap {
    map: Mutex<HashMap<u32, u32>>,
    cap: usize,
}

impl Default for ParentMap {
    fn default() -> Self {
        Self::new()
    }
}

impl ParentMap {
    /// An empty map with the default cap.
    pub fn new() -> Self {
        Self::with_cap(PARENT_MAP_CAP)
    }

    /// An empty map that holds at most `cap` entries (at least 1).
    pub fn with_cap(cap: usize) -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
            cap: cap.max(1),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<u32, u32>> {
        self.map.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Records that `id` lives in folder `parent`.
    pub fn insert(&self, id: u32, parent: u32) {
        self.insert_all(&[(id, parent)]);
    }

    /// Records every `(id, parent)` pair under one lock.
    pub fn insert_all(&self, pairs: &[(u32, u32)]) {
        let mut map = self.lock();
        for &(id, parent) in pairs {
            if map.len() >= self.cap && !map.contains_key(&id) {
                map.clear();
            }
            map.insert(id, parent);
        }
    }

    /// The parent recorded for `id`.
    pub fn get(&self, id: u32) -> Option<u32> {
        self.lock().get(&id).copied()
    }

    /// Entries held.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// True when no entry is held.
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }
}

/// What one record of a walk along a parent's records means.
enum Walk<T> {
    /// The key belongs to another parent: the walk is over.
    End,
    /// A thread record.
    Skip,
    /// A folder or file record, mapped by the caller.
    Entry(T),
}

/// Reads one record of a walk that started at `(parent, "")`. For a folder
/// or file record of `parent`, `f` gets the name bytes, the id and the parsed
/// record; its result is returned in `Walk::Entry`.
fn walk_step<T>(
    rec: Record<'_>,
    parent: u32,
    f: impl FnOnce(&[u8], u32, CatalogRecord) -> T,
) -> Result<Walk<T>, CorruptKind> {
    let key = parse_key(rec.key)?;
    if key.parent_id != parent {
        return Ok(Walk::End);
    }
    let ty = record_type(rec.data)?;
    if is_thread(ty) {
        return Ok(Walk::Skip);
    }
    let (id, record) = if ty == FOLDER_RECORD {
        let folder = FolderRecord::parse(rec.data)?;
        (folder.folder_id, CatalogRecord::Folder(folder))
    } else {
        let file = FileRecord::parse(rec.data)?;
        (file.file_id, CatalogRecord::File(file))
    };
    Ok(Walk::Entry(f(key.name, id, record)))
}

/// Decision 4: equal when every character matches after lowercasing.
/// Case-insensitive and accent-sensitive.
fn names_equal(a: &str, b: &str) -> bool {
    a.chars()
        .flat_map(char::to_lowercase)
        .eq(b.chars().flat_map(char::to_lowercase))
}

/// The classic HFS catalog. Clone is cheap.
#[derive(Clone)]
pub struct Catalog {
    tree: BTree,
    encoding: Encoding,
}

impl Catalog {
    /// A catalog read from `tree`, with names in `encoding`.
    pub fn new(tree: BTree, encoding: Encoding) -> Self {
        Self { tree, encoding }
    }

    /// The underlying B-tree (a scan walks its leaves).
    pub fn tree(&self) -> &BTree {
        &self.tree
    }

    /// The encoding names are decoded with.
    pub fn encoding(&self) -> Encoding {
        self.encoding
    }

    /// A cursor at the first record of `parent_id`: the first leaf record
    /// whose key is at least `(parent_id, "")`. The comparator knows no name
    /// order: an empty name is equal to the target, any other is greater.
    fn seek_parent(&self, parent_id: u32) -> Result<LeafCursor> {
        // The tree hands the comparator key bytes only, so a bad key here
        // cannot say which record it was.
        let mut cmp = |key: &[u8]| -> Result<Ordering> {
            let k = parse_key(key).map_err(|kind| corrupt(Location::Unspecified, kind))?;
            Ok(k.parent_id.cmp(&parent_id).then(if k.name.is_empty() {
                Ordering::Equal
            } else {
                Ordering::Greater
            }))
        };
        self.tree.seek(&mut cmp)
    }

    /// The thread record of `id`; `Ok(None)` when there is none.
    pub fn thread(&self, id: u32) -> Result<Option<ThreadRecord>> {
        let mut cursor = self.seek_parent(id)?;
        let parsed = match cursor.next_record()? {
            None => return Ok(None),
            Some(rec) => thread_of(rec, id),
        };
        parsed.map_err(|kind| corrupt(record_location(&cursor), kind))
    }

    /// Opens folder `folder_id` for listing.
    ///
    /// Fails with `Error::NotADirectory` when `folder_id` has a file thread
    /// and `Error::NotFound` when it has no thread record.
    pub fn list(&self, folder_id: u32) -> Result<HfsDir> {
        self.open_list(folder_id, None)
    }

    /// Same as [`list`](Self::list); every entry the stream yields is also
    /// recorded in `parents` (its CNID and `folder_id`).
    pub fn list_tracked(&self, folder_id: u32, parents: Arc<ParentMap>) -> Result<HfsDir> {
        self.open_list(folder_id, Some(parents))
    }

    fn open_list(&self, folder_id: u32, parents: Option<Arc<ParentMap>>) -> Result<HfsDir> {
        let mut cursor = self.seek_parent(folder_id)?;
        let first = match cursor.next_record()? {
            None => {
                return Err(Error::NotFound {
                    node: u64::from(folder_id),
                });
            }
            Some(rec) => (|| {
                let key = parse_key(rec.key)?;
                if key.parent_id != folder_id || !key.name.is_empty() {
                    return Ok(None);
                }
                record_type(rec.data).map(Some)
            })(),
        };
        match first {
            Ok(Some(FOLDER_THREAD)) => Ok(HfsDir {
                cursor,
                folder_id,
                encoding: self.encoding,
                parents,
                pending: Vec::new(),
                name: String::new(),
                done: false,
            }),
            Ok(Some(FILE_THREAD)) => Err(Error::NotADirectory {
                node: u64::from(folder_id),
            }),
            Ok(_) => Err(Error::NotFound {
                node: u64::from(folder_id),
            }),
            Err(kind) => Err(corrupt(record_location(&cursor), kind)),
        }
    }

    /// The folder or file record `id` of folder `parent`, found by walking
    /// the parent's records from `(parent, "")`. `Ok(None)` when `parent`
    /// holds no such id. Never returns a thread variant.
    pub fn record_in(&self, parent: u32, id: u32) -> Result<Option<CatalogRecord>> {
        let mut cursor = self.seek_parent(parent)?;
        // Bounded by the cursor, which walks the leaf chain at most once.
        loop {
            let step = match cursor.next_record()? {
                None => return Ok(None),
                Some(rec) => walk_step(rec, parent, |_, found, record| {
                    (found == id).then_some(record)
                }),
            };
            match step {
                Ok(Walk::Entry(Some(record))) => return Ok(Some(record)),
                Ok(Walk::Entry(None) | Walk::Skip) => {}
                Ok(Walk::End) => return Ok(None),
                Err(kind) => return Err(corrupt(record_location(&cursor), kind)),
            }
        }
    }

    /// The folder or file record of `id` and its parent's id, through the
    /// thread record. `Ok(None)` when `id` has no thread (a file without one
    /// is reachable only through its folder: see [`record_in`](Self::record_in)).
    /// A thread whose record is missing or of the other kind is `Corrupt`.
    /// Never returns a thread variant.
    pub fn record(&self, id: u32) -> Result<Option<(u32, CatalogRecord)>> {
        let Some(thread) = self.thread(id)? else {
            return Ok(None);
        };
        let bad =
            |text: &'static str| corrupt(Location::Object(u64::from(id)), CorruptKind::Other(text));
        let Some(record) = self.record_in(thread.parent_id, id)? else {
            return Err(bad("thread names a record that does not exist"));
        };
        let kind_matches = matches!(
            (&record, thread.kind),
            (CatalogRecord::Folder(_), ThreadKind::Folder)
                | (CatalogRecord::File(_), ThreadKind::File)
        );
        if !kind_matches {
            return Err(bad("thread kind does not match its record"));
        }
        Ok(Some((thread.parent_id, record)))
    }

    /// The entry of folder `parent` called `name`: its id and record.
    /// Scans the folder; names are decoded and compared character by
    /// character, lowercased (case-insensitive, accent-sensitive).
    /// Never returns a thread variant.
    pub fn find(&self, parent: u32, name: &str) -> Result<Option<(u32, CatalogRecord)>> {
        let mut cursor = self.seek_parent(parent)?;
        let mut decoded = String::new();
        let encoding = self.encoding;
        // Bounded by the cursor, which walks the leaf chain at most once.
        loop {
            let step = match cursor.next_record()? {
                None => return Ok(None),
                Some(rec) => walk_step(rec, parent, |stored, id, record| {
                    decoded.clear();
                    encoding.decode_into(stored, &mut decoded);
                    names_equal(&decoded, name).then_some((id, record))
                }),
            };
            match step {
                Ok(Walk::Entry(Some(found))) => return Ok(Some(found)),
                Ok(Walk::Entry(None) | Walk::Skip) => {}
                Ok(Walk::End) => return Ok(None),
                Err(kind) => return Err(corrupt(record_location(&cursor), kind)),
            }
        }
    }

    /// The volume name: the name in the root folder's thread. `Ok(None)`
    /// when the root has no thread; a root thread that is not a folder
    /// thread, or whose parent is not 1, is `Corrupt`.
    pub fn volume_name(&self) -> Result<Option<String>> {
        let Some(thread) = self.thread(ROOT_FOLDER_ID)? else {
            return Ok(None);
        };
        if thread.kind != ThreadKind::Folder || thread.parent_id != ROOT_PARENT_ID {
            return Err(corrupt(
                Location::Object(u64::from(ROOT_FOLDER_ID)),
                CorruptKind::Other("the root folder's thread record is not a root folder thread"),
            ));
        }
        let mut name = String::new();
        self.encoding.decode_into(&thread.name, &mut name);
        Ok(Some(name))
    }
}

/// The thread of `id` if `rec` is it.
fn thread_of(rec: Record<'_>, id: u32) -> Result<Option<ThreadRecord>, CorruptKind> {
    let key = parse_key(rec.key)?;
    if key.parent_id != id || !key.name.is_empty() {
        return Ok(None);
    }
    if !is_thread(record_type(rec.data)?) {
        return Ok(None);
    }
    let thread = ThreadRecord::parse(rec.data)?;
    if thread.parent_id == 0 {
        return Err(CorruptKind::InvalidValue {
            field: "thread parent ID",
            value: 0,
        });
    }
    Ok(Some(thread))
}

/// The fixed-size facts of one listed entry; the name is in `HfsDir::name`.
struct Facts {
    node: u32,
    kind: NodeKind,
    size: Option<u64>,
    resource_size: Option<u64>,
    modified: u32,
    finder_flags: u16,
}

fn facts_of(id: u32, record: &CatalogRecord) -> Option<Facts> {
    match record {
        CatalogRecord::Folder(f) => Some(Facts {
            node: id,
            kind: NodeKind::Directory,
            size: None,
            resource_size: None,
            modified: f.modified,
            finder_flags: f.finder_flags(),
        }),
        CatalogRecord::File(f) => Some(Facts {
            node: id,
            kind: NodeKind::File,
            size: Some(u64::from(f.data_logical)),
            resource_size: Some(u64::from(f.rsrc_logical)),
            modified: f.modified,
            finder_flags: f.finder_flags(),
        }),
        CatalogRecord::FolderThread(_) | CatalogRecord::FileThread(_) => None,
    }
}

/// A folder listing. Owns a [`LeafCursor`], so it is `Send` and does not
/// borrow the [`Catalog`]. Entries carry everything the listing shows; the
/// name is decoded with the catalog's encoding.
pub struct HfsDir {
    cursor: LeafCursor,
    folder_id: u32,
    encoding: Encoding,
    /// Where listed entries are recorded, when the caller wants that.
    parents: Option<Arc<ParentMap>>,
    /// `(id, folder id)` pairs of the current batch, flushed once per call.
    pending: Vec<(u32, u32)>,
    /// The current entry's display name, reused across entries.
    name: String,
    done: bool,
}

impl HfsDir {
    /// One step of the walk; errors finish the stream.
    fn step(&mut self) -> Result<Walk<Option<Facts>>> {
        let step = match self.cursor.next_record()? {
            None => return Ok(Walk::End),
            Some(rec) => {
                let (name, encoding) = (&mut self.name, self.encoding);
                walk_step(rec, self.folder_id, |stored, id, record| {
                    name.clear();
                    encoding.decode_into(stored, name);
                    facts_of(id, &record)
                })
            }
        };
        step.map_err(|kind| corrupt(record_location(&self.cursor), kind))
    }

    fn flush(&mut self) {
        if let Some(parents) = &self.parents {
            parents.insert_all(&self.pending);
        }
        self.pending.clear();
    }
}

impl DirStream for HfsDir {
    fn next_batch(&mut self, out: &mut DirBatch, max: usize) -> Result<DirProgress> {
        if self.done {
            return Ok(DirProgress::Done);
        }
        let mut added = 0usize;
        let mut result = Ok(DirProgress::More);
        // Bounded: every turn takes one record from the cursor, which ends.
        while added < max {
            match self.step() {
                Ok(Walk::Entry(Some(f))) => {
                    out.push(DirEntry {
                        node: NodeId(u64::from(f.node)),
                        name: &self.name,
                        kind: f.kind,
                        size: f.size,
                        resource_size: f.resource_size,
                        modified: hfs_time(f.modified),
                        flags: entry_flags(false, f.finder_flags),
                    });
                    if self.parents.is_some() {
                        self.pending.push((f.node, self.folder_id));
                    }
                    added = added.saturating_add(1);
                }
                Ok(Walk::Entry(None) | Walk::Skip) => {}
                Ok(Walk::End) => {
                    self.done = true;
                    result = Ok(DirProgress::Done);
                    break;
                }
                Err(e) => {
                    self.done = true;
                    result = Err(e);
                    break;
                }
            }
        }
        self.flush();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extents::HfsExtent;
    use crate::test_volume::{self, FileSpec, VolumeBuilder, ids, open_catalog};
    use atlas_fs_core::fs::EntryFlags;

    // ---- parsers ----

    fn key_bytes(parent: u32, name: &[u8]) -> Vec<u8> {
        // Without the key length byte, as the B-tree hands it over.
        encode_key(parent, name).split_off(1)
    }

    #[test]
    fn key_parses_and_round_trips() {
        let key = key_bytes(0x0102_0304, b"Hello");
        let k = CatalogKey::parse(&key).unwrap();
        assert_eq!(k.parent_id, 0x0102_0304);
        assert_eq!(k.name, b"Hello");
        assert_eq!(encode_key(7, b"ab"), [8, 0, 0, 0, 0, 7, 2, b'a', b'b']);
        let bytes = key_bytes(5, b"");
        let empty = CatalogKey::parse(&bytes).unwrap();
        assert_eq!((empty.parent_id, empty.name), (5, &b""[..]));
    }

    #[test]
    fn key_ignores_index_padding() {
        let mut key = key_bytes(9, b"abc");
        key.resize(INDEX_KEY_LEN, 0xEE);
        assert_eq!(CatalogKey::parse(&key).unwrap().name, b"abc");
        let longest = [b'x'; MAX_NAME_LEN];
        assert_eq!(
            CatalogKey::parse(&key_bytes(9, &longest)).unwrap().name,
            &longest[..]
        );
    }

    #[test]
    fn key_rejects_malformed_input() {
        // Key length 0: nothing after the length byte.
        assert_eq!(CatalogKey::parse(&[]), None);
        assert_eq!(CatalogKey::parse(&[0, 0, 0, 0, 1]), None);
        // Name length 32.
        let mut long = key_bytes(9, &[b'x'; MAX_NAME_LEN]);
        long.push(b'y');
        long[5] = 32;
        assert_eq!(CatalogKey::parse(&long), None);
        // The name runs past the key.
        let mut cut = key_bytes(9, b"abcdef");
        cut.truncate(cut.len() - 1);
        assert_eq!(CatalogKey::parse(&cut), None);
        // 0xFF name length.
        assert_eq!(CatalogKey::parse(&[0, 0, 0, 0, 1, 0xFF, b'a']), None);
    }

    fn sample_folder() -> FolderRecord {
        FolderRecord {
            flags: 0x0102,
            valence: 7,
            folder_id: 0x1234_5678,
            created: 1,
            modified: 2,
            backed_up: 3,
            finder_info: core::array::from_fn(|i| i as u8),
            ext_finder_info: core::array::from_fn(|i| 0x40 + i as u8),
            reserved: core::array::from_fn(|i| 0x80 + i as u8),
        }
    }

    fn ext(start_block: u16, block_count: u16) -> HfsExtent {
        HfsExtent {
            start_block,
            block_count,
        }
    }

    fn sample_file() -> FileRecord {
        let mut finder_info: [u8; 16] = core::array::from_fn(|i| 0x10 + i as u8);
        finder_info[..4].copy_from_slice(b"TEXT");
        finder_info[4..8].copy_from_slice(b"ttxt");
        finder_info[8..10].copy_from_slice(&0x4000u16.to_be_bytes());
        FileRecord {
            flags: 1,
            file_kind: 0,
            finder_info,
            file_id: 77,
            data_start_block: 5,
            data_logical: 1000,
            data_physical: 1024,
            rsrc_start_block: 9,
            rsrc_logical: 50,
            rsrc_physical: 512,
            created: 11,
            modified: 12,
            backed_up: 13,
            ext_finder_info: core::array::from_fn(|i| 0x70 + i as u8),
            clump_size: 0x0400,
            data_extents: [ext(5, 2), ext(20, 1), ext(0, 0)],
            rsrc_extents: [ext(9, 1), ext(0, 0), ext(0, 0)],
            reserved: 0xDEAD_BEEF,
        }
    }

    fn sample_thread(kind: ThreadKind) -> ThreadRecord {
        ThreadRecord {
            kind,
            reserved: [1, 2, 3, 4, 5, 6, 7, 8],
            parent_id: 16,
            name: b"A name".to_vec(),
        }
    }

    #[test]
    fn folder_record_round_trips() {
        let f = sample_folder();
        let bytes = f.encode();
        assert_eq!(bytes.len(), FOLDER_RECORD_LEN);
        assert_eq!(&bytes[..2], &[1, 0]);
        assert_eq!(&bytes[6..10], &f.folder_id.to_be_bytes());
        assert_eq!(FolderRecord::parse(&bytes), Ok(f));
        // A pad byte after the record is ignored.
        let mut padded = bytes;
        padded.push(0);
        assert_eq!(FolderRecord::parse(&padded), Ok(f));
        assert_eq!(f.finder_flags(), u16::from_be_bytes([8, 9]));
    }

    #[test]
    fn folder_record_rejects_malformed_input() {
        let bytes = sample_folder().encode();
        assert!(matches!(
            FolderRecord::parse(&bytes[..FOLDER_RECORD_LEN - 1]),
            Err(CorruptKind::Truncated {
                needed: 70,
                available: 69
            })
        ));
        assert!(matches!(
            FolderRecord::parse(&[]),
            Err(CorruptKind::Truncated { .. })
        ));
        let mut wrong = bytes.clone();
        wrong[0] = FILE_RECORD;
        assert!(matches!(
            FolderRecord::parse(&wrong),
            Err(CorruptKind::InvalidValue { value: 2, .. })
        ));
        let mut zero = bytes;
        zero[6..10].fill(0);
        assert!(matches!(
            FolderRecord::parse(&zero),
            Err(CorruptKind::InvalidValue {
                field: "catalog node ID",
                ..
            })
        ));
    }

    #[test]
    fn file_record_round_trips() {
        let f = sample_file();
        let bytes = f.encode();
        assert_eq!(bytes.len(), FILE_RECORD_LEN);
        assert_eq!(&bytes[..4], &[2, 0, 1, 0]);
        assert_eq!(&bytes[20..24], &77u32.to_be_bytes());
        assert_eq!(&bytes[74..78], &[0, 5, 0, 2]);
        assert_eq!(FileRecord::parse(&bytes), Ok(f));
        assert_eq!(f.file_type(), *b"TEXT");
        assert_eq!(f.creator(), *b"ttxt");
        assert_eq!(f.finder_flags(), 0x4000);
        assert!(f.locked());
    }

    #[test]
    fn file_record_rejects_malformed_input() {
        let bytes = sample_file().encode();
        for len in [0, 1, 50, FILE_RECORD_LEN - 1] {
            assert!(
                matches!(
                    FileRecord::parse(&bytes[..len]),
                    Err(CorruptKind::Truncated { .. })
                ),
                "length {len}"
            );
        }
        let mut wrong = bytes.clone();
        wrong[0] = FOLDER_RECORD;
        assert!(FileRecord::parse(&wrong).is_err());
        let mut zero = bytes;
        zero[20..24].fill(0);
        assert!(FileRecord::parse(&zero).is_err());
    }

    #[test]
    fn thread_record_round_trips() {
        for kind in [ThreadKind::Folder, ThreadKind::File] {
            let t = sample_thread(kind);
            let bytes = t.encode();
            assert_eq!(bytes.len(), THREAD_RECORD_LEN);
            assert_eq!(bytes[0], if kind == ThreadKind::Folder { 3 } else { 4 });
            assert_eq!(&bytes[10..14], &16u32.to_be_bytes());
            assert_eq!(bytes[14], 6);
            assert_eq!(ThreadRecord::parse(&bytes), Ok(t));
        }
        // The longest name.
        let long = ThreadRecord {
            name: vec![b'n'; MAX_NAME_LEN],
            ..sample_thread(ThreadKind::File)
        };
        assert_eq!(ThreadRecord::parse(&long.encode()), Ok(long));
    }

    #[test]
    fn thread_record_rejects_malformed_input() {
        let bytes = sample_thread(ThreadKind::Folder).encode();
        // Name length 32.
        let mut long = bytes.clone();
        long[14] = 32;
        assert!(matches!(
            ThreadRecord::parse(&long),
            Err(CorruptKind::OutOfRange {
                field: "thread name length",
                value: 32,
                ..
            })
        ));
        // The name runs past the record.
        assert!(matches!(
            ThreadRecord::parse(&bytes[..18]),
            Err(CorruptKind::Truncated { .. })
        ));
        // No room for the header.
        assert!(matches!(
            ThreadRecord::parse(&bytes[..14]),
            Err(CorruptKind::Truncated { .. })
        ));
        // A folder record is not a thread.
        assert!(ThreadRecord::parse(&sample_folder().encode()).is_err());
    }

    #[test]
    fn catalog_record_dispatches_on_the_type_byte() {
        let records = [
            CatalogRecord::Folder(sample_folder()),
            CatalogRecord::File(sample_file()),
            CatalogRecord::FolderThread(sample_thread(ThreadKind::Folder)),
            CatalogRecord::FileThread(sample_thread(ThreadKind::File)),
        ];
        for r in &records {
            assert_eq!(CatalogRecord::parse(&r.encode()).as_ref(), Ok(r));
        }
        assert_eq!(records[0].id(), Some(0x1234_5678));
        assert_eq!(records[1].id(), Some(77));
        assert_eq!(records[2].id(), None);
        assert_eq!(records[1].finder_flags(), Some(0x4000));
        for ty in [0u8, 5, 9, 0xFF] {
            let mut bytes = sample_file().encode();
            bytes[0] = ty;
            assert!(
                matches!(
                    CatalogRecord::parse(&bytes),
                    Err(CorruptKind::InvalidValue {
                        field: "catalog record type",
                        ..
                    })
                ),
                "type {ty}"
            );
        }
        assert!(matches!(
            CatalogRecord::parse(&[]),
            Err(CorruptKind::Truncated { .. })
        ));
    }

    // ---- against the builder ----

    /// One listed entry, owned.
    #[derive(Debug, Clone, PartialEq)]
    struct Listed {
        id: u64,
        name: String,
        kind: NodeKind,
        size: Option<u64>,
        resource_size: Option<u64>,
        modified: Option<atlas_fs_core::fs::Timestamp>,
        flags: EntryFlags,
    }

    /// Everything a stream yields in batches of `max`, and its error.
    fn drain(mut dir: HfsDir, max: usize) -> (Vec<Listed>, Option<Error>) {
        let mut out = Vec::new();
        let mut batch = DirBatch::new();
        loop {
            batch.clear();
            let progress = dir.next_batch(&mut batch, max);
            out.extend(batch.iter().map(|e| Listed {
                id: e.node.0,
                name: e.name.to_owned(),
                kind: e.kind,
                size: e.size,
                resource_size: e.resource_size,
                modified: e.modified,
                flags: e.flags,
            }));
            match progress {
                Ok(DirProgress::More) => {}
                Ok(DirProgress::Done) => return (out, None),
                Err(e) => return (out, Some(e)),
            }
        }
    }

    fn list_all(catalog: &Catalog, folder: u32) -> Vec<Listed> {
        let (entries, err) = drain(catalog.list(folder).unwrap(), 4);
        assert!(err.is_none(), "{err:?}");
        entries
    }

    fn standard() -> (Catalog, test_volume::Manifest) {
        let built = test_volume::standard_volume();
        (open_catalog(built.image), built.manifest)
    }

    #[test]
    fn standard_catalog_has_index_nodes_and_an_overflowing_extent_list() {
        let (catalog, manifest) = standard();
        assert!(catalog.tree().header().depth >= 2);
        assert!(manifest.catalog_extents.len() > 3);
    }

    #[test]
    fn lists_the_root_and_nested_folders() {
        let (catalog, manifest) = standard();
        for folder in [ROOT_FOLDER_ID, ids::DOCUMENTS, ids::LETTERS, ids::Y1991] {
            let mut listed = list_all(&catalog, folder);
            let mut want: Vec<&test_volume::Entry> = manifest.children(folder).collect();
            assert!(!want.is_empty());
            assert_eq!(listed.len(), want.len(), "folder {folder}");
            want.sort_by_key(|e| e.id);
            listed.sort_by_key(|e| e.id);
            for (l, e) in listed.iter().zip(want) {
                assert_eq!(l.id, u64::from(e.id));
                assert_eq!(l.name, e.name);
                assert_eq!(
                    l.kind,
                    if e.is_dir {
                        NodeKind::Directory
                    } else {
                        NodeKind::File
                    }
                );
                assert_eq!(l.size, (!e.is_dir).then_some(e.data_len), "{}", e.path);
                assert_eq!(
                    l.resource_size,
                    (!e.is_dir).then_some(e.rsrc_len),
                    "{}",
                    e.path
                );
                assert_eq!(l.modified, hfs_time(e.modified));
            }
        }
        let names: Vec<String> = list_all(&catalog, ids::Y1991)
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert_eq!(names, ["Dear John"]);
        assert!(list_all(&catalog, ids::EMPTY).is_empty());
    }

    #[test]
    fn listing_in_small_batches_gives_the_same_entries() {
        let (catalog, _) = standard();
        let whole = drain(catalog.list(ROOT_FOLDER_ID).unwrap(), 1000).0;
        for max in [1, 2, 3, 5] {
            let (parts, err) = drain(catalog.list(ROOT_FOLDER_ID).unwrap(), max);
            assert!(err.is_none());
            assert_eq!(parts, whole, "batch size {max}");
        }
        // A zero-sized batch adds nothing and does not end the listing.
        let mut dir = catalog.list(ROOT_FOLDER_ID).unwrap();
        let mut batch = DirBatch::new();
        assert_eq!(dir.next_batch(&mut batch, 0).unwrap(), DirProgress::More);
        assert!(batch.is_empty());
    }

    #[test]
    fn listing_flags_follow_the_finder_flags_only() {
        let (catalog, _) = standard();
        let root = list_all(&catalog, ROOT_FOLDER_ID);
        let flags = |name: &str| root.iter().find(|e| e.name == name).unwrap().flags;
        assert!(flags("Desktop").contains(EntryFlags::HIDDEN));
        // No dot-file convention on classic Mac OS.
        assert!(!flags(".profile").contains(EntryFlags::HIDDEN));
        assert!(!flags("Fragmented").contains(EntryFlags::HIDDEN));
        assert!(flags("SimpleText").contains(EntryFlags::PACKAGE));
        let docs = list_all(&catalog, ids::DOCUMENTS);
        let alias = docs.iter().find(|e| e.name == "ReadMe alias").unwrap();
        assert!(alias.flags.contains(EntryFlags::ALIAS));
        // An alias is an ordinary file.
        assert_eq!(alias.kind, NodeKind::File);
    }

    fn list_error(catalog: &Catalog, id: u32) -> Error {
        match catalog.list(id) {
            Err(e) => e,
            Ok(_) => panic!("listing {id} should fail"),
        }
    }

    #[test]
    fn listing_a_file_or_an_unknown_id_fails() {
        let (catalog, _) = standard();
        // A file with a thread is known to be a file.
        assert!(matches!(
            list_error(&catalog, ids::DEAR_JOHN),
            Error::NotADirectory { node } if node == u64::from(ids::DEAR_JOHN)
        ));
        // A file without one cannot be told from a missing id.
        for id in [ids::README, 9999, 0, 1, u32::MAX] {
            assert!(
                matches!(list_error(&catalog, id), Error::NotFound { node } if node == u64::from(id)),
                "id {id}"
            );
        }
    }

    #[test]
    fn thread_finds_threads_only() {
        let (catalog, _) = standard();
        let t = catalog.thread(ids::LETTERS).unwrap().unwrap();
        assert_eq!(t.kind, ThreadKind::Folder);
        assert_eq!(t.parent_id, ids::DOCUMENTS);
        assert_eq!(t.name, b"Letters");
        let t = catalog.thread(ids::DEAR_JOHN).unwrap().unwrap();
        assert_eq!(t.kind, ThreadKind::File);
        assert_eq!(t.parent_id, ids::Y1991);
        let root = catalog.thread(ROOT_FOLDER_ID).unwrap().unwrap();
        assert_eq!((root.kind, root.parent_id), (ThreadKind::Folder, 1));
        assert_eq!(root.name, b"Atlas HD");
        // No thread: a file that has none, an unknown id.
        assert_eq!(catalog.thread(ids::README).unwrap(), None);
        assert_eq!(catalog.thread(9999).unwrap(), None);
        assert_eq!(catalog.thread(0).unwrap(), None);
    }

    #[test]
    fn record_goes_through_the_thread() {
        let (catalog, _) = standard();
        let (parent, rec) = catalog.record(ids::LETTERS).unwrap().unwrap();
        assert_eq!(parent, ids::DOCUMENTS);
        assert!(
            matches!(rec, CatalogRecord::Folder(f) if f.folder_id == ids::LETTERS && f.valence == 3)
        );
        let (parent, rec) = catalog.record(ids::DEAR_JOHN).unwrap().unwrap();
        assert_eq!(parent, ids::Y1991);
        assert!(matches!(rec, CatalogRecord::File(f) if f.data_logical == 700));
        // The root's thread names parent 1; its record is keyed by the
        // volume name.
        let (parent, rec) = catalog.record(ROOT_FOLDER_ID).unwrap().unwrap();
        assert_eq!(parent, ROOT_PARENT_ID);
        assert_eq!(rec.id(), Some(ROOT_FOLDER_ID));
        // No thread, no answer.
        assert_eq!(catalog.record(ids::README).unwrap(), None);
        assert_eq!(catalog.record(9999).unwrap(), None);
    }

    #[test]
    fn record_in_finds_a_file_without_a_thread() {
        let (catalog, manifest) = standard();
        let readme = manifest.by_id(ids::README).unwrap();
        assert!(!readme.has_thread);
        let rec = catalog
            .record_in(readme.parent, readme.id)
            .unwrap()
            .unwrap();
        assert!(
            matches!(rec, CatalogRecord::File(f) if f.file_id == ids::README && f.data_logical == 1200)
        );
        // A folder.
        let rec = catalog
            .record_in(ids::DOCUMENTS, ids::LETTERS)
            .unwrap()
            .unwrap();
        assert!(matches!(rec, CatalogRecord::Folder(_)));
        // The wrong parent, and an id that is not there.
        assert_eq!(catalog.record_in(ids::LETTERS, ids::README).unwrap(), None);
        assert_eq!(catalog.record_in(ids::DOCUMENTS, 9999).unwrap(), None);
        assert_eq!(catalog.record_in(9999, 1).unwrap(), None);
        // Every entry of the manifest is found in its parent.
        for e in &manifest.entries {
            let rec = catalog.record_in(e.parent, e.id).unwrap().unwrap();
            assert_eq!(rec.id(), Some(e.id), "{}", e.path);
        }
    }

    #[test]
    fn find_is_case_insensitive_and_accent_sensitive() {
        let (catalog, _) = standard();
        let find = |parent, name: &str| catalog.find(parent, name).unwrap().map(|(id, _)| id);
        assert_eq!(find(ids::DOCUMENTS, "ReadMe"), Some(ids::README));
        assert_eq!(find(ids::DOCUMENTS, "readme"), Some(ids::README));
        assert_eq!(find(ids::DOCUMENTS, "README"), Some(ids::README));
        assert_eq!(
            find(ids::DOCUMENTS, "readme ALIAS"),
            Some(ids::README_ALIAS)
        );
        assert_eq!(find(ids::DOCUMENTS, "Letters"), Some(ids::LETTERS));
        // Accents separate names.
        assert_eq!(find(ids::LETTERS, "Resume"), Some(ids::RESUME_PLAIN));
        assert_eq!(find(ids::LETTERS, "resume"), Some(ids::RESUME_PLAIN));
        assert_eq!(
            find(ids::LETTERS, "R\u{e9}sum\u{e9}"),
            Some(ids::RESUME_ACCENT)
        );
        assert_eq!(
            find(ids::LETTERS, "r\u{c9}SUM\u{c9}"),
            Some(ids::RESUME_ACCENT)
        );
        assert_eq!(find(ids::LETTERS, "Resum\u{e9}"), None);
        assert_eq!(find(ids::LETTERS, "R\u{e9}sume"), None);
        // Misses: unknown, in another folder, empty, a prefix, in an empty
        // folder, of an unknown folder.
        assert_eq!(find(ids::DOCUMENTS, "nope"), None);
        assert_eq!(find(ROOT_FOLDER_ID, "ReadMe"), None);
        assert_eq!(find(ids::DOCUMENTS, ""), None);
        assert_eq!(find(ids::DOCUMENTS, "Read"), None);
        assert_eq!(find(ids::EMPTY, "x"), None);
        assert_eq!(find(9999, "x"), None);
        // The record comes with the id.
        let (id, rec) = catalog.find(ROOT_FOLDER_ID, "simpletext").unwrap().unwrap();
        assert_eq!(id, ids::SIMPLETEXT);
        assert!(matches!(rec, CatalogRecord::File(f) if f.file_type() == *b"APPL"));
    }

    #[test]
    fn a_folder_spanning_two_leaves_lists_completely() {
        let (catalog, manifest) = standard();
        // Its records really are in more than one leaf node.
        let mut cursor = catalog.seek_parent(ids::BIG_FOLDER).unwrap();
        let mut nodes = std::collections::BTreeSet::new();
        while let Some(rec) = cursor.next_record().unwrap() {
            if CatalogKey::parse(rec.key).unwrap().parent_id != ids::BIG_FOLDER {
                break;
            }
            nodes.insert(cursor.position().unwrap().0);
        }
        assert!(nodes.len() >= 2, "records are in nodes {nodes:?}");
        let mut listed = list_all(&catalog, ids::BIG_FOLDER);
        assert_eq!(listed.len() as u32, ids::BIG_FILES);
        listed.sort_by(|a, b| a.name.cmp(&b.name));
        let want: Vec<String> = (1..=ids::BIG_FILES)
            .map(|i| format!("File {i:02}"))
            .collect();
        assert_eq!(
            listed.iter().map(|e| e.name.clone()).collect::<Vec<_>>(),
            want
        );
        for (i, e) in listed.iter().enumerate() {
            assert_eq!(e.id, u64::from(ids::FIRST_BIG_FILE) + i as u64);
            assert_eq!(e.size, Some(10 + i as u64));
        }
        assert_eq!(manifest.children(ids::BIG_FOLDER).count(), listed.len());
        // The neighbours on both sides are not mixed in.
        assert_eq!(list_all(&catalog, ids::NAMES).len(), 5);
    }

    #[test]
    fn high_roman_bytes_decode_to_their_characters() {
        let (catalog, _) = standard();
        let mut names: Vec<String> = list_all(&catalog, ids::NAMES)
            .into_iter()
            .map(|e| e.name)
            .collect();
        names.sort();
        let all: String = names.iter().flat_map(|n| n.chars().skip(3)).collect();
        let mut want = String::new();
        Encoding::Roman.decode_into(&(0x80..=0xFFu8).collect::<Vec<_>>(), &mut want);
        assert_eq!(all.chars().count(), 128);
        assert_eq!(all, want);
        // Spot checks: 0x8E is e acute, 0xA5 the bullet.
        assert!(all.contains('\u{e9}'));
        assert!(all.contains('\u{2022}'));
    }

    #[test]
    fn names_decode_with_the_catalogs_encoding() {
        let mut v = VolumeBuilder::new("Disk");
        // 0xC0 is a different letter in Mac OS Roman and in Cyrillic.
        v.file_raw(ROOT_FOLDER_ID, &[0xC0, b'x'], 16, FileSpec::new(1));
        let built = v.finish();
        let roman = open_catalog(built.image.clone());
        let cyrillic = Catalog::new(roman.tree().clone(), Encoding::Cyrillic);
        let name = |c: &Catalog| list_all(c, ROOT_FOLDER_ID).remove(0).name;
        assert_ne!(name(&roman), name(&cyrillic));
        assert_eq!(cyrillic.encoding(), Encoding::Cyrillic);
        assert!(
            cyrillic
                .find(ROOT_FOLDER_ID, &name(&cyrillic))
                .unwrap()
                .is_some()
        );
        assert!(
            cyrillic
                .find(ROOT_FOLDER_ID, &name(&roman))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn volume_name_is_the_root_threads() {
        let (catalog, _) = standard();
        assert_eq!(catalog.volume_name().unwrap().as_deref(), Some("Atlas HD"));
        let mut v = test_volume::standard();
        v.without_root_thread();
        let catalog = open_catalog(v.build());
        assert_eq!(catalog.volume_name().unwrap(), None);
        // Without a root thread the root cannot be listed either.
        assert!(matches!(
            list_error(&catalog, ROOT_FOLDER_ID),
            Error::NotFound { .. }
        ));
    }

    #[test]
    fn a_tracked_listing_fills_the_parent_map() {
        let (catalog, manifest) = standard();
        let parents = Arc::new(ParentMap::new());
        assert!(parents.is_empty());
        let (listed, err) = drain(
            catalog
                .list_tracked(ids::DOCUMENTS, Arc::clone(&parents))
                .unwrap(),
            2,
        );
        assert!(err.is_none());
        assert_eq!(parents.len(), listed.len());
        for e in manifest.children(ids::DOCUMENTS) {
            assert_eq!(parents.get(e.id), Some(ids::DOCUMENTS), "{}", e.path);
        }
        assert_eq!(parents.get(ids::SIMPLETEXT), None);
        // An untracked listing records nothing.
        let (_, err) = drain(catalog.list(ids::LETTERS).unwrap(), 2);
        assert!(err.is_none());
        assert_eq!(parents.get(ids::RESUME_PLAIN), None);
        // A file without a thread is then found through the map.
        let parent = parents.get(ids::README).unwrap();
        assert!(catalog.record_in(parent, ids::README).unwrap().is_some());
    }

    #[test]
    fn the_parent_map_starts_over_when_full() {
        let map = ParentMap::with_cap(3);
        for id in 1..=3 {
            map.insert(id, 100);
        }
        assert_eq!(map.len(), 3);
        // An id already there does not count as a new entry.
        map.insert(2, 200);
        assert_eq!((map.len(), map.get(2)), (3, Some(200)));
        // A new one clears the map first.
        map.insert(4, 300);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(4), Some(300));
        assert_eq!(map.get(1), None);
        let big = ParentMap::new();
        big.insert_all(&[(1, 2), (3, 4)]);
        assert_eq!(
            (big.get(1), big.get(3), big.get(5)),
            (Some(2), Some(4), None)
        );
        assert_eq!(PARENT_MAP_CAP, 1_000_000);
    }

    #[test]
    fn a_damaged_record_ends_the_listing_with_what_came_before() {
        let built = test_volume::standard_volume();
        let mut image = built.image.clone();
        // Find the record of "File 05" in the catalog and give it type 9.
        let key = encode_key(ids::BIG_FOLDER, b"File 05");
        let at = image
            .windows(key.len())
            .position(|w| w == key.as_slice())
            .unwrap();
        let data = at + key.len() + key.len() % 2;
        assert_eq!(image[data], FILE_RECORD);
        image[data] = 9;
        let catalog = open_catalog(image);
        let (entries, err) = drain(catalog.list(ids::BIG_FOLDER).unwrap(), 100);
        assert!(matches!(
            err,
            Some(Error::Corrupt(c)) if matches!(c.kind, CorruptKind::InvalidValue { value: 9, .. })
        ));
        // "File 01" to "File 04" sort before it and are kept.
        assert_eq!(entries.len(), 4);
        // The stream is finished.
        let mut dir = catalog.list(ids::BIG_FOLDER).unwrap();
        let mut batch = DirBatch::new();
        assert!(dir.next_batch(&mut batch, 100).is_err());
        assert_eq!(dir.next_batch(&mut batch, 100).unwrap(), DirProgress::Done);
        // Lookups that walk over it fail the same way.
        assert!(catalog.find(ids::BIG_FOLDER, "File 09").is_err());
    }
}
