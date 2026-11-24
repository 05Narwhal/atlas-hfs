// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Builds classic HFS volumes in memory for tests and fuzz seeds.
//!
//! Layout, in allocation blocks of [`DEFAULT_BLOCK_SIZE`] (512) bytes unless
//! the builder is told otherwise: sectors 0 and 1 are boot blocks, sector 2
//! holds the master directory block (and a copy in the second-to-last sector
//! of the image), the volume bitmap follows, then the allocation blocks.
//! In them, in this order: the catalog B-tree, the data and resource forks of
//! the files in the order they were added, the extents overflow B-tree, and
//! a few free blocks.
//!
//! A fork can be split into several extents ([`FileSpec::fragments`]); the
//! pieces have one free block between them, and extents past the third are
//! records of the extents overflow tree. The catalog is split the same way
//! (five pieces by default), so its overflow records are under file id 4.
//!
//! Both trees use 512-byte nodes, 8-bit keys and fixed-size index keys (37
//! bytes for the catalog, 7 for the overflow tree), and grow index levels as
//! needed. Catalog records are ordered by parent id, then by the stored name
//! folded to ASCII upper case, then by its bytes. That is not Apple's name
//! order (the reader has no use for it: see `catalog.rs`), only a consistent
//! one. A file's fork bytes are [`fork_bytes`] of its id, so a reader's
//! output can be checked without storing the content.

// The API is for the mount tests, the end-to-end tests and the fuzz seeds of
// the tasks after this one.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use super::catalog::{
    Catalog, FIRST_USER_ID, FileRecord, FolderRecord, INDEX_KEY_LEN, ROOT_FOLDER_ID,
    ROOT_PARENT_ID, ThreadKind, ThreadRecord, encode_key,
};
use super::encoding::Encoding;
use super::extents::{
    CATALOG_FILE_ID, EXTENT_KEY_LEN, EXTENT_RECORD_LEN, EXTENTS_FILE_ID, EXTENTS_PER_RECORD,
    ExtentKey, HfsExtent, HfsExtentRecord, fork_map, write_extent_record,
};
use super::mdb::{MAX_NAME_LEN as MDB_NAME_LEN, MDB_OFFSET, Mdb, SIGNATURE, block_area, read_mdb};
use atlas_fs_core::io::block::{BlockSource, MemSource};
use atlas_hfsplus::btree::fixtures::{INDEX, LEAF, header, header_node, node, set_node_used};
use atlas_hfsplus::btree::{BTree, NODE_DESCRIPTOR_LEN};
use atlas_hfsplus::extents::{ForkMap, ForkSource, ForkType};

/// Allocation block size unless a test asks for another.
pub const DEFAULT_BLOCK_SIZE: u32 = 512;

/// B-tree node size of both trees.
pub const NODE_SIZE: usize = 512;

/// Free blocks left at the end of the allocation area.
const SPARE_BLOCKS: u32 = 2;

/// Pieces the catalog is split into unless told otherwise.
const DEFAULT_CATALOG_FRAGMENTS: usize = 5;

/// The most nodes the header node's map record covers.
const MAX_NODES: usize = 2048;

/// Finder flag `kIsInvisible`.
pub const FINDER_INVISIBLE: u16 = 0x4000;
/// Finder flag `kIsAlias`.
pub const FINDER_ALIAS: u16 = 0x8000;
/// Finder flag `kHasBundle`.
pub const FINDER_HAS_BUNDLE: u16 = 0x2000;

/// Default dates, raw HFS seconds: 1970-01-01 00:08:20, 00:16:40 UTC.
pub const CREATED: u32 = 2_082_844_800 + 500;
pub const MODIFIED: u32 = 2_082_844_800 + 1_000;

/// The byte at `offset` of a fork of file `id`.
pub fn fork_byte(id: u32, resource: bool, offset: u64) -> u8 {
    let salt = if resource { 101 } else { 0 };
    let v = offset
        .wrapping_mul(131)
        .wrapping_add(u64::from(id).wrapping_mul(17))
        .wrapping_add(salt);
    (v % 251) as u8
}

/// The `len` bytes of a fork of file `id`.
pub fn fork_bytes(id: u32, resource: bool, len: u64) -> Vec<u8> {
    (0..len).map(|o| fork_byte(id, resource, o)).collect()
}

/// What a file record holds besides its name and id.
#[derive(Debug, Clone)]
pub struct FileSpec {
    /// Data fork logical size; its content is [`fork_bytes`].
    pub data_len: u64,
    /// Resource fork logical size.
    pub rsrc_len: u64,
    /// Extents the data fork is split into (at most its block count).
    pub data_fragments: usize,
    /// Extents the resource fork is split into.
    pub rsrc_fragments: usize,
    pub finder_flags: u16,
    pub file_type: [u8; 4],
    pub creator: [u8; 4],
    pub locked: bool,
    /// Whether the file has a thread record (most classic files do not).
    pub thread: bool,
    pub created: u32,
    pub modified: u32,
}

impl FileSpec {
    /// A `TEXT` file of `data_len` bytes in one extent, with no resource
    /// fork and no thread.
    pub fn new(data_len: u64) -> Self {
        Self {
            data_len,
            rsrc_len: 0,
            data_fragments: 1,
            rsrc_fragments: 1,
            finder_flags: 0,
            file_type: *b"TEXT",
            creator: *b"ttxt",
            locked: false,
            thread: false,
            created: CREATED,
            modified: MODIFIED,
        }
    }

    /// With a resource fork of `len` bytes.
    pub fn rsrc(self, len: u64) -> Self {
        Self {
            rsrc_len: len,
            ..self
        }
    }

    /// With the data fork split into `n` extents.
    pub fn fragments(self, n: usize) -> Self {
        Self {
            data_fragments: n,
            ..self
        }
    }

    /// With the resource fork split into `n` extents.
    pub fn rsrc_fragments(self, n: usize) -> Self {
        Self {
            rsrc_fragments: n,
            ..self
        }
    }

    /// With these Finder flags.
    pub fn flags(self, finder_flags: u16) -> Self {
        Self {
            finder_flags,
            ..self
        }
    }

    /// With `kIsInvisible` set.
    pub fn invisible(self) -> Self {
        let flags = self.finder_flags | FINDER_INVISIBLE;
        self.flags(flags)
    }

    /// With `kIsAlias` set.
    pub fn alias(self) -> Self {
        let flags = self.finder_flags | FINDER_ALIAS;
        self.flags(flags)
    }

    /// With a file thread record.
    pub fn with_thread(self) -> Self {
        Self {
            thread: true,
            ..self
        }
    }

    /// With the type and creator codes.
    pub fn types(self, file_type: [u8; 4], creator: [u8; 4]) -> Self {
        Self {
            file_type,
            creator,
            ..self
        }
    }

    /// With the modification date.
    pub fn modified(self, modified: u32) -> Self {
        Self { modified, ..self }
    }

    /// Locked.
    pub fn locked(self) -> Self {
        Self {
            locked: true,
            ..self
        }
    }
}

#[derive(Debug, Clone)]
enum Body {
    Folder { finder_flags: u16 },
    File(FileSpec),
}

#[derive(Debug, Clone)]
struct Item {
    parent: u32,
    id: u32,
    /// The name as stored (Mac OS Roman unless the test says otherwise).
    name: Vec<u8>,
    body: Body,
}

/// One file or folder of a built volume.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// Path from the root, `/`-separated, names decoded as Mac OS Roman.
    pub path: String,
    /// The decoded name.
    pub name: String,
    /// The name as stored.
    pub raw_name: Vec<u8>,
    pub id: u32,
    pub parent: u32,
    pub is_dir: bool,
    /// Children of a folder.
    pub valence: u16,
    pub data_len: u64,
    pub rsrc_len: u64,
    pub finder_flags: u16,
    pub file_type: [u8; 4],
    pub creator: [u8; 4],
    pub has_thread: bool,
    pub created: u32,
    pub modified: u32,
    /// Every extent of the data fork, (first allocation block, count),
    /// overflow records included.
    pub data_extents: Vec<(u32, u32)>,
    /// Every extent of the resource fork.
    pub rsrc_extents: Vec<(u32, u32)>,
}

impl Entry {
    /// The data fork's content.
    pub fn data_bytes(&self) -> Vec<u8> {
        fork_bytes(self.id, false, self.data_len)
    }

    /// The resource fork's content.
    pub fn rsrc_bytes(&self) -> Vec<u8> {
        fork_bytes(self.id, true, self.rsrc_len)
    }
}

/// What a built volume contains.
#[derive(Debug, Clone)]
pub struct Manifest {
    /// The volume name as stored.
    pub volume_name_bytes: Vec<u8>,
    /// The volume name, decoded.
    pub volume_name: String,
    pub block_size: u32,
    pub total_blocks: u32,
    pub free_blocks: u32,
    /// Byte offset of allocation block 0 in the image.
    pub alloc_start: u64,
    /// The catalog file's extents, overflow ones included.
    pub catalog_extents: Vec<(u32, u32)>,
    /// The extents overflow file's extents (one).
    pub extents_extents: Vec<(u32, u32)>,
    /// Nodes in the catalog file (used or free).
    pub catalog_nodes: u32,
    /// Everything but the root folder, in the order it was added.
    pub entries: Vec<Entry>,
}

impl Manifest {
    /// The entry at `path` (`/Documents/ReadMe`).
    pub fn by_path(&self, path: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.path == path)
    }

    /// The entry with id `id`.
    pub fn by_id(&self, id: u32) -> Option<&Entry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// The entries inside folder `parent`, in the order they were added.
    pub fn children(&self, parent: u32) -> impl Iterator<Item = &Entry> {
        self.entries.iter().filter(move |e| e.parent == parent)
    }

    /// Files only.
    pub fn files(&self) -> impl Iterator<Item = &Entry> {
        self.entries.iter().filter(|e| !e.is_dir)
    }
}

/// A built volume and what is in it.
#[derive(Debug, Clone)]
pub struct Built {
    pub image: Vec<u8>,
    pub manifest: Manifest,
}

/// Builds a volume. Every folder gets its thread record; a file only when
/// its spec asks.
#[derive(Debug, Clone)]
pub struct VolumeBuilder {
    name: Vec<u8>,
    block_size: u32,
    catalog_fragments: usize,
    root_thread: bool,
    items: Vec<Item>,
}

impl VolumeBuilder {
    /// A volume called `name` (Mac OS Roman, at most 27 bytes) with an
    /// empty root.
    pub fn new(name: &str) -> Self {
        Self {
            name: roman(name),
            block_size: DEFAULT_BLOCK_SIZE,
            catalog_fragments: DEFAULT_CATALOG_FRAGMENTS,
            root_thread: true,
            items: Vec::new(),
        }
    }

    /// Allocation block size, a multiple of 512.
    pub fn block_size(&mut self, block_size: u32) -> &mut Self {
        assert!(block_size >= 512 && block_size.is_multiple_of(512));
        self.block_size = block_size;
        self
    }

    /// Pieces the catalog file is split into (capped at its block count).
    /// More than three put its extra extents in the overflow tree.
    pub fn catalog_fragments(&mut self, n: usize) -> &mut Self {
        self.catalog_fragments = n;
        self
    }

    /// Leaves out the root folder's thread record.
    pub fn without_root_thread(&mut self) -> &mut Self {
        self.root_thread = false;
        self
    }

    /// A folder `id` called `name` in `parent`.
    pub fn folder(&mut self, parent: u32, name: &str, id: u32) -> &mut Self {
        self.folder_raw(parent, &roman(name), id, 0)
    }

    /// A folder with a stored name and Finder flags.
    pub fn folder_raw(
        &mut self,
        parent: u32,
        name: &[u8],
        id: u32,
        finder_flags: u16,
    ) -> &mut Self {
        self.items.push(Item {
            parent,
            id,
            name: name.to_vec(),
            body: Body::Folder { finder_flags },
        });
        self
    }

    /// A file `id` called `name` in `parent`.
    pub fn file(&mut self, parent: u32, name: &str, id: u32, spec: FileSpec) -> &mut Self {
        self.file_raw(parent, &roman(name), id, spec)
    }

    /// A file with a stored name.
    pub fn file_raw(&mut self, parent: u32, name: &[u8], id: u32, spec: FileSpec) -> &mut Self {
        self.items.push(Item {
            parent,
            id,
            name: name.to_vec(),
            body: Body::File(spec),
        });
        self
    }

    /// The volume image.
    pub fn build(&self) -> Vec<u8> {
        self.finish().image
    }

    /// The volume image and its manifest.
    pub fn finish(&self) -> Built {
        build(self)
    }
}

fn roman(text: &str) -> Vec<u8> {
    Encoding::Roman
        .encode(text)
        .unwrap_or_else(|| panic!("{text:?} has no Mac OS Roman encoding"))
}

fn decode(name: &[u8]) -> String {
    let mut out = String::new();
    Encoding::Roman.decode_into(name, &mut out);
    out
}

/// Where the forks of a volume are, in allocation blocks.
#[derive(Default)]
struct Placement {
    /// (file id, resource fork?) to its runs (first block, count).
    forks: HashMap<(u32, bool), Vec<(u32, u32)>>,
}

impl Placement {
    fn runs(&self, id: u32, resource: bool) -> &[(u32, u32)] {
        self.forks.get(&(id, resource)).map_or(&[], Vec::as_slice)
    }
}

/// A bump allocator: pieces of one fork have a free block between them.
struct Alloc {
    next: u32,
    used: Vec<(u32, u32)>,
}

impl Alloc {
    fn runs(&mut self, blocks: u32, fragments: usize) -> Vec<(u32, u32)> {
        if blocks == 0 {
            return Vec::new();
        }
        let pieces = u32::try_from(fragments.max(1))
            .unwrap_or(u32::MAX)
            .min(blocks);
        let (base, extra) = (blocks / pieces, blocks % pieces);
        let mut runs = Vec::new();
        for i in 0..pieces {
            let count = base + u32::from(i < extra);
            runs.push((self.next, count));
            self.used.push((self.next, count));
            self.next += count;
            if i + 1 < pieces {
                self.next += 1;
            }
        }
        runs
    }
}

fn extent(start: u32, count: u32) -> HfsExtent {
    HfsExtent {
        start_block: u16::try_from(start).expect("block number fits 16 bits"),
        block_count: u16::try_from(count).expect("block count fits 16 bits"),
    }
}

/// Splits `runs` into extent records of three, each with the fork-relative
/// block it starts at.
fn extent_records(runs: &[(u32, u32)]) -> Vec<(u32, HfsExtentRecord)> {
    let mut out = Vec::new();
    let mut mapped = 0u32;
    for chunk in runs.chunks(EXTENTS_PER_RECORD) {
        let mut record = [HfsExtent::default(); EXTENTS_PER_RECORD];
        for (slot, &(start, count)) in record.iter_mut().zip(chunk) {
            *slot = extent(start, count);
        }
        out.push((mapped, record));
        mapped += chunk.iter().map(|&(_, c)| c).sum::<u32>();
    }
    out
}

fn first_extents(runs: &[(u32, u32)]) -> HfsExtentRecord {
    extent_records(runs)
        .first()
        .map_or([HfsExtent::default(); EXTENTS_PER_RECORD], |&(_, r)| r)
}

/// The extents overflow records of one fork: every record past the first.
fn overflow_records(
    id: u32,
    resource: bool,
    runs: &[(u32, u32)],
) -> Vec<(ExtentKey, HfsExtentRecord)> {
    extent_records(runs)
        .into_iter()
        .skip(1)
        .map(|(start, record)| {
            let key = ExtentKey {
                fork_type: if resource { 0xFF } else { 0x00 },
                file_id: id,
                start_block: u16::try_from(start).expect("start block fits 16 bits"),
            };
            (key, record)
        })
        .collect()
}

fn blocks_of(len: u64, block_size: u32) -> u32 {
    u32::try_from(len.div_ceil(u64::from(block_size))).expect("fork fits the volume")
}

/// Groups `records` (leaf or index record bytes) into nodes: ranges of
/// indexes that fit one node each.
fn pack(records: &[Vec<u8>]) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut used = NODE_DESCRIPTOR_LEN + 2;
    for (i, r) in records.iter().enumerate() {
        let cost = r.len() + 2;
        assert!(
            NODE_DESCRIPTOR_LEN + 2 + cost <= NODE_SIZE,
            "record too big"
        );
        if used + cost > NODE_SIZE {
            out.push(start..i);
            start = i;
            used = NODE_DESCRIPTOR_LEN + 2;
        }
        used += cost;
    }
    if start < records.len() {
        out.push(start..records.len());
    }
    out
}

/// A classic HFS B-tree over `records` (key with its length byte, data), in
/// key order: header node, leaves, then index levels up to a root. Padded
/// with free nodes to a whole number of allocation blocks. Returns the
/// file's bytes.
fn build_tree(records: &[(Vec<u8>, Vec<u8>)], max_key_len: usize, block_size: u32) -> Vec<u8> {
    let leaf_recs: Vec<Vec<u8>> = records
        .iter()
        .map(|(k, d)| {
            let mut r = k.clone();
            if r.len() % 2 == 1 {
                r.push(0);
            }
            r.extend_from_slice(d);
            r
        })
        .collect();
    let mut nodes: Vec<Vec<u8>> = Vec::new();
    // (first key with its length byte, node number) of each node of the
    // level built last.
    let mut level: Vec<(Vec<u8>, u32)> = Vec::new();
    let groups = pack(&leaf_recs);
    let leaves = groups.len() as u32;
    for (j, g) in groups.iter().enumerate() {
        let number = 1 + j as u32;
        let forward = if (j as u32) + 1 < leaves {
            number + 1
        } else {
            0
        };
        let backward = if j == 0 { 0 } else { number - 1 };
        nodes.push(node(
            NODE_SIZE,
            LEAF,
            1,
            forward,
            backward,
            &leaf_recs[g.clone()],
        ));
        level.push((records[g.start].0.clone(), number));
    }
    let mut height = u8::from(!records.is_empty());
    let mut next_number = 1 + leaves;
    while level.len() > 1 {
        height += 1;
        let index_recs: Vec<Vec<u8>> = level
            .iter()
            .map(|(key, child)| {
                let mut r = key.clone();
                r.resize(1 + max_key_len, 0);
                if r.len() % 2 == 1 {
                    r.push(0);
                }
                r.extend_from_slice(&child.to_be_bytes());
                r
            })
            .collect();
        let groups = pack(&index_recs);
        let count = groups.len() as u32;
        let mut upper = Vec::new();
        for (j, g) in groups.iter().enumerate() {
            let number = next_number + j as u32;
            let forward = if (j as u32) + 1 < count {
                number + 1
            } else {
                0
            };
            let backward = if j == 0 { 0 } else { number - 1 };
            nodes.push(node(
                NODE_SIZE,
                INDEX,
                height,
                forward,
                backward,
                &index_recs[g.clone()],
            ));
            upper.push((level[g.start].0.clone(), number));
        }
        next_number += count;
        level = upper;
    }
    let used = 1 + nodes.len();
    let per_block = (block_size as usize) / NODE_SIZE;
    let total = used.div_ceil(per_block) * per_block;
    assert!(total <= MAX_NODES, "tree needs map nodes the fixture lacks");
    let root = if records.is_empty() { 0 } else { level[0].1 };
    let mut h = header(
        u16::from(height),
        root,
        u32::from(!records.is_empty()),
        leaves,
        total as u32,
    );
    h.attributes = 0;
    h.max_key_length = max_key_len as u16;
    h.leaf_records = records.len() as u32;
    h.free_nodes = (total - used) as u32;
    let mut header_bytes = header_node(&h);
    for n in used..total {
        set_node_used(&mut header_bytes, n, false);
    }
    let mut out = header_bytes;
    for n in nodes {
        out.extend_from_slice(&n);
    }
    out.resize(total * NODE_SIZE, 0);
    out
}

/// Copies `bytes` into the runs of `area`.
fn write_runs(area: &mut [u8], block_size: u32, runs: &[(u32, u32)], bytes: &[u8]) {
    let bs = block_size as usize;
    let mut at = 0;
    for &(start, count) in runs {
        let len = (count as usize * bs).min(bytes.len() - at);
        let dst = start as usize * bs;
        area[dst..dst + len].copy_from_slice(&bytes[at..at + len]);
        at += len;
    }
    assert_eq!(at, bytes.len(), "fork does not fit its runs");
}

fn folded(name: &[u8]) -> Vec<u8> {
    name.iter().map(u8::to_ascii_uppercase).collect()
}

fn finder_info(flags: u16) -> [u8; 16] {
    let mut info = [0u8; 16];
    info[8..10].copy_from_slice(&flags.to_be_bytes());
    info
}

fn build(b: &VolumeBuilder) -> Built {
    let bs = b.block_size;
    let children = |id: u32| b.items.iter().filter(|i| i.parent == id).count();

    // The catalog's leaf records. Their sizes do not depend on where the
    // forks are, so a first pass with no placement sizes the file.
    let catalog_records = |placement: &Placement| -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut recs: Vec<(u32, Vec<u8>, Vec<u8>)> = Vec::new();
        let thread = |kind, parent, name: &[u8]| {
            ThreadRecord {
                kind,
                reserved: [0; 8],
                parent_id: parent,
                name: name.to_vec(),
            }
            .encode()
        };
        recs.push((
            ROOT_PARENT_ID,
            b.name.clone(),
            FolderRecord {
                flags: 0,
                valence: children(ROOT_FOLDER_ID) as u16,
                folder_id: ROOT_FOLDER_ID,
                created: CREATED,
                modified: MODIFIED,
                backed_up: 0,
                finder_info: [0; 16],
                ext_finder_info: [0; 16],
                reserved: [0; 16],
            }
            .encode(),
        ));
        if b.root_thread {
            recs.push((
                ROOT_FOLDER_ID,
                Vec::new(),
                thread(ThreadKind::Folder, ROOT_PARENT_ID, &b.name),
            ));
        }
        for item in &b.items {
            match &item.body {
                Body::Folder { finder_flags } => {
                    recs.push((
                        item.parent,
                        item.name.clone(),
                        FolderRecord {
                            flags: 0,
                            valence: children(item.id) as u16,
                            folder_id: item.id,
                            created: CREATED,
                            modified: MODIFIED,
                            backed_up: 0,
                            finder_info: finder_info(*finder_flags),
                            ext_finder_info: [0; 16],
                            reserved: [0; 16],
                        }
                        .encode(),
                    ));
                    recs.push((
                        item.id,
                        Vec::new(),
                        thread(ThreadKind::Folder, item.parent, &item.name),
                    ));
                }
                Body::File(spec) => {
                    let data = placement.runs(item.id, false);
                    let rsrc = placement.runs(item.id, true);
                    let physical = |runs: &[(u32, u32)]| runs.iter().map(|r| r.1).sum::<u32>() * bs;
                    let mut info = finder_info(spec.finder_flags);
                    info[0..4].copy_from_slice(&spec.file_type);
                    info[4..8].copy_from_slice(&spec.creator);
                    recs.push((
                        item.parent,
                        item.name.clone(),
                        FileRecord {
                            flags: u8::from(spec.locked),
                            file_kind: 0,
                            finder_info: info,
                            file_id: item.id,
                            data_start_block: data.first().map_or(0, |r| r.0 as u16),
                            data_logical: spec.data_len as u32,
                            data_physical: physical(data),
                            rsrc_start_block: rsrc.first().map_or(0, |r| r.0 as u16),
                            rsrc_logical: spec.rsrc_len as u32,
                            rsrc_physical: physical(rsrc),
                            created: spec.created,
                            modified: spec.modified,
                            backed_up: 0,
                            ext_finder_info: [0; 16],
                            clump_size: 0,
                            data_extents: first_extents(data),
                            rsrc_extents: first_extents(rsrc),
                            reserved: 0,
                        }
                        .encode(),
                    ));
                    if spec.thread {
                        recs.push((
                            item.id,
                            Vec::new(),
                            thread(ThreadKind::File, item.parent, &item.name),
                        ));
                    }
                }
            }
        }
        recs.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| folded(&a.1).cmp(&folded(&b.1)))
                .then_with(|| a.1.cmp(&b.1))
        });
        recs.into_iter()
            .map(|(parent, name, data)| (encode_key(parent, &name), data))
            .collect()
    };

    // Pass 1: how big is the catalog?
    let sizing = build_tree(&catalog_records(&Placement::default()), INDEX_KEY_LEN, bs);
    let catalog_blocks = blocks_of(sizing.len() as u64, bs);

    // Place everything.
    let mut alloc = Alloc {
        next: 0,
        used: Vec::new(),
    };
    let mut placement = Placement::default();
    let catalog_runs = alloc.runs(catalog_blocks, b.catalog_fragments);
    let mut overflow: Vec<(ExtentKey, HfsExtentRecord)> =
        overflow_records(CATALOG_FILE_ID, false, &catalog_runs);
    for item in &b.items {
        if let Body::File(spec) = &item.body {
            for (resource, len, fragments) in [
                (false, spec.data_len, spec.data_fragments),
                (true, spec.rsrc_len, spec.rsrc_fragments),
            ] {
                let runs = alloc.runs(blocks_of(len, bs), fragments);
                overflow.extend(overflow_records(item.id, resource, &runs));
                placement.forks.insert((item.id, resource), runs);
            }
        }
    }
    overflow.sort_by_key(|(k, _)| k.order());
    let overflow_leaf: Vec<(Vec<u8>, Vec<u8>)> = overflow
        .iter()
        .map(|(k, rec)| {
            let mut data = [0u8; EXTENT_RECORD_LEN];
            write_extent_record(rec, &mut data).expect("record fits");
            (k.encode().to_vec(), data.to_vec())
        })
        .collect();
    let extents_tree = build_tree(&overflow_leaf, usize::from(EXTENT_KEY_LEN), bs);
    let extents_runs = alloc.runs(blocks_of(extents_tree.len() as u64, bs), 1);
    assert!(extents_runs.len() <= EXTENTS_PER_RECORD);

    let total_blocks = alloc.next + SPARE_BLOCKS;
    assert!(total_blocks <= u32::from(u16::MAX), "volume too large");
    let used_blocks: u32 = alloc.used.iter().map(|r| r.1).sum();

    // The real catalog.
    let catalog_tree = build_tree(&catalog_records(&placement), INDEX_KEY_LEN, bs);
    assert_eq!(catalog_tree.len(), sizing.len());

    // The allocation area.
    let mut area = vec![0u8; total_blocks as usize * bs as usize];
    write_runs(&mut area, bs, &catalog_runs, &catalog_tree);
    write_runs(&mut area, bs, &extents_runs, &extents_tree);
    for item in &b.items {
        if let Body::File(spec) = &item.body {
            write_runs(
                &mut area,
                bs,
                placement.runs(item.id, false),
                &fork_bytes(item.id, false, spec.data_len),
            );
            write_runs(
                &mut area,
                bs,
                placement.runs(item.id, true),
                &fork_bytes(item.id, true, spec.rsrc_len),
            );
        }
    }

    // Volume bitmap, most significant bit first, 1 = used.
    let bitmap_len = (total_blocks as usize).div_ceil(8);
    let bitmap_sectors = bitmap_len.div_ceil(512).max(1);
    let mut bitmap = vec![0u8; bitmap_sectors * 512];
    for &(start, count) in &alloc.used {
        for block in start..start + count {
            bitmap[block as usize / 8] |= 0x80 >> (block % 8);
        }
    }

    // The master directory block.
    let bitmap_start: u16 = 3;
    let alloc_start_sector = 3 + bitmap_sectors as u16;
    let mut name = [0u8; MDB_NAME_LEN];
    assert!(
        !b.name.is_empty() && b.name.len() <= MDB_NAME_LEN,
        "volume name must be 1 to 27 bytes"
    );
    name[..b.name.len()].copy_from_slice(&b.name);
    let files = b
        .items
        .iter()
        .filter(|i| matches!(i.body, Body::File(_)))
        .count();
    let dirs = b.items.len() - files;
    let root_files = b
        .items
        .iter()
        .filter(|i| i.parent == ROOT_FOLDER_ID && matches!(i.body, Body::File(_)))
        .count();
    let root_dirs = children(ROOT_FOLDER_ID) - root_files;
    let ct_extents = first_extents(&catalog_runs);
    let mdb = Mdb {
        signature: SIGNATURE,
        create_date: CREATED,
        modify_date: MODIFIED,
        attributes: 0x0100,
        root_file_count: root_files as u16,
        bitmap_start,
        alloc_ptr: 0,
        total_blocks: total_blocks as u16,
        block_size: bs,
        clump_size: bs * 4,
        alloc_start_sector,
        next_cnid: b
            .items
            .iter()
            .map(|i| i.id)
            .max()
            .map_or(FIRST_USER_ID, |m| m.max(FIRST_USER_ID - 1) + 1),
        free_blocks: (total_blocks - used_blocks) as u16,
        name_len: b.name.len() as u8,
        name,
        backup_date: 0,
        backup_seq: 0,
        write_count: 1,
        xt_clump_size: bs,
        ct_clump_size: bs,
        root_dir_count: root_dirs as u16,
        file_count: files as u32,
        dir_count: dirs as u32,
        finder_info: [0; 32],
        embed_sig: 0,
        embed_extent: HfsExtent::default(),
        xt_file_size: extents_tree.len() as u32,
        xt_extents: first_extents(&extents_runs),
        ct_file_size: catalog_tree.len() as u32,
        ct_extents,
    };

    // The image.
    let alloc_start = usize::from(alloc_start_sector) * 512;
    let mut image = vec![0u8; alloc_start + area.len() + 1024];
    image[3 * 512..3 * 512 + bitmap.len()].copy_from_slice(&bitmap);
    image[alloc_start..alloc_start + area.len()].copy_from_slice(&area);
    let alternate = image.len() - 1024;
    for at in [MDB_OFFSET as usize, alternate] {
        mdb.encode_into(&mut image[at..]).expect("room for the MDB");
    }

    // The manifest.
    let by_id: HashMap<u32, &Item> = b.items.iter().map(|i| (i.id, i)).collect();
    let path_of = |item: &Item| -> String {
        let mut parts = vec![decode(&item.name)];
        let mut parent = item.parent;
        while let Some(p) = by_id.get(&parent) {
            parts.push(decode(&p.name));
            parent = p.parent;
        }
        parts.reverse();
        format!("/{}", parts.join("/"))
    };
    let entries = b
        .items
        .iter()
        .map(|item| {
            let mut e = Entry {
                path: path_of(item),
                name: decode(&item.name),
                raw_name: item.name.clone(),
                id: item.id,
                parent: item.parent,
                is_dir: false,
                valence: 0,
                data_len: 0,
                rsrc_len: 0,
                finder_flags: 0,
                file_type: [0; 4],
                creator: [0; 4],
                has_thread: true,
                created: CREATED,
                modified: MODIFIED,
                data_extents: Vec::new(),
                rsrc_extents: Vec::new(),
            };
            match &item.body {
                Body::Folder { finder_flags } => {
                    e.is_dir = true;
                    e.valence = children(item.id) as u16;
                    e.finder_flags = *finder_flags;
                }
                Body::File(spec) => {
                    e.data_len = spec.data_len;
                    e.rsrc_len = spec.rsrc_len;
                    e.finder_flags = spec.finder_flags;
                    e.file_type = spec.file_type;
                    e.creator = spec.creator;
                    e.has_thread = spec.thread;
                    e.created = spec.created;
                    e.modified = spec.modified;
                    e.data_extents = placement.runs(item.id, false).to_vec();
                    e.rsrc_extents = placement.runs(item.id, true).to_vec();
                }
            }
            e
        })
        .collect();
    let manifest = Manifest {
        volume_name_bytes: b.name.clone(),
        volume_name: decode(&b.name),
        block_size: bs,
        total_blocks,
        free_blocks: total_blocks - used_blocks,
        alloc_start: alloc_start as u64,
        catalog_extents: catalog_runs,
        extents_extents: extents_runs,
        catalog_nodes: (catalog_tree.len() / NODE_SIZE) as u32,
        entries,
    };
    Built { image, manifest }
}

/// Ids of the entries of [`standard`].
pub mod ids {
    pub const DOCUMENTS: u32 = 16;
    pub const LETTERS: u32 = 17;
    pub const Y1991: u32 = 18;
    pub const DEAR_JOHN: u32 = 19;
    pub const RESUME_ACCENT: u32 = 20;
    pub const RESUME_PLAIN: u32 = 21;
    pub const README: u32 = 22;
    pub const README_ALIAS: u32 = 23;
    pub const SIMPLETEXT: u32 = 24;
    pub const FRAGMENTED: u32 = 25;
    pub const DESKTOP: u32 = 26;
    pub const DOTFILE: u32 = 27;
    pub const EMPTY: u32 = 28;
    pub const NAMES: u32 = 29;
    /// The five files of `Names`, 30 to 34, hold every byte 0x80 to 0xFF.
    pub const FIRST_NAME_FILE: u32 = 30;
    pub const BIG_FOLDER: u32 = 35;
    /// The files of `Big Folder`, 36 to 44.
    pub const FIRST_BIG_FILE: u32 = 36;
    pub const BIG_FILES: u32 = 9;
}

/// The volume most tests use: `Atlas HD` with nested folders (three
/// levels), a file with both forks, a file of eight one-block extents (two
/// overflow records), a catalog in five pieces (one overflow record under
/// id 4), an invisible file, an alias, a dot file, an empty folder, `Names`
/// (every high Roman byte), `Big Folder` (nine files, more than a leaf) and
/// one file with a thread (`Dear John`).
pub fn standard() -> VolumeBuilder {
    use ids::*;
    let mut v = VolumeBuilder::new("Atlas HD");
    v.folder(ROOT_FOLDER_ID, "Documents", DOCUMENTS)
        .folder(DOCUMENTS, "Letters", LETTERS)
        .folder(LETTERS, "1991", Y1991)
        .file(
            Y1991,
            "Dear John",
            DEAR_JOHN,
            FileSpec::new(700).with_thread(),
        )
        .file(
            LETTERS,
            "R\u{e9}sum\u{e9}",
            RESUME_ACCENT,
            FileSpec::new(100),
        )
        .file(LETTERS, "Resume", RESUME_PLAIN, FileSpec::new(50))
        .file(DOCUMENTS, "ReadMe", README, FileSpec::new(1200))
        .file(
            DOCUMENTS,
            "ReadMe alias",
            README_ALIAS,
            FileSpec::new(0).rsrc(300).alias().types(*b"alis", *b"MACS"),
        )
        .file(
            ROOT_FOLDER_ID,
            "SimpleText",
            SIMPLETEXT,
            FileSpec::new(1500)
                .rsrc(2000)
                .flags(FINDER_HAS_BUNDLE)
                .types(*b"APPL", *b"ttxt"),
        )
        .file(
            ROOT_FOLDER_ID,
            "Fragmented",
            FRAGMENTED,
            FileSpec::new(8 * 512)
                .fragments(8)
                .rsrc(1000)
                .rsrc_fragments(2),
        )
        .file(
            ROOT_FOLDER_ID,
            "Desktop",
            DESKTOP,
            FileSpec::new(64).invisible().types(*b"FNDR", *b"ERIK"),
        )
        .file(ROOT_FOLDER_ID, ".profile", DOTFILE, FileSpec::new(10))
        .folder(ROOT_FOLDER_ID, "Empty", EMPTY)
        .folder(ROOT_FOLDER_ID, "Names", NAMES);
    // Every byte 0x80 to 0xFF once, in five names.
    for (i, start) in [0x80u8, 0x9A, 0xB4, 0xCE, 0xE8].into_iter().enumerate() {
        let end = if start == 0xE8 { 0xFF } else { start + 25 };
        let mut name = format!("N{i} ").into_bytes();
        name.extend(start..=end);
        v.file_raw(NAMES, &name, FIRST_NAME_FILE + i as u32, FileSpec::new(10));
    }
    v.folder(ROOT_FOLDER_ID, "Big Folder", BIG_FOLDER);
    for i in 0..BIG_FILES {
        v.file(
            BIG_FOLDER,
            &format!("File {:02}", i + 1),
            FIRST_BIG_FILE + i,
            FileSpec::new(10 + u64::from(i)),
        );
    }
    v
}

/// The standard volume, built.
pub fn standard_volume() -> Built {
    standard().finish()
}

/// A volume opened as a mount would: its master directory block, block
/// area, both trees and the catalog.
pub struct Opened {
    pub mdb: Mdb,
    /// Allocation block `n` is at byte `n * block_size`.
    pub area: Arc<dyn BlockSource>,
    pub extents: BTree,
    pub catalog: Catalog,
}

/// A fork of the area as a source, through `map`.
pub fn fork_source(area: &Arc<dyn BlockSource>, map: ForkMap) -> Arc<dyn BlockSource> {
    Arc::new(ForkSource::new(Arc::clone(area), Arc::new(map)))
}

/// Opens `bytes`: `read_mdb`, `block_area`, the extents file from the MDB's
/// three extents, its tree, the catalog file with that tree, its tree.
pub fn open(bytes: Vec<u8>) -> Opened {
    let source: Arc<dyn BlockSource> = Arc::new(MemSource::new(bytes));
    let mdb = read_mdb(&*source)
        .expect("a valid master directory block")
        .mdb;
    let area: Arc<dyn BlockSource> = Arc::new(block_area(source, &mdb).expect("block area"));
    let geometry = mdb.geometry();
    let map = fork_map(
        geometry,
        EXTENTS_FILE_ID,
        ForkType::Data,
        u64::from(mdb.xt_file_size),
        &mdb.xt_extents,
        None,
    )
    .expect("extents file map");
    let extents =
        BTree::open(fork_source(&area, map), "extents overflow B-tree").expect("extents tree");
    let map = fork_map(
        geometry,
        CATALOG_FILE_ID,
        ForkType::Data,
        u64::from(mdb.ct_file_size),
        &mdb.ct_extents,
        Some(&extents),
    )
    .expect("catalog file map");
    let tree = BTree::open(fork_source(&area, map), "catalog B-tree").expect("catalog tree");
    Opened {
        mdb,
        area,
        extents,
        catalog: Catalog::new(tree, Encoding::Roman),
    }
}

/// The catalog of `bytes`.
pub fn open_catalog(bytes: Vec<u8>) -> Catalog {
    open(bytes).catalog
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::CatalogRecord;
    use crate::mdb::MdbCopy;

    #[test]
    fn both_master_directory_blocks_are_the_same() {
        let image = standard_volume().image;
        let located = read_mdb(&MemSource::new(image.clone())).unwrap();
        assert_eq!(located.copy, MdbCopy::Primary);
        let alt = image.len() - 1024;
        assert_eq!(image[1024..1024 + 162], image[alt..alt + 162]);
    }

    #[test]
    fn the_bitmap_marks_exactly_the_allocated_blocks() {
        let built = standard_volume();
        let m = &built.manifest;
        let mut want = vec![false; m.total_blocks as usize];
        let runs = m
            .catalog_extents
            .iter()
            .chain(&m.extents_extents)
            .chain(m.entries.iter().flat_map(|e| &e.data_extents))
            .chain(m.entries.iter().flat_map(|e| &e.rsrc_extents));
        for &(start, count) in runs {
            for b in start..start + count {
                assert!(!want[b as usize], "block {b} allocated twice");
                want[b as usize] = true;
            }
        }
        let opened = open(built.image.clone());
        let bitmap_at = opened.mdb.bitmap_offset() as usize;
        for (block, &used) in want.iter().enumerate() {
            let bit = built.image[bitmap_at + block / 8] & (0x80 >> (block % 8)) != 0;
            assert_eq!(bit, used, "block {block}");
        }
        let free = want.iter().filter(|u| !**u).count();
        assert_eq!(free as u32, m.free_blocks);
        assert_eq!(opened.mdb.free_blocks as usize, free);
    }

    #[test]
    fn the_catalog_and_the_fragmented_file_use_overflow_records() {
        let built = standard_volume();
        let m = &built.manifest;
        assert!(m.catalog_extents.len() > 3);
        let fragmented = m.by_id(ids::FRAGMENTED).unwrap();
        assert_eq!(fragmented.data_extents.len(), 8);
        // The catalog has 5 extents (one overflow record), the fragmented
        // data fork 8 (records at blocks 3 and 6), its resource fork 2 (none).
        assert_eq!(opened_overflow_records(&built.image), 3);
    }

    fn opened_overflow_records(image: &[u8]) -> u32 {
        open(image.to_vec()).extents.header().leaf_records
    }

    #[test]
    fn forks_read_back_through_the_catalog_and_the_overflow_tree() {
        check_forks(standard_volume());
    }

    #[test]
    fn larger_allocation_blocks_work_too() {
        for bs in [1024, 2048] {
            let built = standard().block_size(bs).finish();
            assert_eq!(built.manifest.block_size, bs);
            check_forks(built);
        }
    }

    fn check_forks(built: Built) {
        let opened = open(built.image.clone());
        let geometry = opened.mdb.geometry();
        for e in built.manifest.files() {
            let Some(CatalogRecord::File(rec)) = opened.catalog.record_in(e.parent, e.id).unwrap()
            else {
                panic!("{} has no file record", e.path);
            };
            for (resource, len, first, expected) in [
                (false, e.data_len, &rec.data_extents, e.data_bytes()),
                (true, e.rsrc_len, &rec.rsrc_extents, e.rsrc_bytes()),
            ] {
                let ty = if resource {
                    ForkType::Resource
                } else {
                    ForkType::Data
                };
                let map = fork_map(geometry, e.id, ty, len, first, Some(&opened.extents)).unwrap();
                let src = fork_source(&opened.area, map);
                let mut got = vec![0u8; expected.len()];
                src.read_at(0, &mut got).unwrap();
                assert_eq!(got, expected, "{} (resource: {resource})", e.path);
            }
        }
    }
}
