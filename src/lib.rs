// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Classic HFS (big-endian), the Mac OS volume format before HFS+.
//!
//! The catalog and extents overflow files are B-trees read through the
//! shared [`atlas_hfsplus::btree`] reader; forks map through
//! [`atlas_hfsplus::extents::ForkMap`].
//!
//! [`mount`] reads the master directory block (or its alternate copy), cuts
//! the *block area* out of the source (allocation block `n` is at byte
//! `n * block_size`, exactly HFS+'s geometry), and opens the extents and
//! catalog B-trees in it. [`Hfs`] then answers the [`FileSystem`] calls.
//!
//! Classic HFS cannot descend its catalog to a named key (design decision
//! 3), so a file with no thread record is found by id only through the
//! listing of its folder: every listing and lookup records the pair in a
//! [`ParentMap`] that `metadata` falls back on.
//!
//! Damaged input is an error, never a panic: every length, offset and count
//! read from the disk is checked before use. The volume is read-only.
//!
//! ```no_run
//! use std::path::Path;
//! use std::sync::Arc;
//!
//! use atlas_fs_core::fs::{DirBatch, DirProgress, FileSystem};
//! use atlas_fs_core::io::file::FileSource;
//!
//! # fn main() -> Result<(), atlas_fs_core::error::Error> {
//! let source = Arc::new(FileSource::open(Path::new("classic.img"))?);
//! let volume = atlas_hfs::mount(source)?;
//! println!("{}", volume.info().name);
//!
//! let mut stream = volume.read_dir(volume.root())?;
//! let mut batch = DirBatch::new();
//! while stream.next_batch(&mut batch, 256)? == DirProgress::More {}
//! for entry in batch.iter() {
//!     println!("{}", entry.name);
//! }
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]
// On-disk bytes are hostile. These lints make the usual ways of trusting
// them (unchecked indexing, arithmetic and casts, panicking shortcuts) fail
// the clippy gate. Unit tests and the test fixtures are exempt.
#![warn(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unimplemented,
    clippy::unreachable,
    clippy::unwrap_used
)]
#![cfg_attr(
    any(test, feature = "test-util"),
    allow(
        clippy::arithmetic_side_effects,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used
    )
)]

use std::ops::ControlFlow;
use std::sync::Arc;

use atlas_fs_core::error::{self, Error, Result};
use atlas_fs_core::fs::{
    AllocationMap, DirStream, Extent, FileSystem, FinderCodes, Metadata, NodeId, NodeKind,
    ScanEntry, ScanSummary, Stream, VolumeIdentity, VolumeInfo,
};
use atlas_fs_core::io::block::{BlockSource, Slice};
use atlas_hfsplus::allocation::density;
use atlas_hfsplus::btree::BTree;
use atlas_hfsplus::catalog::entry_flags;
use atlas_hfsplus::extents::{ForkMap, ForkSource, ForkType};
use atlas_hfsplus::volume_header::hfs_time;

use catalog::{Catalog, CatalogRecord, FileRecord, ParentMap, ROOT_FOLDER_ID, ROOT_PARENT_ID};
use encoding::Encoding;
use extents::{CATALOG_FILE_ID, EXTENTS_FILE_ID, fork_map};
use mdb::{Mdb, block_area, read_mdb};

pub mod catalog;
pub mod encoding;
pub mod extents;
#[cfg(test)]
mod fuzz_seeds;
pub mod mdb;
mod scan;
#[cfg(any(test, feature = "test-util"))]
#[doc(hidden)]
#[allow(missing_docs)]
pub mod test_volume;

/// A mounted classic HFS volume.
pub struct Hfs {
    /// The volume itself, for the bitmap (which is before the block area).
    volume: Arc<dyn BlockSource>,
    /// The allocation blocks: block `n` is at byte `n * block_size`.
    area: Arc<dyn BlockSource>,
    mdb: Mdb,
    /// The extents overflow B-tree, for mapping forks past three extents.
    extents: BTree,
    catalog: Catalog,
    /// CNID to parent id, filled by listings and lookups.
    parents: Arc<ParentMap>,
    info: VolumeInfo,
    identity: VolumeIdentity,
}

// `FileSystem` requires it; this names the failure here if a field breaks it.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Hfs>();
};

/// Mounts the classic HFS volume in `source`, with names in Mac OS Roman.
///
/// Fails when no master directory block (primary or alternate) is valid or
/// when the extents or catalog B-tree cannot be opened.
pub fn mount(source: Arc<dyn BlockSource>) -> Result<Hfs> {
    mount_with(source, Encoding::Roman)
}

/// Mounts the classic HFS volume in `source`, decoding names (and the
/// volume name) with `encoding`.
pub fn mount_with(source: Arc<dyn BlockSource>, encoding: Encoding) -> Result<Hfs> {
    let mdb = read_mdb(&*source)?.mdb;
    let area: Arc<dyn BlockSource> = Arc::new(block_area(Arc::clone(&source), &mdb)?);
    let geometry = mdb.geometry();

    let extents_map = fork_map(
        geometry,
        EXTENTS_FILE_ID,
        ForkType::Data,
        u64::from(mdb.xt_file_size),
        &mdb.xt_extents,
        None,
    )?;
    let extents = BTree::open(
        Arc::new(ForkSource::new(Arc::clone(&area), Arc::new(extents_map))),
        "extents overflow B-tree",
    )?;
    let catalog_map = fork_map(
        geometry,
        CATALOG_FILE_ID,
        ForkType::Data,
        u64::from(mdb.ct_file_size),
        &mdb.ct_extents,
        Some(&extents),
    )?;
    let catalog_tree = BTree::open(
        Arc::new(ForkSource::new(Arc::clone(&area), Arc::new(catalog_map))),
        "catalog B-tree",
    )?;
    let catalog = Catalog::new(catalog_tree, encoding);

    // The root thread's name; the master directory block's when there is
    // none or the thread cannot be read (the volume still lists).
    let name = match catalog.volume_name() {
        Ok(Some(name)) => name,
        Ok(None) | Err(_) => {
            let mut name = String::new();
            encoding.decode_into(mdb.name_bytes(), &mut name);
            name
        }
    };
    let mut info = VolumeInfo::new(name);
    info.format = Some("HFS");
    info.total_bytes = Some(area.len());
    info.free_bytes = Some(mdb.free_bytes());
    info.resource_forks = true;
    info.local_times = true;

    Ok(Hfs {
        volume: source,
        area,
        identity: identity_of(&mdb),
        mdb,
        extents,
        catalog,
        parents: Arc::new(ParentMap::new()),
        info,
    })
}

/// `parts` back to back, zero-padded (or cut) to 16 bytes.
fn pack(parts: &[&[u8]]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (dst, src) in out.iter_mut().zip(parts.iter().flat_map(|p| p.iter())) {
        *dst = *src;
    }
    out
}

/// The identity of the volume `mdb` describes.
///
/// Classic HFS has no unique identifier, so `id` is the best there is: the
/// creation date, the allocation block count and size, and the first six
/// bytes of the volume name. `change`: the write count, the modification
/// date and the file and folder counts. All big-endian.
fn identity_of(mdb: &Mdb) -> VolumeIdentity {
    let name = mdb.name_bytes();
    VolumeIdentity {
        id: pack(&[
            &mdb.create_date.to_be_bytes(),
            &mdb.total_blocks.to_be_bytes(),
            &mdb.block_size.to_be_bytes(),
            name.get(..6).unwrap_or(name),
        ]),
        change: pack(&[
            &mdb.write_count.to_be_bytes(),
            &mdb.modify_date.to_be_bytes(),
            &mdb.file_count.to_be_bytes(),
            &mdb.dir_count.to_be_bytes(),
        ]),
    }
}

/// The CNID a node ID names. 0 and IDs past `u32` are `NotFound`.
fn cnid(node: NodeId) -> Result<u32> {
    match u32::try_from(node.0) {
        Ok(id) if id != 0 => Ok(id),
        _ => Err(Error::NotFound { node: node.0 }),
    }
}

/// The type and creator codes of `file`, as stored; `None` when both are 0.
fn finder_codes_of(file: &FileRecord) -> Option<FinderCodes> {
    let (file_type, creator) = (file.file_type(), file.creator());
    (file_type != [0; 4] || creator != [0; 4]).then_some(FinderCodes { file_type, creator })
}

/// The 32 bytes of Finder information as stored (`FInfo` then `FXInfo`, or
/// `DInfo` then `DXInfo`); `None` when all are zero.
fn finder_info_of(info: [u8; 16], extended: [u8; 16]) -> Option<[u8; 32]> {
    let mut out = [0u8; 32];
    for (dst, src) in out.iter_mut().zip(info.iter().chain(extended.iter())) {
        *dst = *src;
    }
    out.iter().any(|&b| b != 0).then_some(out)
}

/// The runs of `map` as bytes of the volume: `alloc_start` is the byte
/// offset of allocation block 0. In fork order, each merged into the one
/// before when they touch on disk. Whole allocation blocks, past the
/// logical size too.
fn disk_runs(
    map: &ForkMap,
    file_id: u32,
    block_size: u32,
    alloc_start: u64,
) -> Result<Vec<Extent>> {
    let overflow = || {
        Error::Corrupt(error::Corruption::new(
            "fork",
            error::Location::Object(u64::from(file_id)),
            error::CorruptKind::Overflow {
                field: "extent byte offset",
            },
        ))
    };
    let bs = u64::from(block_size);
    let runs = map.runs();
    let mut out: Vec<Extent> = Vec::with_capacity(runs.size_hint().0);
    // Bounded by the fork map's run count.
    for (_, run) in runs {
        let len = u64::from(run.block_count)
            .checked_mul(bs)
            .ok_or_else(overflow)?;
        if len == 0 {
            continue;
        }
        let offset = u64::from(run.start_block)
            .checked_mul(bs)
            .and_then(|b| b.checked_add(alloc_start))
            .ok_or_else(overflow)?;
        if let Some(prev) = out.last_mut()
            && prev.offset.checked_add(prev.len) == Some(offset)
        {
            prev.len = prev.len.checked_add(len).ok_or_else(overflow)?;
            continue;
        }
        out.push(Extent { offset, len });
    }
    Ok(out)
}

impl Hfs {
    /// The folder or file record of `id`: through its thread when it has
    /// one, else through the folder the listings and lookups saw it in. The
    /// root has a fixed parent, so it needs neither.
    fn record_of(&self, node: NodeId) -> Result<CatalogRecord> {
        let id = cnid(node)?;
        if let Some((_, record)) = self.catalog.record(id)? {
            return Ok(record);
        }
        let parent = if id == ROOT_FOLDER_ID {
            Some(ROOT_PARENT_ID)
        } else {
            self.parents.get(id)
        };
        if let Some(parent) = parent
            && let Some(record) = self.catalog.record_in(parent, id)?
        {
            return Ok(record);
        }
        Err(Error::NotFound { node: node.0 })
    }

    /// The file record of `node`; `IsADirectory` for a folder.
    fn file_of(&self, node: NodeId) -> Result<FileRecord> {
        match self.record_of(node)? {
            CatalogRecord::File(file) => Ok(file),
            CatalogRecord::Folder(_) => Err(Error::IsADirectory { node: node.0 }),
            CatalogRecord::FolderThread(_) | CatalogRecord::FileThread(_) => {
                Err(Error::NotFound { node: node.0 })
            }
        }
    }

    /// The map of one fork of `file`.
    fn map_of(&self, file: &FileRecord, fork_type: ForkType) -> Result<ForkMap> {
        let (logical, first) = match fork_type {
            ForkType::Data => (file.data_logical, &file.data_extents),
            ForkType::Resource => (file.rsrc_logical, &file.rsrc_extents),
        };
        fork_map(
            self.mdb.geometry(),
            file.file_id,
            fork_type,
            u64::from(logical),
            first,
            Some(&self.extents),
        )
    }
}

impl FileSystem for Hfs {
    fn info(&self) -> VolumeInfo {
        self.info.clone()
    }

    fn root(&self) -> NodeId {
        NodeId(u64::from(ROOT_FOLDER_ID))
    }

    fn metadata(&self, node: NodeId) -> Result<Metadata> {
        let mut meta = Metadata {
            node,
            kind: NodeKind::File,
            size: None,
            created: None,
            modified: None,
            accessed: None,
            flags: entry_flags(false, 0),
            link_target: None,
            resource_size: None,
            finder_codes: None,
            owner: None,
            attributes: None,
            finder_info: None,
        };
        match self.record_of(node)? {
            CatalogRecord::Folder(f) => {
                meta.kind = NodeKind::Directory;
                meta.created = hfs_time(f.created);
                meta.modified = hfs_time(f.modified);
                meta.flags = entry_flags(false, f.finder_flags());
                meta.finder_info = finder_info_of(f.finder_info, f.ext_finder_info);
            }
            CatalogRecord::File(f) => {
                meta.size = Some(u64::from(f.data_logical));
                meta.resource_size = Some(u64::from(f.rsrc_logical));
                meta.created = hfs_time(f.created);
                meta.modified = hfs_time(f.modified);
                meta.flags = entry_flags(false, f.finder_flags());
                meta.finder_codes = finder_codes_of(&f);
                meta.finder_info = finder_info_of(f.finder_info, f.ext_finder_info);
            }
            CatalogRecord::FolderThread(_) | CatalogRecord::FileThread(_) => {
                return Err(Error::NotFound { node: node.0 });
            }
        }
        Ok(meta)
    }

    fn lookup(&self, dir: NodeId, name: &str) -> Result<Option<NodeId>> {
        let dir_id = cnid(dir)?;
        if name.is_empty() {
            return Ok(None);
        }
        let Some((id, _)) = self.catalog.find(dir_id, name)? else {
            return Ok(None);
        };
        // So `metadata` finds a file that has no thread.
        self.parents.insert(id, dir_id);
        Ok(Some(NodeId(u64::from(id))))
    }

    fn read_dir(&self, dir: NodeId) -> Result<Box<dyn DirStream>> {
        let listing = self
            .catalog
            .list_tracked(cnid(dir)?, Arc::clone(&self.parents))?;
        Ok(Box::new(listing))
    }

    /// The allocated blocks of a fork, as bytes of the volume (the block
    /// area starts at `drAlBlSt` sectors).
    fn extents(&self, node: NodeId, stream: Stream) -> Result<Vec<Extent>> {
        let file = self.file_of(node)?;
        let fork_type = match stream {
            Stream::Data => ForkType::Data,
            Stream::Resource => ForkType::Resource,
        };
        let map = self.map_of(&file, fork_type)?;
        disk_runs(
            &map,
            file.file_id,
            self.mdb.block_size,
            self.mdb.alloc_start(),
        )
    }

    /// The data fork, or the resource fork for `Stream::Resource`.
    /// `Error::NoSuchStream` for an empty resource fork, `IsADirectory` for
    /// a folder.
    fn open(&self, node: NodeId, stream: Stream) -> Result<Arc<dyn BlockSource>> {
        let file = self.file_of(node)?;
        let fork_type = match stream {
            Stream::Data => ForkType::Data,
            Stream::Resource => {
                if file.rsrc_logical == 0 {
                    return Err(Error::NoSuchStream { node: node.0 });
                }
                ForkType::Resource
            }
        };
        let map = self.map_of(&file, fork_type)?;
        Ok(Arc::new(ForkSource::new(
            Arc::clone(&self.area),
            Arc::new(map),
        )))
    }

    /// Reads the volume bitmap (at most 8 KiB) and counts its set bits per
    /// bucket (see [`density`]).
    fn allocation_map(&self, buckets: usize) -> Result<AllocationMap> {
        let bitmap = Slice::new(
            Arc::clone(&self.volume),
            self.mdb.bitmap_offset(),
            self.mdb.bitmap_len(),
        )?;
        let density = density(&bitmap, u32::from(self.mdb.total_blocks), buckets)?;
        Ok(AllocationMap::new(
            self.mdb.alloc_start(),
            self.area.len(),
            density,
        ))
    }

    /// Reads the catalog's leaves front to back, once (see the `scan`
    /// module).
    fn scan(&self, visit: &mut dyn FnMut(ScanEntry<'_>) -> ControlFlow<()>) -> Result<ScanSummary> {
        scan::scan(&self.catalog, visit)
    }

    fn identity(&self) -> Option<VolumeIdentity> {
        Some(self.identity)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::test_volume::{
        Built, CREATED, Entry as Spec, FINDER_ALIAS, FINDER_INVISIBLE, MODIFIED, ids, standard,
        standard_volume,
    };
    use super::*;
    use atlas_fs_core::fs::{DirBatch, DirProgress, EntryFlags, Timestamp};
    use atlas_fs_core::io::block::MemSource;

    const ROOT: u64 = 2;

    fn mounted_bytes(bytes: Vec<u8>) -> Result<Hfs> {
        mount(Arc::new(MemSource::new(bytes)))
    }

    fn mounted(built: &Built) -> Hfs {
        mounted_bytes(built.image.clone()).expect("mount")
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Listed {
        id: u64,
        name: String,
        kind: NodeKind,
        size: Option<u64>,
        resource_size: Option<u64>,
        modified: Option<Timestamp>,
        flags: EntryFlags,
    }

    fn list(fs: &Hfs, dir: u64) -> Result<Vec<Listed>> {
        let mut stream = fs.read_dir(NodeId(dir))?;
        let mut batch = DirBatch::new();
        let mut out = Vec::new();
        loop {
            batch.clear();
            let progress = stream.next_batch(&mut batch, 3)?;
            out.extend(batch.iter().map(|e| Listed {
                id: e.node.0,
                name: e.name.to_owned(),
                kind: e.kind,
                size: e.size,
                resource_size: e.resource_size,
                modified: e.modified,
                flags: e.flags,
            }));
            if progress == DirProgress::Done {
                return Ok(out);
            }
        }
    }

    /// Every entry below the root, by path, from recursive listings.
    fn walk(fs: &Hfs) -> BTreeMap<String, Listed> {
        let mut out = BTreeMap::new();
        let mut stack = vec![(ROOT, String::new())];
        while let Some((dir, path)) = stack.pop() {
            for entry in list(fs, dir).unwrap() {
                let full = format!("{path}/{}", entry.name);
                if entry.kind == NodeKind::Directory {
                    stack.push((entry.id, full.clone()));
                }
                assert!(out.insert(full, entry).is_none());
            }
        }
        out
    }

    fn flags_of(spec: &Spec) -> EntryFlags {
        entry_flags(false, spec.finder_flags)
    }

    fn check_listing(listed: &Listed, spec: &Spec) {
        assert_eq!(listed.id, u64::from(spec.id), "{}", spec.path);
        assert_eq!(listed.name, spec.name, "{}", spec.path);
        assert_eq!(listed.modified, hfs_time(spec.modified), "{}", spec.path);
        assert_eq!(listed.flags, flags_of(spec), "{}", spec.path);
        if spec.is_dir {
            assert_eq!(listed.kind, NodeKind::Directory);
            assert_eq!(listed.size, None);
            assert_eq!(listed.resource_size, None);
        } else {
            assert_eq!(listed.kind, NodeKind::File);
            assert_eq!(listed.size, Some(spec.data_len), "{}", spec.path);
            assert_eq!(listed.resource_size, Some(spec.rsrc_len), "{}", spec.path);
        }
    }

    fn read_all(src: &Arc<dyn BlockSource>) -> Vec<u8> {
        let mut buf = vec![0u8; usize::try_from(src.len()).unwrap()];
        src.read_at(0, &mut buf).unwrap();
        buf
    }

    // ---- info and identity ----

    #[test]
    fn info_of_the_standard_volume() {
        let built = standard_volume();
        let fs = mounted(&built);
        let m = &built.manifest;
        let info = fs.info();
        assert_eq!(info.name, "Atlas HD");
        assert_eq!(info.format, Some("HFS"));
        assert_eq!(
            info.total_bytes,
            Some(u64::from(m.total_blocks) * u64::from(m.block_size))
        );
        assert_eq!(
            info.free_bytes,
            Some(u64::from(m.free_blocks) * u64::from(m.block_size))
        );
        assert!(info.resource_forks);
        assert!(info.journal.is_none());
        assert!(info.local_times);
        assert_eq!(fs.root(), NodeId(2));
    }

    #[test]
    fn name_falls_back_to_the_master_directory_block() {
        let built = standard().without_root_thread().finish();
        let fs = mounted(&built);
        assert_eq!(fs.info().name, "Atlas HD");
        // The root record is still reachable by its fixed parent.
        let meta = fs.metadata(fs.root()).unwrap();
        assert_eq!(meta.kind, NodeKind::Directory);
    }

    #[test]
    fn identity_is_stable_and_change_follows_the_write_count() {
        let built = standard_volume();
        let a = mounted(&built).identity().unwrap();
        let b = mounted(&built).identity().unwrap();
        assert_eq!(a, b);
        assert_eq!(&a.id[..4], &CREATED.to_be_bytes());
        assert_eq!(&a.id[10..16], b"Atlas ");
        assert_eq!(&a.change[4..8], &MODIFIED.to_be_bytes());

        let mut bytes = built.image.clone();
        let at = 1024 + mdb::offset::WR_CNT;
        bytes[at..at + 4].copy_from_slice(&77u32.to_be_bytes());
        let c = mounted_bytes(bytes).unwrap().identity().unwrap();
        assert_eq!(c.id, a.id);
        assert_ne!(c.change, a.change);
        assert_eq!(&c.change[..4], &77u32.to_be_bytes());
    }

    #[test]
    fn a_short_volume_name_is_zero_padded_in_the_identity() {
        let built = test_volume::VolumeBuilder::new("Ab").finish();
        let id = mounted(&built).identity().unwrap().id;
        assert_eq!(&id[10..], b"Ab\0\0\0\0");
    }

    // ---- listing and metadata ----

    #[test]
    fn a_recursive_walk_equals_the_manifest() {
        for built in [
            standard_volume(),
            standard().block_size(1024).finish(),
            standard().block_size(2048).finish(),
        ] {
            let fs = mounted(&built);
            let walked = walk(&fs);
            let want: Vec<&str> = built
                .manifest
                .entries
                .iter()
                .map(|e| e.path.as_str())
                .collect();
            let mut want_sorted = want.clone();
            want_sorted.sort_unstable();
            assert_eq!(walked.keys().map(String::as_str).collect::<Vec<_>>(), {
                let mut v = want_sorted;
                v.dedup();
                v
            });
            for spec in &built.manifest.entries {
                check_listing(&walked[&spec.path], spec);
            }
            // The invisible file and the alias carry their flags.
            let desktop = built.manifest.by_id(ids::DESKTOP).unwrap();
            assert!(flags_of(desktop).contains(EntryFlags::HIDDEN));
            assert_ne!(desktop.finder_flags & FINDER_INVISIBLE, 0);
            let alias = built.manifest.by_id(ids::README_ALIAS).unwrap();
            assert!(flags_of(alias).contains(EntryFlags::ALIAS));
            assert_ne!(alias.finder_flags & FINDER_ALIAS, 0);
            // No dot-file hiding.
            let dot = &walked["/.profile"];
            assert!(!dot.flags.contains(EntryFlags::HIDDEN));
        }
    }

    #[test]
    fn metadata_of_every_entry_after_listing() {
        let built = standard_volume();
        let fs = mounted(&built);
        let walked = walk(&fs);
        assert_eq!(walked.len(), built.manifest.entries.len());
        for spec in &built.manifest.entries {
            let meta = fs.metadata(NodeId(u64::from(spec.id))).unwrap();
            assert_eq!(meta.node, NodeId(u64::from(spec.id)));
            assert_eq!(meta.created, hfs_time(spec.created), "{}", spec.path);
            assert_eq!(meta.modified, hfs_time(spec.modified), "{}", spec.path);
            assert_eq!(meta.accessed, None);
            assert_eq!(meta.flags, flags_of(spec), "{}", spec.path);
            assert_eq!(meta.link_target, None);
            assert_eq!(meta.owner, None);
            assert_eq!(meta.attributes, None);
            if spec.is_dir {
                assert_eq!(meta.kind, NodeKind::Directory);
                assert_eq!(meta.size, None);
                assert_eq!(meta.resource_size, None);
                assert_eq!(meta.finder_codes, None);
            } else {
                assert_eq!(meta.kind, NodeKind::File);
                assert_eq!(meta.size, Some(spec.data_len), "{}", spec.path);
                assert_eq!(meta.resource_size, Some(spec.rsrc_len));
                let codes =
                    (spec.file_type != [0; 4] || spec.creator != [0; 4]).then_some(FinderCodes {
                        file_type: spec.file_type,
                        creator: spec.creator,
                    });
                assert_eq!(meta.finder_codes, codes, "{}", spec.path);
                let mut info = [0u8; 32];
                info[0..4].copy_from_slice(&spec.file_type);
                info[4..8].copy_from_slice(&spec.creator);
                info[8..10].copy_from_slice(&spec.finder_flags.to_be_bytes());
                let want = info.iter().any(|&b| b != 0).then_some(info);
                assert_eq!(meta.finder_info, want, "{}", spec.path);
            }
        }
        let simple = fs
            .metadata(NodeId(u64::from(ids::SIMPLETEXT)))
            .unwrap()
            .finder_codes
            .unwrap();
        assert_eq!(&simple.file_type, b"APPL");
        assert_eq!(&simple.creator, b"ttxt");
    }

    #[test]
    fn root_metadata() {
        let fs = mounted(&standard_volume());
        let meta = fs.metadata(fs.root()).unwrap();
        assert_eq!(meta.kind, NodeKind::Directory);
        assert_eq!(meta.size, None);
        assert_eq!(meta.created, hfs_time(CREATED));
    }

    #[test]
    fn a_threadless_file_needs_its_listing_but_a_threaded_one_does_not() {
        let built = standard_volume();
        let fs = mounted(&built);
        let readme = NodeId(u64::from(ids::README));
        assert!(matches!(
            fs.metadata(readme),
            Err(Error::NotFound { node }) if node == readme.0
        ));
        // A file with a thread record is found from nothing.
        let john = fs.metadata(NodeId(u64::from(ids::DEAR_JOHN))).unwrap();
        assert_eq!(john.size, Some(700));
        // Listing its folder teaches the map.
        list(&fs, u64::from(ids::DOCUMENTS)).unwrap();
        assert_eq!(fs.metadata(readme).unwrap().size, Some(1200));
    }

    #[test]
    fn lookup_is_case_insensitive_and_accent_sensitive() {
        let fs = mounted(&standard_volume());
        let id = |dir: u32, name: &str| {
            fs.lookup(NodeId(u64::from(dir)), name)
                .unwrap()
                .map(|n| n.0)
        };
        assert_eq!(id(2, "documents"), Some(u64::from(ids::DOCUMENTS)));
        assert_eq!(id(2, "SIMPLETEXT"), Some(u64::from(ids::SIMPLETEXT)));
        assert_eq!(
            id(ids::LETTERS, "R\u{e9}sum\u{e9}"),
            Some(u64::from(ids::RESUME_ACCENT))
        );
        assert_eq!(
            id(ids::LETTERS, "r\u{c9}SUM\u{c9}"),
            Some(u64::from(ids::RESUME_ACCENT))
        );
        assert_eq!(
            id(ids::LETTERS, "resume"),
            Some(u64::from(ids::RESUME_PLAIN))
        );
        assert_eq!(id(ids::LETTERS, "Resum\u{e9}"), None);
        assert_eq!(id(2, "nope"), None);
        assert_eq!(id(2, ""), None);
        // Not in this folder; not a folder; no such folder.
        assert_eq!(id(2, "ReadMe"), None);
        assert_eq!(id(ids::README, "x"), None);
        assert_eq!(id(9999, "x"), None);
    }

    #[test]
    fn lookup_then_metadata_of_a_threadless_file() {
        let fs = mounted(&standard_volume());
        let found = fs
            .lookup(NodeId(u64::from(ids::DOCUMENTS)), "readme")
            .unwrap()
            .unwrap();
        assert_eq!(found, NodeId(u64::from(ids::README)));
        assert_eq!(fs.metadata(found).unwrap().size, Some(1200));
    }

    #[test]
    fn bad_node_ids() {
        let fs = mounted(&standard_volume());
        for node in [0, u64::from(u32::MAX) + 1, u64::MAX] {
            let n = NodeId(node);
            assert!(matches!(fs.metadata(n), Err(Error::NotFound { node: x }) if x == node));
            assert!(matches!(fs.read_dir(n), Err(Error::NotFound { node: x }) if x == node));
            assert!(matches!(fs.lookup(n, "a"), Err(Error::NotFound { node: x }) if x == node));
            assert!(matches!(
                fs.open(n, Stream::Data),
                Err(Error::NotFound { .. })
            ));
            assert!(matches!(
                fs.extents(n, Stream::Data),
                Err(Error::NotFound { .. })
            ));
        }
        assert!(matches!(
            fs.metadata(NodeId(9999)),
            Err(Error::NotFound { node: 9999 })
        ));
        assert!(matches!(
            fs.read_dir(NodeId(9999)),
            Err(Error::NotFound { node: 9999 })
        ));
        // A file with a thread is not a folder.
        assert!(matches!(
            fs.read_dir(NodeId(u64::from(ids::DEAR_JOHN))),
            Err(Error::NotADirectory { .. })
        ));
    }

    // ---- content ----

    fn check_forks(built: &Built) {
        let fs = mounted(built);
        walk(&fs);
        for spec in built.manifest.files() {
            let node = NodeId(u64::from(spec.id));
            let data = fs.open(node, Stream::Data).unwrap();
            assert_eq!(read_all(&data), spec.data_bytes(), "{}", spec.path);
            if spec.rsrc_len == 0 {
                assert!(matches!(
                    fs.open(node, Stream::Resource),
                    Err(Error::NoSuchStream { .. })
                ));
            } else {
                let rsrc = fs.open(node, Stream::Resource).unwrap();
                assert_eq!(read_all(&rsrc), spec.rsrc_bytes(), "{}", spec.path);
            }
        }
        for spec in built.manifest.entries.iter().filter(|e| e.is_dir) {
            let node = NodeId(u64::from(spec.id));
            assert!(matches!(
                fs.open(node, Stream::Data),
                Err(Error::IsADirectory { .. })
            ));
            assert!(matches!(
                fs.extents(node, Stream::Data),
                Err(Error::IsADirectory { .. })
            ));
        }
    }

    #[test]
    fn every_fork_reads_back() {
        check_forks(&standard_volume());
    }

    #[test]
    fn forks_read_back_with_larger_allocation_blocks() {
        for bs in [1024, 2048] {
            check_forks(&standard().block_size(bs).finish());
        }
    }

    #[test]
    fn extents_are_volume_byte_ranges_of_the_same_data() {
        for bs in [512, 2048] {
            let built = standard().block_size(bs).finish();
            let fs = mounted(&built);
            walk(&fs);
            for spec in built.manifest.files() {
                let node = NodeId(u64::from(spec.id));
                for (stream, want) in [
                    (Stream::Data, spec.data_bytes()),
                    (Stream::Resource, spec.rsrc_bytes()),
                ] {
                    let runs = fs.extents(node, stream).unwrap();
                    let mut got = Vec::new();
                    for run in &runs {
                        assert!(run.len > 0);
                        assert_eq!((run.offset - built.manifest.alloc_start) % u64::from(bs), 0);
                        let at = usize::try_from(run.offset).unwrap();
                        let len = usize::try_from(run.len).unwrap();
                        got.extend_from_slice(&built.image[at..at + len]);
                    }
                    assert!(got.len() >= want.len(), "{}", spec.path);
                    assert_eq!(got[..want.len()], want[..], "{}", spec.path);
                    if want.is_empty() {
                        continue;
                    }
                    // Whole blocks only: no more than the fork's blocks.
                    assert!(got.len() < want.len() + usize::try_from(bs).unwrap() * 2);
                }
            }
            let fragmented = fs
                .extents(NodeId(u64::from(ids::FRAGMENTED)), Stream::Data)
                .unwrap();
            if bs == 512 {
                assert_eq!(fragmented.len(), 8);
            }
        }
    }

    #[test]
    fn allocation_map_matches_the_free_blocks() {
        for bs in [512, 1024] {
            let built = standard().block_size(bs).finish();
            let m = &built.manifest;
            let fs = mounted(&built);
            let total = usize::try_from(m.total_blocks).unwrap();
            let map = fs.allocation_map(total).unwrap();
            assert_eq!(map.start, m.alloc_start);
            assert_eq!(map.len, u64::from(m.total_blocks) * u64::from(bs));
            assert_eq!(map.density.len(), total);
            let used = map.density.iter().filter(|&&d| d == 255).count();
            assert_eq!(used, total - usize::try_from(m.free_blocks).unwrap());
            assert!(map.density.iter().all(|&d| d == 0 || d == 255));
            // Folded into fewer buckets, the same blocks are counted.
            let coarse = fs.allocation_map(4).unwrap();
            assert_eq!(coarse.density.len(), 4);
            assert!(coarse.density.iter().any(|&d| d > 0));
        }
    }

    // ---- scan ----

    #[test]
    fn scan_equals_the_walk() {
        let built = standard_volume();
        let fs = mounted(&built);
        let walked = walk(&fs);
        // path by node id, to turn the scan's parent ids into paths
        let paths: BTreeMap<u64, &str> = walked.iter().map(|(p, l)| (l.id, p.as_str())).collect();
        let mut seen = BTreeMap::new();
        let mut types = BTreeMap::new();
        let summary = fs
            .scan(&mut |e| {
                let parent = if e.parent == fs.root() {
                    ""
                } else {
                    paths[&e.parent.0]
                };
                let path = format!("{parent}/{}", e.name);
                types.insert(path.clone(), e.finder_type);
                let listed = Listed {
                    id: e.node.0,
                    name: e.name.to_owned(),
                    kind: e.kind,
                    size: e.size,
                    resource_size: e.resource_size,
                    modified: e.modified,
                    flags: e.flags,
                };
                assert!(seen.insert(path, listed).is_none());
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(seen, walked);
        assert_eq!(summary.entries, walked.len() as u64);
        assert_eq!(summary.skipped, 0);
        assert_eq!(types["/SimpleText"], Some(*b"APPL"));
        assert_eq!(types["/Documents"], None);
        assert_eq!(types["/Documents/ReadMe"], Some(*b"TEXT"));
    }

    #[test]
    fn scan_stops_when_the_visitor_breaks() {
        let fs = mounted(&standard_volume());
        let mut n = 0;
        let summary = fs
            .scan(&mut |_| {
                n += 1;
                if n == 5 {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })
            .unwrap();
        assert_eq!(n, 5);
        assert_eq!(summary.entries, 5);
    }

    // ---- damage ----

    #[test]
    fn mounts_through_the_alternate_block_when_the_primary_is_smashed() {
        let built = standard_volume();
        let mut bytes = built.image.clone();
        bytes[1024..1024 + 162].fill(0xA5);
        let fs = mounted_bytes(bytes).unwrap();
        assert_eq!(fs.info().name, "Atlas HD");
        assert_eq!(walk(&fs).len(), built.manifest.entries.len());
    }

    #[test]
    fn both_master_directory_blocks_smashed_fails_the_mount() {
        let mut bytes = standard_volume().image;
        bytes[1024..1024 + 162].fill(0xA5);
        let alt = bytes.len() - 1024;
        bytes[alt..alt + 162].fill(0xA5);
        assert!(matches!(mounted_bytes(bytes), Err(Error::Corrupt(_))));
        assert!(mounted_bytes(Vec::new()).is_err());
        assert!(mounted_bytes(vec![0u8; 8192]).is_err());
    }

    /// Byte offset in the image of catalog node `n` (512-byte nodes and
    /// allocation blocks).
    fn catalog_node_at(built: &Built, n: u32) -> usize {
        let mut skip = n;
        for &(start, count) in &built.manifest.catalog_extents {
            if skip < count {
                return usize::try_from(built.manifest.alloc_start).unwrap()
                    + usize::try_from(start + skip).unwrap() * 512;
            }
            skip -= count;
        }
        panic!("node {n} is past the catalog");
    }

    #[test]
    fn a_smashed_catalog_node_gives_errors_never_panics() {
        let built = standard_volume();
        let nodes = built.manifest.catalog_nodes;
        assert!(nodes > 4);
        for n in 1..nodes {
            for fill in [0xFFu8, 0x00] {
                let mut bytes = built.image.clone();
                let at = catalog_node_at(&built, n);
                bytes[at..at + 512].fill(fill);
                let Ok(fs) = mounted_bytes(bytes) else {
                    continue;
                };
                // Every call returns; errors are fine.
                let mut stack = vec![ROOT];
                let mut visited = 0;
                while let Some(dir) = stack.pop() {
                    visited += 1;
                    assert!(visited < 1000);
                    let Ok(mut stream) = fs.read_dir(NodeId(dir)) else {
                        continue;
                    };
                    let mut batch = DirBatch::new();
                    loop {
                        batch.clear();
                        let progress = stream.next_batch(&mut batch, 2);
                        for e in batch.iter() {
                            if e.kind == NodeKind::Directory {
                                stack.push(e.node.0);
                            }
                            let _ = fs.metadata(e.node);
                            let _ = fs.open(e.node, Stream::Data);
                            let _ = fs.extents(e.node, Stream::Resource);
                            let _ = fs.lookup(NodeId(dir), e.name);
                        }
                        if !matches!(progress, Ok(DirProgress::More)) {
                            break;
                        }
                    }
                }
                let _ = fs.scan(&mut |_| ControlFlow::Continue(()));
                let _ = fs.allocation_map(16);
            }
        }
    }

    #[test]
    fn a_catalog_leaf_with_garbage_records_is_an_error_in_the_listing() {
        let built = standard_volume();
        // The leaf holding Documents' siblings: smash every leaf but keep
        // the header and index nodes by trying each node and expecting that
        // at least one smash makes a listing fail.
        let mut failed = false;
        for n in 1..built.manifest.catalog_nodes {
            let mut bytes = built.image.clone();
            let at = catalog_node_at(&built, n);
            bytes[at..at + 512].fill(0xFF);
            if let Ok(fs) = mounted_bytes(bytes)
                && (list(&fs, ROOT).is_err() || list(&fs, u64::from(ids::BIG_FOLDER)).is_err())
            {
                failed = true;
            }
        }
        assert!(failed);
    }

    #[test]
    fn mount_with_decodes_names_in_the_chosen_encoding() {
        // Byte 0x80 is A-umlaut in Roman and a different letter in Cyrillic.
        let roman = mounted(&standard_volume());
        let names = list(&roman, u64::from(ids::NAMES)).unwrap();
        let built = standard_volume();
        let cyr = mount_with(
            Arc::new(MemSource::new(built.image.clone())),
            Encoding::Cyrillic,
        )
        .unwrap();
        let other = list(&cyr, u64::from(ids::NAMES)).unwrap();
        assert_eq!(names.len(), other.len());
        assert_ne!(names[0].name, other[0].name);
    }
}
