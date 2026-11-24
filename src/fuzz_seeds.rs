// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Writes seed inputs for the classic HFS fuzz targets in `fuzz/`.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use super::catalog::{
    CatalogKey, CatalogRecord, FileRecord, FolderRecord, ROOT_FOLDER_ID, ThreadKind, ThreadRecord,
    encode_key,
};
use super::extents::{
    EXTENT_RECORD_LEN, ExtentKey, HfsExtent, HfsExtentRecord, parse_extent_record,
    write_extent_record,
};
use super::mdb::{MDB_LEN, MDB_OFFSET, Mdb};
use super::test_volume::{self, standard, standard_volume};
use atlas_fs_core::fs::FileSystem;
use atlas_fs_core::io::block::MemSource;

/// The largest input the fuzz run accepts (`-max_len=1048576`).
const MAX_SEED: usize = 1 << 20;

/// Volumes for the mount target: the standard one, one with 1024-byte
/// allocation blocks, one without the root thread and one whose catalog is
/// in nine pieces (so six of its extents are overflow records).
fn volumes() -> Vec<(&'static str, Vec<u8>)> {
    let mut big_blocks = standard();
    big_blocks.block_size(1024);
    let mut no_thread = standard();
    no_thread.without_root_thread();
    let mut fragmented = standard();
    fragmented.catalog_fragments(9);
    vec![
        ("hfs-standard", standard_volume().image),
        ("hfs-1k-blocks", big_blocks.build()),
        ("hfs-no-root-thread", no_thread.build()),
        ("hfs-fragmented-catalog", fragmented.build()),
    ]
}

/// The raw-record seeds, named. The target feeds each input to every parser
/// from byte 0, so each seed is the bytes one parser reads.
fn records() -> Vec<(&'static str, Vec<u8>)> {
    let image = standard_volume().image;
    let at = usize::try_from(MDB_OFFSET).unwrap();
    let mdb = image[at..at + MDB_LEN].to_vec();

    let folder = FolderRecord {
        flags: 0,
        valence: 3,
        folder_id: 16,
        created: test_volume::CREATED,
        modified: test_volume::MODIFIED,
        backed_up: 0,
        finder_info: [0x11; 16],
        ext_finder_info: [0x22; 16],
        reserved: [0; 16],
    };
    let file = FileRecord {
        flags: 0,
        file_kind: 0,
        finder_info: *b"TEXTttxt\0\0\0\0\0\0\0\0",
        file_id: 22,
        data_start_block: 7,
        data_logical: 1200,
        data_physical: 1536,
        rsrc_start_block: 11,
        rsrc_logical: 300,
        rsrc_physical: 512,
        created: test_volume::CREATED,
        modified: test_volume::MODIFIED,
        backed_up: 0,
        ext_finder_info: [0; 16],
        clump_size: 0,
        data_extents: extent_record(),
        rsrc_extents: [HfsExtent::default(); 3],
        reserved: 0,
    };
    let thread = ThreadRecord {
        kind: ThreadKind::Folder,
        reserved: [0; 8],
        parent_id: ROOT_FOLDER_ID,
        name: b"Documents".to_vec(),
    };
    let mut ext_record = [0u8; EXTENT_RECORD_LEN];
    write_extent_record(&extent_record(), &mut ext_record).unwrap();
    let ext_key = ExtentKey {
        fork_type: 0,
        file_id: 25,
        start_block: 3,
    };
    vec![
        ("hfs-mdb", mdb),
        ("hfs-folder-record", CatalogRecord::Folder(folder).encode()),
        ("hfs-file-record", CatalogRecord::File(file).encode()),
        (
            "hfs-thread-record",
            CatalogRecord::FolderThread(thread).encode(),
        ),
        ("hfs-extent-key", ext_key.encode()[1..].to_vec()),
        ("hfs-extent-record", ext_record.to_vec()),
        (
            "hfs-catalog-key",
            encode_key(ROOT_FOLDER_ID, b"Documents")[1..].to_vec(),
        ),
    ]
}

fn extent_record() -> HfsExtentRecord {
    [
        HfsExtent {
            start_block: 7,
            block_count: 3,
        },
        HfsExtent {
            start_block: 11,
            block_count: 1,
        },
        HfsExtent::default(),
    ]
}

/// Writes seed files for the classic HFS targets in `fuzz/` into the
/// directory named by `ATLAS_FUZZ_SEEDS`, one subfolder per target. Does
/// nothing when the variable is unset. Run it with
/// `cargo test --lib write_fuzz_seeds -- --ignored`. Disk Atlas writes the
/// `dc42` and `probe` seeds itself.
#[test]
#[ignore = "writes files; run by the fuzz CI job"]
fn write_fuzz_seeds() {
    let Some(root) = std::env::var_os("ATLAS_FUZZ_SEEDS") else {
        return;
    };
    let root = Path::new(&root);
    let mount_dir = root.join("hfs_mount");
    let records_dir = root.join("hfs_records");
    for dir in [&mount_dir, &records_dir] {
        fs::create_dir_all(dir).expect("create seed directory");
    }
    for (name, image) in volumes() {
        fs::write(mount_dir.join(name), &image).expect("write mount seed");
    }
    for (name, bytes) in records() {
        fs::write(records_dir.join(name), bytes).expect("write record seed");
    }
}

/// Every mount seed fits the run's input limit, mounts and lists its root,
/// so the target starts on the valid path.
#[test]
fn mount_seeds_are_valid_and_small() {
    for (name, image) in volumes() {
        assert!(image.len() <= MAX_SEED, "{name}: {} bytes", image.len());
        let volume =
            super::mount(Arc::new(MemSource::new(image))).unwrap_or_else(|e| panic!("{name}: {e}"));
        // Without a root thread the root cannot be listed, only described.
        if name == "hfs-no-root-thread" {
            assert!(volume.metadata(volume.root()).is_ok(), "{name}");
        } else {
            assert!(volume.read_dir(volume.root()).is_ok(), "{name}");
        }
    }
}

/// Every record seed parses with the parser it is meant for and survives the
/// round trip the target asserts.
#[test]
fn record_seeds_parse() {
    for (name, bytes) in records() {
        match name {
            "hfs-mdb" => {
                let mdb = Mdb::parse(&bytes).expect(name);
                mdb.validate(1024, standard_volume().image.len() as u64)
                    .expect(name);
            }
            "hfs-extent-key" => assert!(ExtentKey::parse(&bytes).is_some(), "{name}"),
            "hfs-extent-record" => {
                assert_eq!(parse_extent_record(&bytes), Some(extent_record()), "{name}")
            }
            "hfs-catalog-key" => assert!(CatalogKey::parse(&bytes).is_some(), "{name}"),
            _ => {
                let record =
                    CatalogRecord::parse(&bytes).unwrap_or_else(|e| panic!("{name}: {e:?}"));
                assert_eq!(record.encode(), bytes, "{name}");
            }
        }
    }
}
