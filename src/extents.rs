// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Classic HFS extents: the 8-byte overflow key, three-extent records and
//! fork mapping through the extents overflow B-tree.

use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};
use atlas_hfsplus::btree::BTree;
use atlas_hfsplus::bytes::{be_u16, be_u32, u8_at};
use atlas_hfsplus::extents::{ExtentDescriptor, ForkMap, ForkType, Geometry};

/// Bytes in one extent descriptor (`HFSExtentDescriptor`).
pub const EXTENT_LEN: usize = 4;

/// Extent descriptors in one extent record (`HFSExtentRecord`).
pub const EXTENTS_PER_RECORD: usize = 3;

/// Bytes in one extent record.
pub const EXTENT_RECORD_LEN: usize = EXTENT_LEN * EXTENTS_PER_RECORD;

/// One run of allocation blocks (`HFSExtentDescriptor`): big-endian
/// `startBlock` u16 then `blockCount` u16. A count of 0 is an unused slot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct HfsExtent {
    /// First allocation block of the run.
    pub start_block: u16,
    /// Allocation blocks in the run.
    pub block_count: u16,
}

/// Three extent descriptors (`HFSExtentRecord`).
pub type HfsExtentRecord = [HfsExtent; EXTENTS_PER_RECORD];

/// Parses an extent record from the first [`EXTENT_RECORD_LEN`] bytes of
/// `b`. `None` when `b` is shorter. Values are not validated here.
pub fn parse_extent_record(b: &[u8]) -> Option<HfsExtentRecord> {
    let mut record = [HfsExtent::default(); EXTENTS_PER_RECORD];
    for (i, slot) in record.iter_mut().enumerate() {
        let at = i.checked_mul(EXTENT_LEN)?;
        *slot = HfsExtent {
            start_block: be_u16(b, at)?,
            block_count: be_u16(b, at.checked_add(2)?)?,
        };
    }
    Some(record)
}

/// Writes `record` into the first [`EXTENT_RECORD_LEN`] bytes of `out`, the
/// inverse of [`parse_extent_record`]. `None` (and `out` unchanged) when
/// `out` is shorter.
pub fn write_extent_record(record: &HfsExtentRecord, out: &mut [u8]) -> Option<()> {
    let out = out.get_mut(..EXTENT_RECORD_LEN)?;
    let bytes = record.iter().flat_map(|e| {
        let [s0, s1] = e.start_block.to_be_bytes();
        let [c0, c1] = e.block_count.to_be_bytes();
        [s0, s1, c0, c1]
    });
    for (dst, src) in out.iter_mut().zip(bytes) {
        *dst = src;
    }
    Some(())
}

/// File id of the extents overflow file.
pub const EXTENTS_FILE_ID: u32 = 3;

/// File id of the catalog file.
pub const CATALOG_FILE_ID: u32 = 4;

/// Bound on the overflow records one fork may use: a fork cannot have more
/// runs than the volume has blocks (at most 65,536), three runs a record.
pub const MAX_EXTENT_RECORDS: u32 = 65_536 / 3 + 1;

/// `keyLength` of an extents overflow key.
pub const EXTENT_KEY_LEN: u8 = 7;

/// Structure name for errors in a fork's mapping.
const FORK: &str = "fork";

/// Structure name for errors in extents overflow records.
const OVERFLOW: &str = "extents overflow B-tree";

/// An extents overflow key (`HFSExtentKey`) without its length byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExtentKey {
    /// `0x00` data fork, `0xFF` resource fork.
    pub fork_type: u8,
    /// The file's catalog node id.
    pub file_id: u32,
    /// Fork-relative allocation block the record starts at.
    pub start_block: u16,
}

impl ExtentKey {
    /// From the key bytes after `keyLength`: `forkType` u8, `fileID` u32,
    /// `startBlock` u16 (7 bytes). `None` when `key` is shorter; bytes past
    /// the seventh are ignored.
    pub fn parse(key: &[u8]) -> Option<Self> {
        Some(Self {
            fork_type: u8_at(key, 0)?,
            file_id: be_u32(key, 1)?,
            start_block: be_u16(key, 5)?,
        })
    }

    /// The `keyLength` byte ([`EXTENT_KEY_LEN`]) then the 7 key bytes: what
    /// a writer puts at the start of a leaf or index record. 8 bytes is even,
    /// so the record's data follows with no pad byte.
    pub fn encode(&self) -> [u8; 8] {
        let mut out = [0u8; 8];
        out[0] = EXTENT_KEY_LEN;
        out[1] = self.fork_type;
        out[2..6].copy_from_slice(&self.file_id.to_be_bytes());
        out[6..8].copy_from_slice(&self.start_block.to_be_bytes());
        out
    }

    /// Comparison order: file id, then fork type, then start block (as the
    /// Linux kernel's `hfs_ext_keycmp`; same order as HFS+).
    pub fn order(&self) -> (u32, u8, u16) {
        (self.file_id, self.fork_type, self.start_block)
    }
}

/// The `forkType` byte of a fork type.
fn fork_type_byte(t: ForkType) -> u8 {
    match t {
        ForkType::Data => 0x00,
        ForkType::Resource => 0xFF,
    }
}

fn len64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

fn corrupt(structure: &'static str, file_id: u32, kind: CorruptKind) -> Error {
    Error::Corrupt(Corruption::new(
        structure,
        Location::Object(u64::from(file_id)),
        kind,
    ))
}

/// The key of the overflow record that must continue a fork mapped up to
/// `mapped` blocks. A key's start block is 16 bits: a fork mapped past that
/// cannot have a further record.
fn expected_key(file_id: u32, fork_type: u8, mapped: u32) -> Result<ExtentKey> {
    let start_block = u16::try_from(mapped).map_err(|_| {
        corrupt(
            FORK,
            file_id,
            CorruptKind::OutOfRange {
                field: "mapped blocks",
                value: u64::from(mapped),
                limit_name: "extent key start block",
                limit: u64::from(u16::MAX),
            },
        )
    })?;
    Ok(ExtentKey {
        fork_type,
        file_id,
        start_block,
    })
}

/// An overflow key shorter than [`EXTENT_KEY_LEN`].
fn short_key(file_id: u32, key: &[u8]) -> Error {
    corrupt(
        OVERFLOW,
        file_id,
        CorruptKind::Truncated {
            needed: u64::from(EXTENT_KEY_LEN),
            available: len64(key.len()),
        },
    )
}

/// Runs collected so far for one fork.
struct Collector {
    geometry: Geometry,
    file_id: u32,
    mapped: u32,
    runs: Vec<ExtentDescriptor>,
}

impl Collector {
    fn error(&self, structure: &'static str, kind: CorruptKind) -> Error {
        corrupt(structure, self.file_id, kind)
    }

    fn missing(&self) -> Error {
        self.error(FORK, CorruptKind::Other("extents overflow record missing"))
    }

    /// Appends the extents of `record` up to the first empty one, widened to
    /// `u32`. Returns the number of blocks added. Each run must end inside
    /// the volume; [`ForkMap::from_runs`] checks again, this reports the
    /// damage before the overflow tree is consulted.
    fn add(&mut self, record: &HfsExtentRecord) -> Result<u32> {
        let before = self.mapped;
        for extent in record.iter().take_while(|e| e.block_count != 0) {
            let run = ExtentDescriptor {
                start_block: u32::from(extent.start_block),
                block_count: u32::from(extent.block_count),
            };
            // Two u16 values: the sum cannot overflow u32.
            let end = run.start_block.saturating_add(run.block_count);
            if end > self.geometry.total_blocks {
                return Err(self.error(
                    FORK,
                    CorruptKind::OutOfRange {
                        field: "extent end",
                        value: u64::from(end),
                        limit_name: "volume blocks",
                        limit: u64::from(self.geometry.total_blocks),
                    },
                ));
            }
            self.mapped = self.mapped.checked_add(run.block_count).ok_or_else(|| {
                self.error(
                    FORK,
                    CorruptKind::Overflow {
                        field: "mapped blocks",
                    },
                )
            })?;
            self.runs.push(run);
        }
        Ok(self.mapped.saturating_sub(before))
    }
}

/// Maps `fork_type` of file `file_id`: the runs of `first` up to its first
/// empty extent, then records from the extents overflow tree while the
/// mapped blocks fall short of `ceil(logical_size / block_size)`.
///
/// `overflow` is `None` for the extents file itself, which cannot overflow;
/// a fork short of its size with no tree fails with "extents overflow
/// record missing", and the caller decides whether that is fatal. Mapping
/// more blocks than the logical size needs is allowed (classic HFS
/// preallocates by clump); every run must still end inside
/// `geometry.total_blocks`.
pub fn fork_map(
    geometry: Geometry,
    file_id: u32,
    fork_type: ForkType,
    logical_size: u64,
    first: &HfsExtentRecord,
    overflow: Option<&BTree>,
) -> Result<ForkMap> {
    let mut c = Collector {
        geometry,
        file_id,
        mapped: 0,
        runs: Vec::with_capacity(EXTENTS_PER_RECORD),
    };
    if geometry.block_size == 0 {
        return Err(c.error(
            FORK,
            CorruptKind::InvalidValue {
                field: "allocation block size",
                value: 0,
            },
        ));
    }
    let target = logical_size.div_ceil(u64::from(geometry.block_size));
    c.add(first)?;

    if u64::from(c.mapped) < target {
        let tree = overflow.ok_or_else(|| c.missing())?;
        let ft = fork_type_byte(fork_type);
        let first_want = expected_key(file_id, ft, c.mapped)?.order();
        let mut cursor = tree.seek(&mut |key: &[u8]| {
            let k = ExtentKey::parse(key).ok_or_else(|| short_key(file_id, key))?;
            Ok(k.order().cmp(&first_want))
        })?;
        let limit = tree.header().leaf_records.min(MAX_EXTENT_RECORDS);
        let mut used: u32 = 0;
        // Bounded by `limit` records; every record must add blocks.
        // Consecutive records of one fork are adjacent in key order, so the
        // cursor walks on from the first instead of seeking again.
        while u64::from(c.mapped) < target {
            if used >= limit {
                return Err(c.error(
                    OVERFLOW,
                    CorruptKind::LimitExceeded {
                        what: "extents overflow records",
                        limit: u64::from(limit),
                    },
                ));
            }
            used = used.saturating_add(1);
            let expected = expected_key(file_id, ft, c.mapped)?;
            let record = {
                let rec = cursor.next_record()?.ok_or_else(|| c.missing())?;
                let key = ExtentKey::parse(rec.key).ok_or_else(|| short_key(file_id, rec.key))?;
                if key != expected {
                    return Err(c.missing());
                }
                parse_extent_record(rec.data).ok_or_else(|| {
                    c.error(
                        OVERFLOW,
                        CorruptKind::Truncated {
                            needed: len64(EXTENT_RECORD_LEN),
                            available: len64(rec.data.len()),
                        },
                    )
                })?
            };
            if c.add(&record)? == 0 {
                return Err(c.error(
                    FORK,
                    CorruptKind::Other("extents overflow record maps no blocks"),
                ));
            }
        }
    }

    ForkMap::from_runs(file_id, logical_size, c.runs, geometry)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use atlas_fs_core::io::block::{BlockSource, MemSource};
    use atlas_hfsplus::btree::HeaderRecord;
    use atlas_hfsplus::btree::fixtures::{INDEX, LEAF, header, header_node, node, tree};
    use atlas_hfsplus::extents::ForkSource;

    #[test]
    fn extent_record_round_trip_and_short_input() {
        let bytes: Vec<u8> = (1..=12).collect();
        let record = parse_extent_record(&bytes).unwrap();
        assert_eq!(
            record[0],
            HfsExtent {
                start_block: 0x0102,
                block_count: 0x0304
            }
        );
        assert_eq!(record[2].block_count, 0x0B0C);
        let mut out = [0u8; EXTENT_RECORD_LEN];
        write_extent_record(&record, &mut out).unwrap();
        assert_eq!(&out[..], &bytes[..]);
        assert_eq!(parse_extent_record(&bytes[..11]), None);
        assert_eq!(write_extent_record(&record, &mut [0u8; 11]), None);
    }

    const BS: usize = 512;
    const FILE: u32 = 20;
    const DATA: u8 = 0x00;
    const RSRC: u8 = 0xFF;

    fn geometry() -> Geometry {
        Geometry {
            block_size: 512,
            total_blocks: 64,
        }
    }

    /// A 64-block area where every byte of block `n` is `n`.
    fn volume() -> Arc<dyn BlockSource> {
        let bytes: Vec<u8> = (0..64u8).flat_map(|b| vec![b; BS]).collect();
        Arc::new(MemSource::new(bytes))
    }

    /// An extent record from up to three (start, count) pairs, the rest empty.
    fn rec(extents: &[(u16, u16)]) -> HfsExtentRecord {
        let mut r = [HfsExtent::default(); EXTENTS_PER_RECORD];
        for (slot, &(start_block, block_count)) in r.iter_mut().zip(extents) {
            *slot = HfsExtent {
                start_block,
                block_count,
            };
        }
        r
    }

    fn ext(start_block: u32, block_count: u32) -> ExtentDescriptor {
        ExtentDescriptor {
            start_block,
            block_count,
        }
    }

    fn key(fork_type: u8, file_id: u32, start_block: u16) -> ExtentKey {
        ExtentKey {
            fork_type,
            file_id,
            start_block,
        }
    }

    /// A leaf record: 8-bit key length, 7-byte key, extent record. 1 + 7 is
    /// even, so no pad byte.
    fn orec(fork_type: u8, file_id: u32, start: u16, extents: &[(u16, u16)]) -> Vec<u8> {
        let mut r = key(fork_type, file_id, start).encode().to_vec();
        let mut data = [0u8; EXTENT_RECORD_LEN];
        write_extent_record(&rec(extents), &mut data).unwrap();
        r.extend_from_slice(&data);
        r
    }

    /// An index record of a tree without variable index keys: the key
    /// occupies max_key_length (7) bytes, then the child node number.
    fn oindex(fork_type: u8, file_id: u32, start: u16, child: u32) -> Vec<u8> {
        let mut r = key(fork_type, file_id, start).encode().to_vec();
        r.extend_from_slice(&child.to_be_bytes());
        r
    }

    /// A classic HFS extents tree header: 8-bit keys, fixed index keys of 7
    /// bytes, 512-byte nodes.
    fn hfs_header(depth: u16, root: u32, first: u32, last: u32, total: u32) -> HeaderRecord {
        let mut h = header(depth, root, first, last, total);
        h.attributes = 0;
        h.max_key_length = u16::from(EXTENT_KEY_LEN);
        h
    }

    fn leaf(forward: u32, backward: u32, records: &[Vec<u8>]) -> Vec<u8> {
        node(BS, LEAF, 1, forward, backward, records)
    }

    fn one_leaf_tree(records: &[Vec<u8>], leaf_records: u32) -> BTree {
        let mut h = hfs_header(1, 1, 1, 1, 2);
        h.leaf_records = leaf_records;
        tree(&[header_node(&h), leaf(0, 0, records)]).unwrap()
    }

    fn map(
        fork_type: ForkType,
        logical_size: u64,
        first: &[(u16, u16)],
        overflow: Option<&BTree>,
    ) -> Result<ForkMap> {
        fork_map(
            geometry(),
            FILE,
            fork_type,
            logical_size,
            &rec(first),
            overflow,
        )
    }

    fn map_data(
        logical_size: u64,
        first: &[(u16, u16)],
        overflow: Option<&BTree>,
    ) -> Result<ForkMap> {
        map(ForkType::Data, logical_size, first, overflow)
    }

    fn source(m: ForkMap) -> ForkSource {
        ForkSource::new(volume(), Arc::new(m))
    }

    /// Reads the whole fork and checks every byte against `phys`, the
    /// physical block of each logical block.
    fn assert_content(src: &ForkSource, phys: &[u8]) {
        let len = usize::try_from(src.len()).unwrap();
        assert!(len <= phys.len() * BS);
        let mut buf = vec![0u8; len];
        src.read_at(0, &mut buf).unwrap();
        for (pos, &b) in buf.iter().enumerate() {
            assert_eq!(b, phys[pos / BS], "byte {pos}");
        }
    }

    fn corrupt<T: std::fmt::Debug>(r: Result<T>) -> Corruption {
        match r {
            Err(Error::Corrupt(c)) => c,
            Err(e) => panic!("expected Corrupt, got {e:?}"),
            Ok(v) => panic!("expected an error, got {v:?}"),
        }
    }

    fn assert_missing(r: Result<ForkMap>) {
        let c = corrupt(r);
        assert_eq!(c.structure, "fork");
        assert_eq!(c.location, Location::Object(u64::from(FILE)));
        assert!(
            matches!(c.kind, CorruptKind::Other(s) if s == "extents overflow record missing"),
            "{c:?}"
        );
    }

    #[test]
    fn extent_key_round_trip() {
        let bytes = [0xFF, 0x00, 0x00, 0x01, 0x05, 0x12, 0x34];
        let k = ExtentKey::parse(&bytes).unwrap();
        assert_eq!(k, key(0xFF, 0x105, 0x1234));
        let enc = k.encode();
        assert_eq!(enc, [7, 0xFF, 0x00, 0x00, 0x01, 0x05, 0x12, 0x34]);
        assert_eq!(enc[0], EXTENT_KEY_LEN);
        assert_eq!(ExtentKey::parse(&enc[1..]), Some(k));
        assert_eq!(ExtentKey::parse(&bytes[..6]), None);
        assert_eq!(ExtentKey::parse(&[]), None);
    }

    #[test]
    fn extent_key_order_is_file_then_fork_then_block() {
        // File 5's resource fork sorts before file 6's data fork.
        assert!(key(RSRC, 5, 0).order() < key(DATA, 6, 0).order());
        // Within a file, every data record precedes every resource record.
        assert!(key(DATA, 5, 900).order() < key(RSRC, 5, 0).order());
        assert!(key(DATA, 5, 3).order() < key(DATA, 5, 4).order());
        let mut keys = vec![
            key(DATA, 6, 0),
            key(RSRC, 5, 2),
            key(DATA, 5, 7),
            key(RSRC, 5, 0),
            key(DATA, 5, 3),
        ];
        keys.sort_by_key(ExtentKey::order);
        assert_eq!(
            keys,
            [
                key(DATA, 5, 3),
                key(DATA, 5, 7),
                key(RSRC, 5, 0),
                key(RSRC, 5, 2),
                key(DATA, 6, 0),
            ]
        );
    }

    #[test]
    fn constants() {
        assert_eq!(MAX_EXTENT_RECORDS, 21_846);
        assert_eq!((EXTENTS_FILE_ID, CATALOG_FILE_ID), (3, 4));
    }

    #[test]
    fn all_in_the_first_record() {
        let m = map_data(6 * 512 - 100, &[(10, 2), (20, 3), (30, 1)], None).unwrap();
        assert_eq!(
            m.runs().collect::<Vec<_>>(),
            [(0, ext(10, 2)), (1024, ext(20, 3)), (2560, ext(30, 1))]
        );
        assert_content(&source(m), &[10, 11, 20, 21, 22, 30]);
    }

    #[test]
    fn first_record_stops_at_the_first_empty_slot() {
        let m = map_data(1024, &[(10, 2), (0, 0), (30, 1)], None).unwrap();
        assert_eq!(m.runs().collect::<Vec<_>>(), [(0, ext(10, 2))]);
        // The slot after the empty one does not count towards the size.
        assert_missing(map_data(3 * 512, &[(10, 2), (0, 0), (30, 1)], None));
    }

    /// Two leaves under a fixed-key index. File FILE's data fork continues
    /// at fork blocks 3 and 7; its resource fork at 1. Files 5 and 9 sit on
    /// either side.
    fn fragmented_tree() -> BTree {
        let mut h = hfs_header(2, 1, 2, 3, 4);
        h.leaf_records = 5;
        let index = node(
            BS,
            INDEX,
            2,
            0,
            0,
            &[oindex(DATA, 5, 3, 2), oindex(DATA, FILE, 7, 3)],
        );
        tree(&[
            header_node(&h),
            index,
            leaf(
                3,
                0,
                &[
                    orec(DATA, 5, 3, &[(60, 2)]),
                    orec(DATA, FILE, 3, &[(20, 2), (24, 1), (26, 1)]),
                ],
            ),
            leaf(
                0,
                2,
                &[
                    orec(DATA, FILE, 7, &[(40, 3)]),
                    orec(RSRC, FILE, 1, &[(52, 2)]),
                    orec(DATA, 9, 0, &[(5, 1)]),
                ],
            ),
        ])
        .unwrap()
    }

    #[test]
    fn fragmented_fork_through_a_two_level_tree() {
        let t = fragmented_tree();
        let m = map_data(10 * 512 - 1, &[(1, 1), (3, 1), (5, 1)], Some(&t)).unwrap();
        let runs: Vec<_> = m.runs().collect();
        assert_eq!(
            runs,
            [
                (0, ext(1, 1)),
                (512, ext(3, 1)),
                (1024, ext(5, 1)),
                (1536, ext(20, 2)),
                (2560, ext(24, 1)),
                (3072, ext(26, 1)),
                (3584, ext(40, 3)),
            ]
        );
        assert_content(&source(m), &[1, 3, 5, 20, 21, 24, 26, 40, 41, 42]);
    }

    #[test]
    fn resource_fork_takes_its_own_records() {
        // The seek descends through the index into the second leaf.
        let t = fragmented_tree();
        let m = map(ForkType::Resource, 3 * 512, &[(50, 1)], Some(&t)).unwrap();
        assert_eq!(
            m.runs().collect::<Vec<_>>(),
            [(0, ext(50, 1)), (512, ext(52, 2))]
        );
        assert_content(&source(m), &[50, 52, 53]);
    }

    #[test]
    fn data_fork_longer_than_its_records_meets_the_resource_fork() {
        // Mapped 10 blocks, wants 11: the next record is the resource fork's.
        let t = fragmented_tree();
        assert_missing(map_data(11 * 512, &[(1, 1), (3, 1), (5, 1)], Some(&t)));
    }

    #[test]
    fn missing_record() {
        let t = one_leaf_tree(
            &[orec(DATA, 5, 3, &[(30, 2)]), orec(DATA, 9, 3, &[(40, 2)])],
            2,
        );
        assert_missing(map_data(5 * 512, &[(10, 3)], Some(&t)));
        // A tree that ends before the target.
        let t = one_leaf_tree(&[orec(DATA, 5, 3, &[(30, 2)])], 1);
        assert_missing(map_data(5 * 512, &[(10, 3)], Some(&t)));
    }

    #[test]
    fn record_with_the_wrong_start_block() {
        let t = one_leaf_tree(&[orec(DATA, FILE, 4, &[(30, 2)])], 1);
        assert_missing(map_data(5 * 512, &[(10, 3)], Some(&t)));
        // Right first record, wrong second.
        let t = one_leaf_tree(
            &[
                orec(DATA, FILE, 3, &[(30, 2)]),
                orec(DATA, FILE, 6, &[(40, 2)]),
            ],
            2,
        );
        assert_missing(map_data(7 * 512, &[(10, 3)], Some(&t)));
    }

    #[test]
    fn record_adding_zero_blocks() {
        let t = one_leaf_tree(&[orec(DATA, FILE, 3, &[])], 1);
        let c = corrupt(map_data(5 * 512, &[(10, 3)], Some(&t)));
        assert!(matches!(c.kind, CorruptKind::Other(s) if s.contains("maps no blocks")));
        // Extents after an empty first slot do not count either.
        let t = one_leaf_tree(&[orec(DATA, FILE, 3, &[(0, 0), (30, 2)])], 1);
        let c = corrupt(map_data(5 * 512, &[(10, 3)], Some(&t)));
        assert!(matches!(c.kind, CorruptKind::Other(s) if s.contains("maps no blocks")));
    }

    #[test]
    fn run_past_total_blocks() {
        let c = corrupt(map_data(5 * 512, &[(60, 5)], None));
        assert_eq!(c.location, Location::Object(u64::from(FILE)));
        assert!(matches!(
            c.kind,
            CorruptKind::OutOfRange {
                field: "extent end",
                value: 65,
                limit_name: "volume blocks",
                limit: 64,
            }
        ));
        // Ending exactly on the last block is fine.
        assert!(map_data(4 * 512, &[(60, 4)], None).is_ok());
        // The same check applies to overflow records.
        let t = one_leaf_tree(&[orec(DATA, FILE, 3, &[(62, 3)])], 1);
        assert!(matches!(
            corrupt(map_data(6 * 512, &[(10, 3)], Some(&t))).kind,
            CorruptKind::OutOfRange {
                limit_name: "volume blocks",
                value: 65,
                ..
            }
        ));
    }

    #[test]
    fn start_plus_count_past_u16_is_computed_in_u32() {
        let g = Geometry {
            block_size: 512,
            total_blocks: 65_535,
        };
        let c = corrupt(fork_map(
            g,
            FILE,
            ForkType::Data,
            512,
            &rec(&[(0xFFFF, 2)]),
            None,
        ));
        // 65,537, not a wrapped 1.
        assert!(matches!(
            c.kind,
            CorruptKind::OutOfRange {
                field: "extent end",
                value: 65_537,
                limit: 65_535,
                ..
            }
        ));
        // The largest legal run.
        let m = fork_map(g, FILE, ForkType::Data, 512, &rec(&[(0, 0xFFFF)]), None).unwrap();
        assert_eq!(m.runs().collect::<Vec<_>>(), [(0, ext(0, 65_535))]);
    }

    #[test]
    fn mapped_past_a_16_bit_key_start() {
        // Three overlapping full-volume runs map 196,605 blocks; a further
        // record would need a start block past 65,535.
        let g = Geometry {
            block_size: 512,
            total_blocks: 65_535,
        };
        let t = one_leaf_tree(&[], 1);
        let first = rec(&[(0, 0xFFFF), (0, 0xFFFF), (0, 0xFFFF)]);
        let c = corrupt(fork_map(
            g,
            FILE,
            ForkType::Data,
            200_000 * 512,
            &first,
            Some(&t),
        ));
        assert!(matches!(
            c.kind,
            CorruptKind::OutOfRange {
                field: "mapped blocks",
                value: 196_605,
                limit_name: "extent key start block",
                ..
            }
        ));
    }

    #[test]
    fn logical_size_zero_is_an_empty_map() {
        // The tree would fail on any use.
        let t = one_leaf_tree(&[], 0);
        let m = map_data(0, &[], Some(&t)).unwrap();
        assert_eq!(m.runs().count(), 0);
        let src = source(m);
        assert_eq!(src.len(), 0);
        src.read_at(0, &mut []).unwrap();
        assert!(src.read_at(0, &mut [0u8; 1]).is_err());
    }

    #[test]
    fn physical_larger_than_logical_is_accepted() {
        // Clump preallocation: four blocks for 100 bytes.
        let m = map_data(100, &[(10, 4)], None).unwrap();
        assert_eq!(m.runs().collect::<Vec<_>>(), [(0, ext(10, 4))]);
        let src = source(m);
        assert_content(&src, &[10]);
        assert!(matches!(
            src.read_at(0, &mut [0u8; 101]),
            Err(Error::OutOfBounds { .. })
        ));
        // An overflow record may also overshoot the target.
        let t = one_leaf_tree(&[orec(DATA, FILE, 3, &[(30, 5)])], 1);
        let m = map_data(4 * 512, &[(10, 3)], Some(&t)).unwrap();
        assert_eq!(m.runs().last(), Some((1536, ext(30, 5))));
        assert_content(&source(m), &[10, 11, 12, 30]);
    }

    #[test]
    fn extents_file_without_a_tree_short_of_its_size() {
        let r = fork_map(
            geometry(),
            EXTENTS_FILE_ID,
            ForkType::Data,
            3 * 512,
            &rec(&[(1, 2)]),
            None,
        );
        let c = corrupt(r);
        assert_eq!(c.location, Location::Object(u64::from(EXTENTS_FILE_ID)));
        assert!(matches!(c.kind, CorruptKind::Other(s) if s.contains("record missing")));
        // Covered by its three extents: fine.
        let m = fork_map(
            geometry(),
            EXTENTS_FILE_ID,
            ForkType::Data,
            3 * 512,
            &rec(&[(1, 2), (4, 1)]),
            None,
        )
        .unwrap();
        assert_content(&source(m), &[1, 2, 4]);
    }

    #[test]
    fn overflow_record_limit() {
        let a = orec(DATA, FILE, 3, &[(30, 2)]);
        let b = orec(DATA, FILE, 5, &[(40, 2)]);
        // The header claims one record; two are needed.
        let t = one_leaf_tree(&[a.clone(), b.clone()], 1);
        let c = corrupt(map_data(7 * 512, &[(10, 3)], Some(&t)));
        assert_eq!(c.structure, "extents overflow B-tree");
        assert!(matches!(
            c.kind,
            CorruptKind::LimitExceeded {
                what: "extents overflow records",
                limit: 1
            }
        ));
        // A header far above MAX_EXTENT_RECORDS is capped, and enough.
        let t = one_leaf_tree(&[a, b], u32::MAX);
        let m = map_data(7 * 512, &[(10, 3)], Some(&t)).unwrap();
        assert_content(&source(m), &[10, 11, 12, 30, 31, 40, 41]);
    }

    #[test]
    fn overflow_record_data_too_short() {
        let mut r = key(DATA, FILE, 3).encode().to_vec();
        r.extend_from_slice(&[0, 30, 0, 2, 0, 0, 0, 0, 0, 0, 0]);
        let t = one_leaf_tree(&[r], 1);
        let c = corrupt(map_data(5 * 512, &[(10, 3)], Some(&t)));
        assert_eq!(c.structure, "extents overflow B-tree");
        assert!(matches!(
            c.kind,
            CorruptKind::Truncated {
                needed: 12,
                available: 11
            }
        ));
    }

    #[test]
    fn overflow_key_too_short() {
        // keyLength 6, one pad byte, then the data at offset 8.
        let mut r = vec![6, DATA, 0, 0, 0, 20, 0, 0];
        r.extend_from_slice(&[0, 30, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]);
        let t = one_leaf_tree(&[r], 1);
        let c = corrupt(map_data(5 * 512, &[(10, 3)], Some(&t)));
        assert!(matches!(
            c.kind,
            CorruptKind::Truncated {
                needed: 7,
                available: 6
            }
        ));
    }

    #[test]
    fn zero_block_size_is_an_error() {
        let g = Geometry {
            block_size: 0,
            total_blocks: 64,
        };
        let c = corrupt(fork_map(
            g,
            FILE,
            ForkType::Data,
            512,
            &rec(&[(1, 1)]),
            None,
        ));
        assert!(matches!(
            c.kind,
            CorruptKind::InvalidValue {
                field: "allocation block size",
                ..
            }
        ));
    }
}
