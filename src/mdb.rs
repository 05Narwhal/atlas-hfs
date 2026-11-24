// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! The classic HFS master directory block (`HFSMasterDirectoryBlock`), its
//! alternate copy, the allocation geometry and the block area.
//!
//! Every field of the 162-byte block is held in [`Mdb`], so a writer can
//! rewrite a block it parsed: [`Mdb::encode_into`] is the inverse of
//! [`Mdb::parse`]. Reading goes through [`read_mdb`], which falls back to the
//! alternate copy near the end of the volume. [`block_area`] cuts the slice
//! in which allocation block `n` starts at byte `n * block_size`.

use std::sync::Arc;

use super::extents::{HfsExtent, HfsExtentRecord, parse_extent_record, write_extent_record};
use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};
use atlas_fs_core::io::block::{BlockSource, Slice};
use atlas_hfsplus::bytes::{array, be_u16, be_u32, u8_at};
use atlas_hfsplus::extents::Geometry;

/// Byte offset of the master directory block from the start of the volume.
pub const MDB_OFFSET: u64 = 1024;
/// Size of the master directory block in bytes.
pub const MDB_LEN: usize = 162;
/// Bytes per sector; `drVBMSt` and `drAlBlSt` count in them.
pub const SECTOR: u64 = 512;
/// `BD`, the signature of a classic HFS master directory block.
pub const SIGNATURE: u16 = 0x4244;
/// Longest volume name in bytes.
pub const MAX_NAME_LEN: usize = 27;
/// Largest allocation block size accepted.
pub const MAX_BLOCK_SIZE: u32 = 1 << 31;

/// Bytes read for one copy of the block: a whole sector.
const SECTOR_LEN: usize = 512;
/// The name of the structure in errors.
const STRUCTURE: &str = "HFS master directory block";

/// Byte offsets of the fields, from the start of the block.
pub mod offset {
    /// `drSigWord`, u16.
    pub const SIG_WORD: usize = 0;
    /// `drCrDate`, u32.
    pub const CR_DATE: usize = 2;
    /// `drLsMod`, u32.
    pub const LS_MOD: usize = 6;
    /// `drAtrb`, u16.
    pub const ATRB: usize = 10;
    /// `drNmFls`, u16.
    pub const NM_FLS: usize = 12;
    /// `drVBMSt`, u16.
    pub const VBM_ST: usize = 14;
    /// `drAllocPtr`, u16.
    pub const ALLOC_PTR: usize = 16;
    /// `drNmAlBlks`, u16.
    pub const NM_AL_BLKS: usize = 18;
    /// `drAlBlkSiz`, u32.
    pub const AL_BLK_SIZ: usize = 20;
    /// `drClpSiz`, u32.
    pub const CLP_SIZ: usize = 24;
    /// `drAlBlSt`, u16.
    pub const AL_BL_ST: usize = 28;
    /// `drNxtCNID`, u32.
    pub const NXT_CNID: usize = 30;
    /// `drFreeBks`, u16.
    pub const FREE_BKS: usize = 34;
    /// `drVN`, a length byte then 27 bytes.
    pub const VN: usize = 36;
    /// `drVolBkUp`, u32.
    pub const VOL_BK_UP: usize = 64;
    /// `drVSeqNum`, u16.
    pub const VSEQ_NUM: usize = 68;
    /// `drWrCnt`, u32.
    pub const WR_CNT: usize = 70;
    /// `drXTClpSiz`, u32.
    pub const XT_CLP_SIZ: usize = 74;
    /// `drCTClpSiz`, u32.
    pub const CT_CLP_SIZ: usize = 78;
    /// `drNmRtDirs`, u16.
    pub const NM_RT_DIRS: usize = 82;
    /// `drFilCnt`, u32.
    pub const FIL_CNT: usize = 84;
    /// `drDirCnt`, u32.
    pub const DIR_CNT: usize = 88;
    /// `drFndrInfo`, 32 bytes.
    pub const FNDR_INFO: usize = 92;
    /// `drEmbedSigWord`, u16.
    pub const EMBED_SIG_WORD: usize = 124;
    /// `drEmbedExtent`, start block u16 then block count u16.
    pub const EMBED_EXTENT: usize = 126;
    /// `drXTFlSize`, u32.
    pub const XT_FL_SIZE: usize = 130;
    /// `drXTExtRec`, three extents.
    pub const XT_EXT_REC: usize = 134;
    /// `drCTFlSize`, u32.
    pub const CT_FL_SIZE: usize = 146;
    /// `drCTExtRec`, three extents.
    pub const CT_EXT_REC: usize = 150;
}

/// The master directory block (`HFSMasterDirectoryBlock`): every field, as
/// stored. Nothing is validated by [`Mdb::parse`]; see [`Mdb::validate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mdb {
    /// `drSigWord`: `0x4244` for a classic HFS volume.
    pub signature: u16,
    /// `drCrDate`: volume created, seconds since 1904, local time.
    pub create_date: u32,
    /// `drLsMod`: last modified.
    pub modify_date: u32,
    /// `drAtrb`: volume attributes (bit 8 unmounted cleanly, bit 15 locked).
    pub attributes: u16,
    /// `drNmFls`: files in the root folder.
    pub root_file_count: u16,
    /// `drVBMSt`: first sector of the volume bitmap.
    pub bitmap_start: u16,
    /// `drAllocPtr`: where the next allocation search starts.
    pub alloc_ptr: u16,
    /// `drNmAlBlks`: allocation blocks on the volume.
    pub total_blocks: u16,
    /// `drAlBlkSiz`: allocation block size in bytes.
    pub block_size: u32,
    /// `drClpSiz`: default clump size in bytes.
    pub clump_size: u32,
    /// `drAlBlSt`: first sector of allocation block 0.
    pub alloc_start_sector: u16,
    /// `drNxtCNID`: next unused catalog node id.
    pub next_cnid: u32,
    /// `drFreeBks`: free allocation blocks.
    pub free_blocks: u16,
    /// Length byte of `drVN`, as stored (valid names are 1 to 27).
    pub name_len: u8,
    /// The 27 name bytes of `drVN`, as stored, including bytes past the length.
    pub name: [u8; MAX_NAME_LEN],
    /// `drVolBkUp`: last backup date.
    pub backup_date: u32,
    /// `drVSeqNum`: volume backup sequence number.
    pub backup_seq: u16,
    /// `drWrCnt`: volume write count.
    pub write_count: u32,
    /// `drXTClpSiz`: extents overflow file clump size.
    pub xt_clump_size: u32,
    /// `drCTClpSiz`: catalog file clump size.
    pub ct_clump_size: u32,
    /// `drNmRtDirs`: folders in the root folder.
    pub root_dir_count: u16,
    /// `drFilCnt`: files on the volume.
    pub file_count: u32,
    /// `drDirCnt`: folders on the volume.
    pub dir_count: u32,
    /// `drFndrInfo`: Finder information.
    pub finder_info: [u8; 32],
    /// `drEmbedSigWord`: `0x482B` when an HFS+ volume is embedded.
    pub embed_sig: u16,
    /// `drEmbedExtent`: where the embedded volume is, in allocation blocks.
    pub embed_extent: HfsExtent,
    /// `drXTFlSize`: extents overflow file size in bytes.
    pub xt_file_size: u32,
    /// `drXTExtRec`: the extents overflow file's first three extents.
    pub xt_extents: HfsExtentRecord,
    /// `drCTFlSize`: catalog file size in bytes.
    pub ct_file_size: u32,
    /// `drCTExtRec`: the catalog file's first three extents.
    pub ct_extents: HfsExtentRecord,
}

/// Writes `bytes` at `at` in `out`, or `None` when they do not fit.
fn put(out: &mut [u8], at: usize, bytes: &[u8]) -> Option<()> {
    out.get_mut(at..at.checked_add(bytes.len())?)?
        .copy_from_slice(bytes);
    Some(())
}

fn put_u16(out: &mut [u8], at: usize, value: u16) -> Option<()> {
    put(out, at, &value.to_be_bytes())
}

fn put_u32(out: &mut [u8], at: usize, value: u32) -> Option<()> {
    put(out, at, &value.to_be_bytes())
}

impl Mdb {
    /// Reads the fields from the first [`MDB_LEN`] bytes of `b`. `None` when
    /// `b` is shorter. No value is checked.
    pub fn parse(b: &[u8]) -> Option<Self> {
        use offset as o;
        if b.len() < MDB_LEN {
            return None;
        }
        Some(Self {
            signature: be_u16(b, o::SIG_WORD)?,
            create_date: be_u32(b, o::CR_DATE)?,
            modify_date: be_u32(b, o::LS_MOD)?,
            attributes: be_u16(b, o::ATRB)?,
            root_file_count: be_u16(b, o::NM_FLS)?,
            bitmap_start: be_u16(b, o::VBM_ST)?,
            alloc_ptr: be_u16(b, o::ALLOC_PTR)?,
            total_blocks: be_u16(b, o::NM_AL_BLKS)?,
            block_size: be_u32(b, o::AL_BLK_SIZ)?,
            clump_size: be_u32(b, o::CLP_SIZ)?,
            alloc_start_sector: be_u16(b, o::AL_BL_ST)?,
            next_cnid: be_u32(b, o::NXT_CNID)?,
            free_blocks: be_u16(b, o::FREE_BKS)?,
            name_len: u8_at(b, o::VN)?,
            name: array(b, o::VN + 1)?,
            backup_date: be_u32(b, o::VOL_BK_UP)?,
            backup_seq: be_u16(b, o::VSEQ_NUM)?,
            write_count: be_u32(b, o::WR_CNT)?,
            xt_clump_size: be_u32(b, o::XT_CLP_SIZ)?,
            ct_clump_size: be_u32(b, o::CT_CLP_SIZ)?,
            root_dir_count: be_u16(b, o::NM_RT_DIRS)?,
            file_count: be_u32(b, o::FIL_CNT)?,
            dir_count: be_u32(b, o::DIR_CNT)?,
            finder_info: array(b, o::FNDR_INFO)?,
            embed_sig: be_u16(b, o::EMBED_SIG_WORD)?,
            embed_extent: HfsExtent {
                start_block: be_u16(b, o::EMBED_EXTENT)?,
                block_count: be_u16(b, o::EMBED_EXTENT + 2)?,
            },
            xt_file_size: be_u32(b, o::XT_FL_SIZE)?,
            xt_extents: parse_extent_record(b.get(o::XT_EXT_REC..)?)?,
            ct_file_size: be_u32(b, o::CT_FL_SIZE)?,
            ct_extents: parse_extent_record(b.get(o::CT_EXT_REC..)?)?,
        })
    }

    /// Writes every field at its offset in the first [`MDB_LEN`] bytes of
    /// `out`, the inverse of [`Mdb::parse`]. Bytes of `out` past the block
    /// are untouched. `None` (and `out` unchanged) when `out` is shorter.
    pub fn encode_into(&self, out: &mut [u8]) -> Option<()> {
        use offset as o;
        let out = out.get_mut(..MDB_LEN)?;
        put_u16(out, o::SIG_WORD, self.signature)?;
        put_u32(out, o::CR_DATE, self.create_date)?;
        put_u32(out, o::LS_MOD, self.modify_date)?;
        put_u16(out, o::ATRB, self.attributes)?;
        put_u16(out, o::NM_FLS, self.root_file_count)?;
        put_u16(out, o::VBM_ST, self.bitmap_start)?;
        put_u16(out, o::ALLOC_PTR, self.alloc_ptr)?;
        put_u16(out, o::NM_AL_BLKS, self.total_blocks)?;
        put_u32(out, o::AL_BLK_SIZ, self.block_size)?;
        put_u32(out, o::CLP_SIZ, self.clump_size)?;
        put_u16(out, o::AL_BL_ST, self.alloc_start_sector)?;
        put_u32(out, o::NXT_CNID, self.next_cnid)?;
        put_u16(out, o::FREE_BKS, self.free_blocks)?;
        put(out, o::VN, &[self.name_len])?;
        put(out, o::VN + 1, &self.name)?;
        put_u32(out, o::VOL_BK_UP, self.backup_date)?;
        put_u16(out, o::VSEQ_NUM, self.backup_seq)?;
        put_u32(out, o::WR_CNT, self.write_count)?;
        put_u32(out, o::XT_CLP_SIZ, self.xt_clump_size)?;
        put_u32(out, o::CT_CLP_SIZ, self.ct_clump_size)?;
        put_u16(out, o::NM_RT_DIRS, self.root_dir_count)?;
        put_u32(out, o::FIL_CNT, self.file_count)?;
        put_u32(out, o::DIR_CNT, self.dir_count)?;
        put(out, o::FNDR_INFO, &self.finder_info)?;
        put_u16(out, o::EMBED_SIG_WORD, self.embed_sig)?;
        put_u16(out, o::EMBED_EXTENT, self.embed_extent.start_block)?;
        put_u16(out, o::EMBED_EXTENT + 2, self.embed_extent.block_count)?;
        put_u32(out, o::XT_FL_SIZE, self.xt_file_size)?;
        write_extent_record(&self.xt_extents, out.get_mut(o::XT_EXT_REC..)?)?;
        put_u32(out, o::CT_FL_SIZE, self.ct_file_size)?;
        write_extent_record(&self.ct_extents, out.get_mut(o::CT_EXT_REC..)?)?;
        Some(())
    }

    /// The checks a mount makes before trusting the geometry: block size
    /// non-zero, a multiple of 512 and at most 2^31; at least one allocation
    /// block; the first allocation block inside a source of `source_len`
    /// bytes; a volume name of 1 to 27 bytes; a non-empty first catalog
    /// extent. The signature is checked by [`read_mdb`].
    ///
    /// `offset` is where the block was read, for the error's location.
    pub fn validate(&self, offset: u64, source_len: u64) -> Result<()> {
        let bad = |kind| Error::Corrupt(Corruption::new(STRUCTURE, Location::Byte(offset), kind));
        let size = self.block_size;
        if size == 0 || !u64::from(size).is_multiple_of(SECTOR) {
            return Err(bad(CorruptKind::InvalidValue {
                field: "allocation block size",
                value: u64::from(size),
            }));
        }
        if size > MAX_BLOCK_SIZE {
            return Err(bad(CorruptKind::OutOfRange {
                field: "allocation block size",
                value: u64::from(size),
                limit_name: "maximum allocation block size",
                limit: u64::from(MAX_BLOCK_SIZE),
            }));
        }
        if self.total_blocks == 0 {
            return Err(bad(CorruptKind::InvalidValue {
                field: "allocation block count",
                value: 0,
            }));
        }
        let alloc_start = u64::from(self.alloc_start_sector)
            .checked_mul(SECTOR)
            .ok_or_else(|| {
                bad(CorruptKind::Overflow {
                    field: "first allocation block",
                })
            })?;
        let alloc_len = u64::from(self.total_blocks)
            .checked_mul(u64::from(size))
            .ok_or_else(|| {
                bad(CorruptKind::Overflow {
                    field: "allocation area length",
                })
            })?;
        alloc_start.checked_add(alloc_len).ok_or_else(|| {
            bad(CorruptKind::Overflow {
                field: "allocation area end",
            })
        })?;
        if alloc_start >= source_len {
            return Err(bad(CorruptKind::OutOfRange {
                field: "first allocation block",
                value: alloc_start,
                limit_name: "source size",
                limit: source_len,
            }));
        }
        match usize::from(self.name_len) {
            0 => {
                return Err(bad(CorruptKind::InvalidValue {
                    field: "volume name length",
                    value: 0,
                }));
            }
            n if n > MAX_NAME_LEN => {
                return Err(bad(CorruptKind::OutOfRange {
                    field: "volume name length",
                    value: u64::from(self.name_len),
                    limit_name: "maximum name length",
                    limit: MAX_NAME_LEN as u64,
                }));
            }
            _ => {}
        }
        if self.ct_extents[0].block_count == 0 {
            return Err(bad(CorruptKind::InvalidValue {
                field: "catalog first extent block count",
                value: 0,
            }));
        }
        Ok(())
    }

    /// Byte offset of allocation block 0 in the volume (`drAlBlSt` sectors).
    pub fn alloc_start(&self) -> u64 {
        u64::from(self.alloc_start_sector).saturating_mul(SECTOR)
    }

    /// The allocation geometry, as the HFS+ code uses it.
    pub fn geometry(&self) -> Geometry {
        Geometry {
            block_size: self.block_size,
            total_blocks: u32::from(self.total_blocks),
        }
    }

    /// The volume name bytes (Mac OS Roman), at most [`MAX_NAME_LEN`] of
    /// them, however large the stored length is.
    pub fn name_bytes(&self) -> &[u8] {
        let len = usize::from(self.name_len).min(MAX_NAME_LEN);
        self.name.get(..len).unwrap_or(&self.name)
    }

    /// Byte offset of the volume bitmap in the volume (`drVBMSt` sectors).
    pub fn bitmap_offset(&self) -> u64 {
        u64::from(self.bitmap_start).saturating_mul(SECTOR)
    }

    /// Bytes in the volume bitmap: one bit per allocation block, rounded up.
    pub fn bitmap_len(&self) -> u64 {
        u64::from(self.total_blocks).div_ceil(8)
    }

    /// Free space in bytes: `drFreeBks` allocation blocks.
    pub fn free_bytes(&self) -> u64 {
        u64::from(self.free_blocks).saturating_mul(u64::from(self.block_size))
    }
}

/// The block area of the volume: allocation block `n` is at byte
/// `n * block_size` in the returned slice.
///
/// The slice starts at [`Mdb::alloc_start`] and is `drNmAlBlks` blocks long,
/// clamped to what `source` holds, so a truncated image still lists (reads of
/// the missing blocks fail on their own). Fails when the start is not inside
/// `source`.
pub fn block_area(source: Arc<dyn BlockSource>, mdb: &Mdb) -> Result<Slice> {
    let start = mdb.alloc_start();
    let size = source.len();
    let corrupt = |kind| Error::Corrupt(Corruption::new(STRUCTURE, Location::Unspecified, kind));
    let available = size.checked_sub(start).filter(|&n| n > 0).ok_or_else(|| {
        corrupt(CorruptKind::OutOfRange {
            field: "first allocation block",
            value: start,
            limit_name: "source size",
            limit: size,
        })
    })?;
    let len = u64::from(mdb.total_blocks)
        .checked_mul(u64::from(mdb.block_size))
        .ok_or_else(|| {
            corrupt(CorruptKind::Overflow {
                field: "allocation area length",
            })
        })?;
    Slice::new(source, start, len.min(available))
}

/// Which copy of the master directory block was used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MdbCopy {
    /// The block at byte 1024.
    Primary,
    /// The block in the second-to-last sector of the volume.
    Alternate,
}

/// A master directory block and where it was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Located {
    /// The block, validated.
    pub mdb: Mdb,
    /// Byte offset it was read from.
    pub offset: u64,
    /// Which copy that is.
    pub copy: MdbCopy,
}

/// Reads, checks and validates the block at `offset`.
fn read_copy(source: &dyn BlockSource, offset: u64) -> Result<Mdb> {
    let size = source.len();
    let corrupt = |kind| Error::Corrupt(Corruption::new(STRUCTURE, Location::Byte(offset), kind));
    let truncated = || {
        corrupt(CorruptKind::Truncated {
            needed: offset.saturating_add(SECTOR),
            available: size,
        })
    };
    if offset.checked_add(SECTOR).is_none_or(|end| end > size) {
        return Err(truncated());
    }
    let mut sector = [0u8; SECTOR_LEN];
    source.read_at(offset, &mut sector)?;
    let mdb = Mdb::parse(&sector).ok_or_else(truncated)?;
    if mdb.signature != SIGNATURE {
        return Err(corrupt(CorruptKind::BadSignature {
            expected: u64::from(SIGNATURE),
            found: u64::from(mdb.signature),
        }));
    }
    mdb.validate(offset, size)?;
    Ok(mdb)
}

/// Where the alternate block of a source of `len` bytes is: the
/// second-to-last whole sector, if that is past the primary.
fn alternate_offset(len: u64) -> Option<u64> {
    (len / SECTOR)
        .checked_sub(2)?
        .checked_mul(SECTOR)
        .filter(|&offset| offset > MDB_OFFSET)
}

/// Reads the master directory block of `source`, which is the volume.
///
/// The block at byte 1024 is tried first. When its signature is not `BD` or
/// it fails [`Mdb::validate`], the alternate copy in the second-to-last
/// sector is tried and used if it is good. When both fail the primary's
/// error is returned. A read error from the device is returned as is.
pub fn read_mdb(source: &dyn BlockSource) -> Result<Located> {
    let primary = match read_copy(source, MDB_OFFSET) {
        Ok(mdb) => {
            return Ok(Located {
                mdb,
                offset: MDB_OFFSET,
                copy: MdbCopy::Primary,
            });
        }
        Err(e @ (Error::Corrupt(_) | Error::OutOfBounds { .. })) => e,
        Err(e) => return Err(e),
    };
    if let Some(offset) = alternate_offset(source.len())
        && let Ok(mdb) = read_copy(source, offset)
    {
        return Ok(Located {
            mdb,
            offset,
            copy: MdbCopy::Alternate,
        });
    }
    Err(primary)
}

/// A valid volume built by hand, for the tests here and the fuzz seeds.
#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;

    /// Allocation block size of the fixture volume.
    pub(crate) const BLOCK_SIZE: u32 = 512;
    /// Allocation blocks of the fixture volume.
    pub(crate) const TOTAL_BLOCKS: u16 = 8;
    /// Sectors in the fixture volume: boot blocks, MDB, bitmap, 8 blocks,
    /// the alternate MDB and the last sector.
    pub(crate) const SECTORS: usize = 14;

    /// A valid block for the fixture volume.
    pub(crate) fn mdb() -> Mdb {
        let mut name = [0u8; MAX_NAME_LEN];
        name[..5].copy_from_slice(b"Atlas");
        let mut finder_info = [0u8; 32];
        finder_info[0] = 0x11;
        Mdb {
            signature: SIGNATURE,
            create_date: 3_000_000_000,
            modify_date: 3_100_000_000,
            attributes: 0x0100,
            root_file_count: 2,
            bitmap_start: 3,
            alloc_ptr: 0,
            total_blocks: TOTAL_BLOCKS,
            block_size: BLOCK_SIZE,
            clump_size: 2048,
            alloc_start_sector: 4,
            next_cnid: 20,
            free_blocks: 5,
            name_len: 5,
            name,
            backup_date: 0,
            backup_seq: 0,
            write_count: 7,
            xt_clump_size: 1024,
            ct_clump_size: 1024,
            root_dir_count: 1,
            file_count: 2,
            dir_count: 1,
            finder_info,
            embed_sig: 0,
            embed_extent: HfsExtent::default(),
            xt_file_size: 512,
            xt_extents: [
                HfsExtent {
                    start_block: 0,
                    block_count: 1,
                },
                HfsExtent::default(),
                HfsExtent::default(),
            ],
            ct_file_size: 1024,
            ct_extents: [
                HfsExtent {
                    start_block: 1,
                    block_count: 2,
                },
                HfsExtent::default(),
                HfsExtent::default(),
            ],
        }
    }

    /// A volume image with `mdb` at both the primary and the alternate
    /// position and a recognisable byte in every allocation block.
    pub(crate) fn volume_with(mdb: &Mdb) -> Vec<u8> {
        let mut image = vec![0u8; SECTORS * 512];
        // The bitmap: blocks 0 to 2 used.
        image[3 * 512] = 0b1110_0000;
        for block in 0..usize::from(TOTAL_BLOCKS) {
            image[(4 + block) * 512] = 0xA0 + block as u8;
        }
        for at in [MDB_OFFSET as usize, (SECTORS - 2) * 512] {
            mdb.encode_into(&mut image[at..]).unwrap();
        }
        image
    }

    /// The valid fixture volume.
    pub(crate) fn volume() -> Vec<u8> {
        volume_with(&mdb())
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::{self, SECTORS};
    use super::*;
    use atlas_fs_core::io::block::MemSource;

    fn mem(bytes: Vec<u8>) -> Arc<MemSource> {
        Arc::new(MemSource::new(bytes))
    }

    fn corruption<T>(r: Result<T>) -> Corruption {
        match r {
            Err(Error::Corrupt(c)) => c,
            Err(other) => panic!("expected Corrupt, got {other:?}"),
            Ok(_) => panic!("expected an error"),
        }
    }

    /// The fixture volume with `edit` applied to its MDB, at both positions.
    fn edited(edit: impl FnOnce(&mut Mdb)) -> Vec<u8> {
        let mut mdb = fixtures::mdb();
        edit(&mut mdb);
        fixtures::volume_with(&mdb)
    }

    fn invalid(field: &'static str, value: u64) -> CorruptKind {
        CorruptKind::InvalidValue { field, value }
    }

    #[test]
    fn offsets_are_consistent_with_the_layout() {
        assert_eq!(offset::CT_EXT_REC + 12, MDB_LEN);
        assert_eq!(offset::VN + 1 + MAX_NAME_LEN, offset::VOL_BK_UP);
        assert_eq!(offset::FNDR_INFO + 32, offset::EMBED_SIG_WORD);
    }

    #[test]
    fn reads_a_valid_primary() {
        let image = fixtures::volume();
        let located = read_mdb(&MemSource::new(image)).unwrap();
        assert_eq!(located.copy, MdbCopy::Primary);
        assert_eq!(located.offset, 1024);
        assert_eq!(located.mdb, fixtures::mdb());
        let mdb = &located.mdb;
        assert_eq!(mdb.name_bytes(), b"Atlas");
        assert_eq!(mdb.alloc_start(), 2048);
        assert_eq!(
            mdb.geometry(),
            Geometry {
                block_size: 512,
                total_blocks: 8
            }
        );
        assert_eq!(mdb.bitmap_offset(), 1536);
        assert_eq!(mdb.bitmap_len(), 1);
        assert_eq!(mdb.free_bytes(), 5 * 512);
    }

    #[test]
    fn bitmap_len_rounds_up() {
        let mut mdb = fixtures::mdb();
        for (blocks, bytes) in [(1u16, 1u64), (8, 1), (9, 2), (u16::MAX, 8192)] {
            mdb.total_blocks = blocks;
            assert_eq!(mdb.bitmap_len(), bytes, "{blocks}");
        }
    }

    #[test]
    fn parse_encode_round_trip_with_garbage_in_every_field() {
        let mut sector = [0u8; 512];
        for (i, byte) in sector.iter_mut().enumerate() {
            *byte = if i < MDB_LEN {
                (i as u8).wrapping_mul(7).wrapping_add(3) | 1
            } else {
                0xEE
            };
        }
        let mdb = Mdb::parse(&sector).unwrap();
        let mut copy = sector;
        mdb.encode_into(&mut copy).unwrap();
        assert_eq!(copy, sector);
        // Into a different buffer the block's bytes come out the same and
        // nothing past it is written.
        let mut fresh = [0x55u8; 512];
        mdb.encode_into(&mut fresh).unwrap();
        assert_eq!(fresh[..MDB_LEN], sector[..MDB_LEN]);
        assert!(fresh[MDB_LEN..].iter().all(|&b| b == 0x55));
        // The garbage went into the right fields.
        assert_eq!(mdb.signature, u16::from_be_bytes([sector[0], sector[1]]));
        assert_eq!(mdb.name_len, sector[36]);
        assert_eq!(mdb.name[26], sector[63]);
        assert_eq!(
            mdb.ct_extents[2].block_count,
            u16::from_be_bytes([sector[160], sector[161]])
        );
    }

    #[test]
    fn short_buffers() {
        let sector = [0u8; 512];
        assert!(Mdb::parse(&sector[..MDB_LEN - 1]).is_none());
        assert!(Mdb::parse(&sector[..MDB_LEN]).is_some());
        assert!(Mdb::parse(&[]).is_none());
        let mdb = fixtures::mdb();
        let mut out = [0x77u8; MDB_LEN - 1];
        assert!(mdb.encode_into(&mut out).is_none());
        assert!(out.iter().all(|&b| b == 0x77));
    }

    #[test]
    fn validation_failures() {
        let cases: Vec<(Vec<u8>, CorruptKind)> = vec![
            (
                edited(|m| m.block_size = 0),
                invalid("allocation block size", 0),
            ),
            (
                edited(|m| m.block_size = 1000),
                invalid("allocation block size", 1000),
            ),
            (
                edited(|m| m.block_size = (1 << 31) + 512),
                CorruptKind::OutOfRange {
                    field: "allocation block size",
                    value: (1 << 31) + 512,
                    limit_name: "maximum allocation block size",
                    limit: 1 << 31,
                },
            ),
            (
                edited(|m| m.total_blocks = 0),
                invalid("allocation block count", 0),
            ),
            (
                edited(|m| m.alloc_start_sector = 14),
                CorruptKind::OutOfRange {
                    field: "first allocation block",
                    value: 14 * 512,
                    limit_name: "source size",
                    limit: (SECTORS * 512) as u64,
                },
            ),
            (edited(|m| m.name_len = 0), invalid("volume name length", 0)),
            (
                edited(|m| m.name_len = 28),
                CorruptKind::OutOfRange {
                    field: "volume name length",
                    value: 28,
                    limit_name: "maximum name length",
                    limit: 27,
                },
            ),
            (
                edited(|m| m.ct_extents[0].block_count = 0),
                invalid("catalog first extent block count", 0),
            ),
        ];
        for (image, kind) in cases {
            // Both copies carry the edit, so the primary's error comes back.
            let c = corruption(read_mdb(&MemSource::new(image)));
            assert_eq!(c.structure, "HFS master directory block");
            assert_eq!(c.location, Location::Byte(MDB_OFFSET));
            assert_eq!(c.kind, kind);
        }
    }

    #[test]
    fn validate_accepts_the_edges() {
        let mut mdb = fixtures::mdb();
        mdb.block_size = 512;
        mdb.name_len = 27;
        assert!(mdb.validate(1024, 2049).is_ok());
        mdb.block_size = 1 << 31;
        assert!(mdb.validate(1024, 2049).is_ok());
        // alloc_start must be strictly inside the source.
        assert!(mdb.validate(1024, 2048).is_err());
    }

    #[test]
    fn bad_signature_is_reported_as_one() {
        let mut image = fixtures::volume();
        // Damage both copies' signatures.
        image[1024] = b'H';
        image[(SECTORS - 2) * 512] = b'H';
        let c = corruption(read_mdb(&MemSource::new(image)));
        assert_eq!(c.location, Location::Byte(1024));
        assert_eq!(
            c.kind,
            CorruptKind::BadSignature {
                expected: 0x4244,
                found: 0x4844
            }
        );
    }

    #[test]
    fn truncated_sources() {
        for len in [0usize, 1, 1024, 1025, 1535] {
            let image = fixtures::volume()[..len].to_vec();
            let c = corruption(read_mdb(&MemSource::new(image)));
            assert_eq!(c.location, Location::Byte(1024), "{len}");
            assert_eq!(
                c.kind,
                CorruptKind::Truncated {
                    needed: 1536,
                    available: len as u64
                },
                "{len}"
            );
        }
    }

    #[test]
    fn alternate_is_used_when_the_primary_is_damaged() {
        let mut image = fixtures::volume();
        image[1024..1024 + 512].fill(0);
        let located = read_mdb(&MemSource::new(image)).unwrap();
        assert_eq!(located.copy, MdbCopy::Alternate);
        assert_eq!(located.offset, ((SECTORS - 2) * 512) as u64);
        assert_eq!(located.mdb, fixtures::mdb());

        // A primary that parses but fails validation also falls back.
        let mut image = fixtures::volume();
        image[1024 + offset::AL_BLK_SIZ..1024 + offset::AL_BLK_SIZ + 4].fill(0);
        let located = read_mdb(&MemSource::new(image)).unwrap();
        assert_eq!(located.copy, MdbCopy::Alternate);
    }

    #[test]
    fn both_damaged_gives_the_primarys_error() {
        let mut image = fixtures::volume();
        // Primary: block size zero. Alternate: no name.
        image[1024 + offset::AL_BLK_SIZ..1024 + offset::AL_BLK_SIZ + 4].fill(0);
        image[(SECTORS - 2) * 512 + offset::VN] = 0;
        let c = corruption(read_mdb(&MemSource::new(image)));
        assert_eq!(c.location, Location::Byte(1024));
        assert_eq!(c.kind, invalid("allocation block size", 0));
    }

    #[test]
    fn the_alternate_is_not_read_when_it_would_be_the_primary() {
        assert_eq!(alternate_offset(0), None);
        assert_eq!(alternate_offset(1023), None);
        assert_eq!(alternate_offset(1536), None); // sector 1, before the MDB
        assert_eq!(alternate_offset(2048), None); // sector 2 is the primary
        assert_eq!(alternate_offset(2559), None);
        assert_eq!(alternate_offset(2560), Some(1536));
        assert_eq!(alternate_offset(2560 + 511), Some(1536));
        assert_eq!(alternate_offset(u64::MAX), Some((u64::MAX / 512 - 2) * 512));
    }

    #[test]
    fn block_area_maps_block_n_at_n_times_block_size() {
        let image = mem(fixtures::volume());
        let mdb = read_mdb(&*image).unwrap().mdb;
        let area = block_area(image, &mdb).unwrap();
        assert_eq!(area.len(), 8 * 512);
        assert_eq!(area.start(), 2048);
        for block in 0..8u64 {
            let mut byte = [0u8; 1];
            area.read_at(block * 512, &mut byte).unwrap();
            assert_eq!(u64::from(byte[0]), 0xA0 + block);
        }
    }

    #[test]
    fn block_area_is_clamped_on_a_truncated_source() {
        let image = fixtures::volume()[..2048 + 1000].to_vec();
        let source = mem(image);
        let mdb = read_mdb(&*source).unwrap().mdb;
        let area = block_area(source, &mdb).unwrap();
        assert_eq!(area.len(), 1000);
        let mut buf = [0u8; 512];
        assert!(area.read_at(0, &mut buf).is_ok());
        assert!(area.read_at(512, &mut buf).is_err());
    }

    #[test]
    fn block_area_fails_when_the_start_is_past_the_end() {
        let mdb = fixtures::mdb();
        for len in [0usize, 100, 2047, 2048] {
            let source: Arc<dyn BlockSource> = mem(vec![0u8; len]);
            let c = corruption(block_area(source, &mdb));
            assert_eq!(
                c.kind,
                CorruptKind::OutOfRange {
                    field: "first allocation block",
                    value: 2048,
                    limit_name: "source size",
                    limit: len as u64
                }
            );
        }
        let source: Arc<dyn BlockSource> = mem(vec![0u8; 2049]);
        assert_eq!(block_area(source, &mdb).unwrap().len(), 1);
    }

    #[test]
    fn arithmetic_at_the_maxima() {
        let mut mdb = fixtures::mdb();
        mdb.total_blocks = u16::MAX;
        mdb.block_size = 1 << 31;
        mdb.alloc_start_sector = u16::MAX;
        mdb.bitmap_start = u16::MAX;
        mdb.free_blocks = u16::MAX;
        assert_eq!(mdb.alloc_start(), 65_535 * 512);
        assert_eq!(mdb.bitmap_offset(), 65_535 * 512);
        assert_eq!(mdb.bitmap_len(), 8192);
        assert_eq!(mdb.free_bytes(), 65_535 * (1u64 << 31));
        assert_eq!(
            mdb.geometry(),
            Geometry {
                block_size: 1 << 31,
                total_blocks: 65_535
            }
        );
        assert!(mdb.validate(1024, u64::MAX).is_ok());
        // A source that ends before the area starts fails; one that ends
        // inside it is clamped, never overflowed.
        let source: Arc<dyn BlockSource> = mem(vec![0u8; 4096]);
        assert!(block_area(source, &mdb).is_err());
        mdb.alloc_start_sector = 1;
        let source: Arc<dyn BlockSource> = mem(vec![0u8; 4096]);
        assert_eq!(block_area(source, &mdb).unwrap().len(), 4096 - 512);
        // Field maxima survive a round trip.
        let mut sector = [0u8; 512];
        mdb.name_len = u8::MAX;
        mdb.encode_into(&mut sector).unwrap();
        let again = Mdb::parse(&sector).unwrap();
        assert_eq!(again, mdb);
        assert_eq!(again.name_bytes().len(), MAX_NAME_LEN);
    }
}
