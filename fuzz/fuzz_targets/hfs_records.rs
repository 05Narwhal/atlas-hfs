// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Fuzzes the classic HFS record parsers without a mount: the master
//! directory block, extents overflow keys and extent records, and catalog
//! keys and records.
//!
//! Every parse that succeeds is encoded again and parsed once more, and the
//! two values must be equal. Errors are fine; a panic or a round trip that
//! changes a value is the bug.

#![no_main]

use atlas_hfs::catalog::{CatalogKey, CatalogRecord, encode_key};
use atlas_hfs::extents::{
    EXTENT_RECORD_LEN, ExtentKey, parse_extent_record, write_extent_record,
};
use atlas_hfs::mdb::{MDB_LEN, Mdb};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    mdb(data);
    extent_key(data);
    extent_record(data);
    catalog_key(data);
    catalog_record(data);
});

fn mdb(data: &[u8]) {
    let Some(parsed) = Mdb::parse(data) else {
        return;
    };
    let _ = parsed.validate(1024, data.len() as u64);
    let mut out = vec![0u8; MDB_LEN];
    assert!(parsed.encode_into(&mut out).is_some());
    assert_eq!(Mdb::parse(&out).as_ref(), Some(&parsed));
}

fn extent_key(data: &[u8]) {
    let Some(parsed) = ExtentKey::parse(data) else {
        return;
    };
    let encoded = parsed.encode();
    assert_eq!(ExtentKey::parse(&encoded[1..]), Some(parsed));
}

fn extent_record(data: &[u8]) {
    let Some(parsed) = parse_extent_record(data) else {
        return;
    };
    let mut out = [0u8; EXTENT_RECORD_LEN];
    assert!(write_extent_record(&parsed, &mut out).is_some());
    assert_eq!(parse_extent_record(&out), Some(parsed));
}

fn catalog_key(data: &[u8]) {
    let Some(parsed) = CatalogKey::parse(data) else {
        return;
    };
    let encoded = encode_key(parsed.parent_id, parsed.name);
    // `encode_key` starts with the key length byte, which `parse` does not.
    assert_eq!(CatalogKey::parse(&encoded[1..]), Some(parsed));
}

fn catalog_record(data: &[u8]) {
    let Ok(parsed) = CatalogRecord::parse(data) else {
        return;
    };
    let encoded = parsed.encode();
    match CatalogRecord::parse(&encoded) {
        Ok(again) => assert_eq!(again, parsed),
        Err(e) => panic!("encoded record does not parse: {e:?}"),
    }
}
