# atlas-hfs

A read-only classic HFS (Mac OS Standard) reader. It implements `atlas_fs_core::fs::FileSystem`, so a mounted
volume lists folders, looks up names, reads data and resource forks and scans the whole catalog through the same
trait as the other `atlas-*` readers.

- Master directory block (and its alternate copy), catalog and extents overflow B-trees, data and resource forks.
- Names decoded from the volume's Mac text encoding (Roman by default, `mount_with` picks another).
- Finder type and creator codes, flags and dates.
- No `unsafe`. Every length, offset and count read from the disk is checked; damage is an error, not a panic.

It builds on `atlas-hfsplus` for the B-tree reader and fork mapping.

```rust,no_run
use std::sync::Arc;
use atlas_fs_core::fs::FileSystem;
use atlas_fs_core::io::block::BlockSource;
use atlas_fs_core::io::file::FileSource;

let source: Arc<dyn BlockSource> = Arc::new(FileSource::open("classic.img".as_ref())?);
let volume = atlas_hfs::mount(source)?;
let mut dir = volume.read_dir(volume.root())?;
# Ok::<(), atlas_fs_core::error::Error>(())
```

The `test-util` feature exposes the test volume builder; it is for tests of crates that use this one and is not
part of the stable API.

## Licence

Licensed under either of Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE)) or MIT license
([LICENSE-MIT](LICENSE-MIT)) at your option.
