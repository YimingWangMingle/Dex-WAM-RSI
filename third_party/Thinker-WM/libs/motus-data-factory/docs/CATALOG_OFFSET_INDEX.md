# Catalog SQLite offset sidecar

The concise Episode Catalog JSONL remains the immutable canonical artifact.
The optional SQLite sidecar is a physical lookup accelerator. It does not
change `catalog_id`, Episode rows, SamplePool files, or model-facing APIs.

## Files

For:

```text
source.episode_catalog.jsonl
source.episode_catalog.meta.json
```

the default sidecar is:

```text
source.episode_catalog.index.sqlite
```

The sidecar stores only:

- immutable Catalog binding metadata;
- Episode ID;
- row ordinal;
- JSONL byte offset and byte length.

It does not duplicate Episode JSON, raw storage paths, model samples, split, or
decoded tensors.

## Build

```bash
python examples/build_catalog_offset_index.py \
  --catalog /path/source.episode_catalog.jsonl
```

The builder streams and validates every Catalog row, recomputes the complete
SHA256, rejects duplicate IDs, runs SQLite `quick_check`, and atomically
publishes the completed sidecar. Existing output is protected unless
`--force` is explicit.

## Repository

The original in-memory mode remains unchanged:

```python
repository = EpisodeRepository(catalog_path)
```

Indexed mode is explicit:

```python
repository = EpisodeRepository(
    catalog_path,
    index_path=index_path,
    verify_sha256=False,
    record_cache_size=256,
)
```

Indexed mode validates sidecar format, Catalog ID, protocol, published SHA256,
file size, filename, and row count before access. Episode JSON is read lazily
with `pread` and validated against Catalog metadata on every cache miss.

`verify_sha256=True` additionally streams the full Catalog at each Repository
startup. It is the strongest check but does not remove startup I/O. After an
offline full audit and immutable publication, `verify_sha256=False` trusts the
SHA-bound sidecar plus Catalog size and validates each accessed row. This is
the intended low-latency distributed training mode.

The sidecar and canonical Catalog must be published and retained together.
Changing Catalog content, metadata, filename, size, SHA, or row count requires
building a new sidecar. A sidecar is never reused across Catalog versions.
