# Concise split-neutral Episode Catalog and SamplePool v2

V2 keeps one machine-readable format marker per file and one `catalog_id`
binding between files. Episode rows and model-owned selections contain only
semantic data. Train/test and sampling policy are not Catalog fields.

## Catalog files

One immutable publication contains:

```text
agibot_beta.episode_catalog.jsonl
agibot_beta.episode_catalog.meta.json
```

The metadata file is concise:

```json
{
  "format": "motus_episode_catalog_v2",
  "catalog_id": "agibot_beta_20260723",
  "source": "agibot_beta",
  "child": "agibot_beta",
  "episodes": 168000,
  "sha256": "..."
}
```

`format` selects the parser. `catalog_id` is the only SamplePool binding.
`episodes` and `sha256` detect truncation or modification. Optional `meta`
stores provenance and builder diagnostics that normal consumers may ignore.
Changing raw data, parsing rules or Catalog contents requires a new immutable
`catalog_id`.

Each JSONL row describes one complete accessible Episode:

```json
{
  "episode_id": "agibot_beta:327:648642",
  "robot": "agibot-a2-d",
  "video": {
    "count": 1295,
    "storage": {
      "type": "tar",
      "path": "/data/agibot_beta/data.tar",
      "member": "648642/videos/head_color.mp4",
      "offset": 123456,
      "size": 987654
    }
  },
  "state": {
    "count": 1295,
    "storage": {
      "type": "tar",
      "path": "/data/agibot_beta/data.tar",
      "member": "327/648642/proprio_stats.h5",
      "offset": 234567,
      "size": 345678
    }
  },
  "action": {
    "count": 1295,
    "storage": {
      "type": "tar",
      "path": "/data/agibot_beta/data.tar",
      "member": "327/648642/proprio_stats.h5",
      "offset": 234567,
      "size": 345678
    }
  },
  "texts": [
    {
      "index": 0,
      "kind": "episode_context",
      "text": "Scene and task."
    },
    {
      "index": 1,
      "kind": "action_segment",
      "text": "Move towards the object.",
      "video_range": [100, 260]
    }
  ],
  "meta": {
    "task_id": 327,
    "raw_episode_id": 648642
  }
}
```

`meta` fields at Catalog, modality, text-item and Episode levels are optional
source provenance. Model-side code does not need them to index samples.

Catalog rows do not repeat format, source or child. They contain no split,
window, source weight, task blacklist or decoded tensor.

## SamplePool

A SamplePool is one JSONL file. Its first non-empty row is:

```json
{
  "format": "motus_sample_pool_v2",
  "pool_id": "agibot_beta_train_20260723",
  "catalog_id": "agibot_beta_20260723",
  "samples": 1000000
}
```

`samples` and optional `meta` are convenience and provenance fields. The only
Catalog compatibility check is exact `catalog_id` equality.

Every remaining row is one model-owned sample:

```json
{
  "episode_id": "agibot_beta:327:648642",
  "video_frame_indices": [100, 104, 108, 112],
  "state_indices": [100],
  "action_indices": [101, 102, 103],
  "text_indices": [0, 1]
}
```

The four index lists are independent. The data factory validates every bound,
preserves order and repeats, and never derives, sorts, pads, truncates or
repairs model-owned indices. Video requires at least one index; state, action
and text may be empty.

Different train, test and demonstration Pool files express membership. The
Catalog itself remains split-neutral.

## Runtime boundary

`EpisodeRepository` converts concise v2 rows into its internal source-reader
shape. Model-facing Episode and batch APIs remain unchanged. State/action
finite checks apply to every source; Agibot Beta also applies its threshold-100
magnitude guard on exact Pool selections before MP4 extraction and decode.

## Compatibility

Historical complete v1 Catalog/SamplePool pairs remain readable. New Catalogs
use concise v2. V1 and v2 files cannot be cross-bound.
