# Episode Catalog and Full-Modality SamplePool Contract v1

This is the immutable historical split-bearing protocol used by accepted smoke
and parity artifacts. New full Catalog publications use the split-neutral v2
protocol in `EPISODE_CATALOG_SAMPLE_POOL_V2.md`.

This contract separates source access from model-owned sample alignment. The
model side supplies the final indices for every modality. The data factory does
not derive state, action or text indices from video frame indices.

## Files

The first Catalog implementation uses:

```text
agibot_beta.episode_catalog.jsonl
agibot_beta.episode_catalog.meta.json
```

SamplePool is one self-contained JSONL. Its first line is a header and all
remaining lines are sample selections:

```text
agibot_beta.sample_pool.jsonl
```

There is no separate SamplePool metadata file.

## Catalog Metadata

```json
{
  "schema_version": "motus_episode_catalog_metadata_v1",
  "record_schema_version": "motus_episode_record_v1",
  "catalog_id": "agibot_beta_20260722_seed20260720_v1",
  "source_dataset": "agibot_beta",
  "source_child": "agibot_beta",
  "data_version": "agibot_beta_source_20260611",
  "split_policy": {
    "type": "stable_ranked_episode_hash",
    "seed": 20260720,
    "test_fraction": 0.01
  }
}
```

## Episode Record

Each Catalog row exposes independent index spaces and access information for
video, state, action and text:

```json
{
  "schema_version": "motus_episode_record_v1",
  "episode_id": "agibot_beta:327:648642",
  "source_dataset": "agibot_beta",
  "source_child": "agibot_beta",
  "robot_name": "agibot-a2-d",
  "split": "train",
  "video": {
    "frame_count": 1500,
    "locator": {
      "storage_backend": "tar_member",
      "uri": "/data/agibot_beta/video.tar",
      "member": "327/648642/video.mp4",
      "offset_data": 1024,
      "size": 12345678
    }
  },
  "state": {
    "count": 1500,
    "locator": {
      "storage_backend": "tar_member",
      "uri": "/data/agibot_beta/proprio.tar",
      "member": "327/648642/proprio_stats.h5",
      "offset_data": 2048,
      "size": 345678
    }
  },
  "action": {
    "count": 1500,
    "locator": {
      "storage_backend": "tar_member",
      "uri": "/data/agibot_beta/proprio.tar",
      "member": "327/648642/proprio_stats.h5",
      "offset_data": 2048,
      "size": 345678
    }
  },
  "text": {
    "items": [
      {"index": 0, "kind": "episode_context", "text": "Scene... Task..."},
      {
        "index": 1,
        "kind": "action_segment",
        "text": "Action: reach the object",
        "frame_range": [0, 400],
        "metadata": {
          "source_segment_index": 0,
          "action_text": "reach the object",
          "skill": "reach"
        }
      }
    ]
  },
  "metadata": {
    "task_id": 327,
    "raw_episode_id": 648642
  }
}
```

`frame_range` helps model-side logic understand a text item but does not cause
the data factory to select that item automatically. Text item indices equal
their list positions and remain stable within one immutable Catalog version.

## SamplePool Header

The first JSONL line binds the Pool to its Catalog and split:

```json
{
  "record_type": "sample_pool_header",
  "schema_version": "motus_sample_pool_header_v1",
  "record_schema_version": "motus_sample_selection_v1",
  "sample_pool_id": "beta_model_selection_20260722_v1",
  "catalog_id": "agibot_beta_20260722_seed20260720_v1",
  "source_dataset": "agibot_beta",
  "source_child": "agibot_beta",
  "split": "train"
}
```

## Full-Modality Sample Selection

Every remaining JSONL line supplies explicit indices for all modalities:

```json
{
  "episode_id": "agibot_beta:327:648642",
  "video_frame_indices": [100, 101, 102, 103],
  "state_indices": [100],
  "action_indices": [101, 102, 103],
  "text_indices": [0, 1]
}
```

The data factory executes these selectors exactly:

- the first video index is the condition frame and the rest are future frames;
- state values are read at exactly `state_indices`;
- action values are read at exactly `action_indices`;
- text items are read and joined in exactly `text_indices` order;
- indices are never derived from another modality, sorted, shifted or padded.

Generic v1 requires at least one video index. State, action and text lists may
be empty when the model-side sampling strategy intentionally excludes those
modalities. Every supplied index must be a non-negative integer within its own
Catalog index space. The data factory preserves model-provided order and
repeated indices.

The Catalog/SamplePool contract does not impose 49 video frames, one state or
48 actions. Those are constraints of one current `motus_model_batch_v1`
integration profile and are checked only when that model profile is requested.
Another model may request any other lengths without changing this protocol.

## Responsibility Boundary

Model side owns:

- inspecting complete Episode data;
- choosing video, state, action and text indices;
- semantic and temporal alignment among those selections;
- producing the SamplePool.

Data factory owns:

- stable Episode access and independent modality index spaces;
- exact indexed reads in the order supplied;
- Catalog/source/split and bounds validation;
- source decoding and value normalization;
- conversion to the frozen sample/batch contract, batching and queueing.

If supplied indices are semantically misaligned but individually valid, that is
a SamplePool/model-side error. The data factory must not silently repair them.

If selections within one batch have different lengths, model-side loader policy
must choose length-homogeneous batches or an explicit padding/collate strategy.
The data factory must not silently truncate or pad them.
