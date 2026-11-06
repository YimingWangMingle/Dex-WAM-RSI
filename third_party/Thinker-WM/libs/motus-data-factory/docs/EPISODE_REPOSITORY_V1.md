# Versioned EpisodeRepository

`EpisodeRepository` is the model-side access boundary for a validated Episode
Catalog. It hides JSONL loading, tar offsets, MP4 extraction, HDF5 parsing and
source normalization.

## Open and enumerate

```python
from motus_data import EpisodeRepository

repository = EpisodeRepository("agibot_beta.episode_catalog.jsonl")

for info in repository.iter_episode_infos(
    source_child="agibot_beta",
):
    print(info.episode_id, info.video_count, info.modality_schemas)
```

Repository construction validates Catalog metadata, every Episode record,
source/child binding, unique Episode IDs, record count and the published
SHA-256 before any source data can be read.

For historical v1 Catalogs, `EpisodeInfo.split` is `train` or `test` and the
optional `split=` filter remains available. For split-neutral v2 Catalogs,
`EpisodeInfo.split` is `None`; requesting a split filter raises an explicit
error. Model-owned SamplePools select training or evaluation membership.

## Complete logical Episode

```python
episode = repository.get_episode("agibot_beta:327:648642")

print(episode.info)
print(episode.video.count, episode.video.schema)
print(episode.state.count, episode.state.schema)
print(episode.action.count, episode.action.schema)
print(episode.text.items)
```

`get_episode()` returns an `EpisodeView`. It represents the complete Episode
and all four index spaces but does not eagerly decode the complete video.

Uniform means every source exposes the same object model, method names, index
rules and result containers. It does not silently pad different robot
dimensions, resolutions or timeline lengths to a common shape.

## Exact independent reads

```python
frames = episode.read_video([10, 12, 15, 15])
states = episode.read_state([10, 12, 15])
actions = episode.read_action([11, 13, 16])
texts = episode.read_text([0, 2, 0])
```

All indices are zero-based and bounds checked against their own Catalog index
space. Order and repeated indices are preserved. The repository never derives
one modality's indices from another modality.

Video output modes:

```python
# Native resolution, uint8 [N,H,W,3], RGB.
frames = episode.read_video(indices)

# Optional model-oriented conversion, float32 [N,3,H,W] in [0,1].
frames = episode.read_video(
    indices,
    output_format="rgb_float32_chw",
    video_size=(480, 640),
)
```

Agibot Beta state and action reads return canonical normalized float32
`[N,26]` rows using the same normalization primitive as the accepted direct
training Dataset.

## Model-owned selection

```python
selected = episode.read_selection({
    "episode_id": episode.episode_id,
    "video_frame_indices": [100, 101, 102],
    "state_indices": [100],
    "action_indices": [101, 102],
    "text_indices": [0, 1],
})
```

The returned `EpisodeSelectionData` includes the original index tuples and the
four decoded values. This is a single-record inspection API and the primitive
that the later SamplePool Dataset will reuse.

## Complete Episode processing

Complete non-video modalities are safe to materialize:

```python
data = episode.materialize(modalities=("state", "action", "text"))
```

Process a complete video without a large allocation:

```python
for chunk in episode.iter_video_chunks(chunk_size=64):
    model_side_visual_processor(chunk.indices, chunk.frames)
```

Complete video materialization must be explicit and guarded:

```python
data = repository.load_full_episode(
    episode.episode_id,
    modalities=("video", "state", "action", "text"),
    max_video_bytes=6 * 1024**3,
)
```

The repository decodes one frame first, estimates the complete output tensor
size, and raises `EpisodeMemoryLimitError` before decoding the full video when
the estimate exceeds `max_video_bytes`.

## Current implementation status

The repository interface and Agibot Beta reader are implemented. Other source
families require their own reader adapters behind the same public API. The
SamplePool Dataset, batch construction and grouped union decode belong to
phase 5 and are not part of `EpisodeRepository` v1.
