# Indexed Batch Adapters

`EpisodeRepository`, `SamplePoolDataset` and `IndexedEpisodeBatch` are the
stable data-factory boundary. A model batch is produced by a replaceable
adapter after indexed decoding:

```python
from motus_data import (
    IndexedBatchCollator,
    build_indexed_batch_adapter,
)

adapter = build_indexed_batch_adapter(
    "motus_model_batch_v1",
    num_video_frames=48,
    action_video_fps_ratio=1,
)
loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    collate_fn=IndexedBatchCollator(adapter),
    num_workers=0,
)
```

The current `motus_model_batch_v1` profile is an integration target, not the
canonical data-factory representation. It maps 49 explicitly selected video
frames to one conditioning frame plus 48 future frames, requires one explicit
state index, 48 explicit action indices and at least one explicit text index,
then validates the existing Motus trainer contract.

The adapter never derives, sorts, pads, truncates or repairs modality indices.
An incompatible selection fails with `BatchAdapterError`. The model side can
change its batch contract by registering another adapter:

```python
register_indexed_batch_adapter("model_batch_v2", ModelBatchV2Adapter)
adapter = build_indexed_batch_adapter("model_batch_v2", ...)
```

No Catalog, SamplePool, Repository, source reader, grouped Dataset or sampler
change is required for a new model profile.

## Phase-6 parity gate

`create_agibot_beta_parity_pool.py` converts the frozen phase-1 direct-Dataset
baseline into an explicit full-modality Pool solely for migration testing. Its
header is marked `legacy_parity_fixture_only=true`; it is not a production
SamplePool generator.

`audit_agibot_beta_phase6.py` then:

1. replays the frozen selections through the old `load_window()` path;
2. checks that replay against the phase-1 full-tensor hashes;
3. reads the same explicit selections through Catalog/SamplePool;
4. applies the current Motus adapter;
5. compares every model tensor, selected index, text and identity field;
6. validates the final `motus_model_batch_v1` batch.

The old and new `manifest_ids` intentionally differ. The old value identifies
a direct-Dataset window; the new value identifies the complete explicit
SamplePool selection. It is diagnostic and is not a model tensor.
