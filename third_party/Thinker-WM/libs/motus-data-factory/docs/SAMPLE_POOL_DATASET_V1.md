# Versioned SamplePool Dataset

This layer consumes a model-owned full-modality SamplePool and converts its
records into source-independent indexed Episode data. It does not generate or
repair production selections.

## Input

```python
from motus_data import EpisodeRepository, SamplePool, SamplePoolDataset

repository = EpisodeRepository("agibot_beta.episode_catalog.jsonl")
pool = SamplePool("model_owned.sample_pool.jsonl", repository)
dataset = SamplePoolDataset(repository, pool)
```

Loading validates:

- the first non-empty record is a supported v1 or v2 SamplePool header;
- v1 header source/child/split bindings or concise v2 `catalog_id` binding
  match the Repository Catalog;
- every selected Episode exists; v1 additionally enforces split, while v2 is
  split-neutral;
- all five selection fields are present;
- every video/state/action/text index is inside its own Catalog index space;
- optional v1 `record_count` or v2 `samples` matches the selections.

SamplePool errors include the failing JSONL line number. No video or HDF5 data
is read during Pool validation.

## Same-Episode grouped batch

```python
sampler = dataset.create_batch_sampler(
    batch_size=4,
    drop_last=True,
    shuffle=True,
    num_replicas=world_size,
    rank=rank,
    seed=training_seed,
)
sampler.set_epoch(epoch)
```

Every emitted list contains selections from exactly one Episode. The sampler
shuffles deterministically by `seed + epoch`, partitions complete batches
across ranks without overlap, and gives every rank the same number of batches.

With `drop_last=True`, an Episode needs at least `batch_size` selections to
produce a batch; its remainder is not mixed with another Episode. A production
SamplePool should account for this when choosing how many selections to publish
per Episode.

## Union read and exact reconstruction

For all selections in one batch:

1. Validate every selection independently.
2. Build independent state and action index unions.
3. Open and normalize the Episode HDF5 timeline once.
4. Reconstruct each sample's exact state/action/text selection.
5. Run data-factory numeric safety validators before video I/O.
6. Build the unique union of requested video indices.
7. Extract the Episode MP4 tar member once and decode the union once.
8. Reconstruct every sample in its original index order, including repeats.

Example:

```text
sample A video = [0, 2, 0]
sample B video = [2, 4, 2]
decoded union  = [0, 2, 4]
```

The result still contains:

```text
A = [frame0, frame2, frame0]
B = [frame2, frame4, frame2]
```

The union is only a physical I/O optimization. It never changes sample
semantics.

## Numeric safety boundary

`SamplePoolDataset` rejects non-finite selected action/state values for every
source. It also enables the Agibot Beta magnitude guard by default with
threshold `100`. Both evaluate the exact state and action indices provided by
the model-owned SamplePool before MP4 extraction or frame decoding, so invalid
numeric data cannot cross the Dataset batch boundary or consume video decode
capacity.

```python
dataset = SamplePoolDataset(
    repository,
    pool,
    action_magnitude_guard_threshold=100.0,
    action_magnitude_guard_sources=["agibot_beta"],
)
```

Additional source magnitude validators can be added as their canonical numeric
semantics are implemented. Setting the threshold to `None` explicitly disables
only the magnitude rule and is intended for diagnosis; finite validation
remains mandatory. A production SamplePool should also be pre-audited before
distributed training. The runtime guard remains the final defense and must not
silently replace a model-owned selection.

## DataLoader output

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    collate_fn=dataset.collate_fn,
    num_workers=0,
)

batch = next(iter(loader))
```

The phase-5 output is `IndexedEpisodeBatch`:

```python
batch.info
batch.episode_id
batch.batch_size
batch.samples  # tuple[EpisodeSelectionData, ...]
```

Each sample contains its original four index tuples and decoded video, state,
action and text values. Lengths may differ across samples, so phase 5 does not
silently stack, pad or truncate them.

The current Motus fixed-shape `motus_model_batch_v1` adapter and direct-Dataset
parity checks belong to phase 6.

## Smoke-only fixture

`examples/create_sample_pool_smoke_fixture.py` creates a small, explicit test
Pool for infrastructure acceptance. Its v2 header contains
`meta.smoke_fixture_only=true`. It is not a production
SamplePool generator and does not replace model-side selection logic.
