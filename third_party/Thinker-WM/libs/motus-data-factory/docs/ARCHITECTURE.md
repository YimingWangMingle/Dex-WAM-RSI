# Architecture and Migration Gates

## Ownership

Each source adapter owns raw storage parsing, canonical modality decoding and
complete Episode access. In the indexed path, the model side owns SamplePool
window selection and cross-modal index alignment; the data factory performs
the exact indexed reads, grouped physical I/O, collation, DataLoader and queue.
Historical direct Datasets retain their runtime-window logic only for migration
and parity.

A SamplePool is anything satisfying the `SupportsSamplePool` protocol
(`selections` / `episode_groups` / `pool_id` / `__len__`, optional
`set_epoch`). The static JSONL-backed `SamplePool` is the built-in
implementation; training code may inject its own runtime-generated pool (or a
factory receiving the opened repository and pipeline seed) through the same
`sample_pool` parameter of `open_training_data` / `IndexedTrainingSource`.
Selection strategy is model-owned; the library only standardizes formats,
loading, packing and transport, and enforces selection/bounds validation at
read and adapt time regardless of pool kind.

`motus_manifest_v1` remains a compatibility adapter. Native adapters use
source-specific Datasets and return the same sample contract.

Configuration is also split by ownership. Model-side configuration owns Pool
selection, mixing, language, model-batch adapter, batch, workers and queue.
Data-side schemas own storage facts (HDF5 keys, field order, dimensions and
source-native timing facts). A model config may select an embodiment but may
not redefine its physical state/action layout.

## Stable Public Boundary

The stable indexed boundary is the versioned Episode Catalog,
`EpisodeRepository`, model-owned SamplePool and `IndexedEpisodeBatch`.
`motus_model_batch_v1` is the currently accepted adapter profile, not the
permanent data-factory representation. A later model batch registers a new
adapter without changing Catalog, Repository, SamplePool or source readers.

Required sample tensors:

- `first_frame`: `[C,H,W]`
- `video_frames`: `[T,C,H,W]`

Optional tensors:

- `initial_state`: `[D_state]`
- `action_sequence`: `[T_action,D_action]`

The collated batch keeps the current Motus key names. A homogeneous batch may
omit both action/state tensors in video-only mode. It may not mix samples with
and without the same optional tensor.

## Acceptance Gates

1. Unit contract: synthetic full-modal and video-only samples pass shape,
   collation, queue and optimizer tests.
2. Decoder contract: generated RobotMind HDF5/JPEG data passes exact frame and
   state/action alignment checks.
3. Legacy equivalence: fixed-window production config yields identical source,
   IDs, text and tensor signatures in old and new loaders.
4. Real-data boundary: at least 200 queued optimizer-smoke steps cover every
   configured child source without decode or non-finite failures.
5. Stability: single-node 8-GPU configuration runs with production batch size,
   worker count and queue size while CPU RSS and `/dev/shm` plateau.
6. Integration: one-node Motus training uses the library without changing model
   input shapes or loss switches.
7. Distributed: three-node source distribution, rank progress and queue metrics
   remain healthy before a long run is authorized.

## Dynamic Dataset Migration

Agibot is migrated first, with `effective_frames`, start-dependent action text,
and action/state slicing checked against the original Dataset. Agibot Beta is
second and preserves union decode by physical tar/video group. RobotMind follows
through one HDF5 source-family Dataset and small immutable per-body field maps.
TianYi is the first real-data gate; Ark, ArkMobile and TienKung reuse the same
reader after their schemas pass fixture and real-data audits. RoboTwin follows
as a separate source family.

For historical direct Datasets, train windows remain reproducible from global
seed, epoch, rank, worker, dataset and sample index. New split-neutral Catalogs
contain every accessible Episode. Separate model-owned Pool files determine
training and evaluation membership and must enforce their desired disjointness.
