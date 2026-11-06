# Motus Model Batch Contract v1

`motus_model_batch_v1` is the frozen boundary between the standalone data
factory and the current Motus trainer. Model code may depend on this contract,
but it must not depend on source names, Manifest/HDF5 layouts, converters or
source-specific Dataset classes.

Required tensors:

- `first_frame`: float `[B,C,H,W]`, finite and in `[0,1]`
- `video_frames`: float `[B,T,C,H,W]`, finite and in `[0,1]`
- `initial_state`: float `[B,D]`, finite
- `action_sequence`: float `[B,T_action,D]`, finite
- `action_loss_weights`: float `[B]`
- `video_loss_weights`: float `[B]`

Required per-sample lists:

- `first_frame_pils`
- `t5_input_messages`
- `vlm_input_messages`
- `robot_names`
- `source_datasets`
- `state_action_policies`

Required placeholders:

- `vlm_inputs`
- `language_embedding`

Before model-side text preprocessing, both placeholders may be `None`.
`source_datasets` must be homogeneous within a batch. Diagnostic keys such as
`source_children`, IDs and sampled frame indices may be extended without
changing v1. Current optional diagnostics include:

- `episode_ids`: source-native episode IDs, list length `B`
- `manifest_ids`: stable sample/window IDs; the historical key name is kept
  for model compatibility even when no Manifest exists
- `frame_indices`: source-frame indices used by each sample
- `action_video_fps_ratios`: positive integers, list length `B`; a homogeneous
  batch has one ratio, and `T_action == T * ratio`

The generic data-factory contract permits missing action/state for consumers
that are truly video-only. The current Motus Stage1 trainer is stricter: it
still reads and transfers action/state before the Stage1 model branch ignores
them. Therefore its adapter must continue emitting these tensors. Training and
loss participation remain controlled by model/YAML switches.

## Acceptance Levels

Data-factory changes are accepted primarily at the interface boundary:

1. Source fixtures and real-data audits prove modal parsing, frame/text/action
   alignment, deterministic split behavior and source-specific safety rules.
2. `validate_motus_model_batch_v1` checks every tensor/list shape, finite value,
   image range, batch homogeneity and optional temporal ratio metadata.
3. The independent tiny consumer executes DataLoader, weighted MixDataset,
   ReadyBatchQueue, device transfer, forward, loss, backward and optimizer
   without importing the large model.

A full Motus smoke is an optional integration gate when the model environment
is available. It is required when this batch schema, the model-side adapter, or
the model's interpretation of existing fields changes. It is not required for
every source-specific Dataset implementation that preserves this contract.

## Ownership Boundary

The data factory owns source Datasets, decoding, runtime sampling, multimodal
alignment, split selection, weighted homogeneous batches, DataLoader and queue
implementation. The model owns its YAML model/training policy, device transfer,
text/VLM preprocessing, branch freezing, losses, forward/backward and optimizer.
Model code must not import source-specific Dataset classes or storage backends.
