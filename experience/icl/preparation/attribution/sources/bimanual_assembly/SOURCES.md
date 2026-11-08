No external website, paper, benchmark repository, pretrained model, or other agent was used. There are no external research sources to disclose and no external runtime resources.

Released sources used:

- `task.json`: task prompt, `bimanual_assembly`, `rand_full`, dual-arm layout, camera mapping, 30 Hz control, and demonstration count.
- Trusted reader `list`, `arrays DEMO_ID`, and `video DEMO_ID CAMERA` only. All 100 released IDs `0000`–`0099` supplied numerical state/actions/timestamps. These were used for phase identification, timing and pose statistics, grasp patterns, and the 16 retained motion examples.
- Released `random_camera` RGB streams for all 100 IDs: sampled initial/acquisition/transfer/insertion/final frames were visually inspected. All initial left/right wrist images were inspected. Selected wrist sequences, especially `0000`, `0038`, `0025`, `0063`, and `0082`, were inspected in greater detail. All right streams were available for geometric peg analysis; 61 of 100 initial silhouette fits met the preparation script's consistency filters. All stored object geometry is inferred from RGB and robot proprioception.
- Pinned robot reference README and the bimanual/right/left MJCF definitions at upstream commit `8d23b0fab23b17a58c4b55f3942e17013aaf8267`: used solely for attachment-site conventions, wrist-camera transforms/FOV, finger ordering, joint limits, and hand kinematics. `hand_kinematics.json` contains only extracted robot constants. The README's mentioned `robot.json` was absent; layouts came from the released task description and numerical data.

Derived package assets:

- `motions.npz`: original action vectors for 16 selected released demonstrations, with public state snapshots every fifth frame and at named checkpoints. No privileged object state is present.
- `references.json`, `corpus_summary.json`: measured phase/timing summaries and approximate RGB-derived peg geometry. Phase names such as `seated` are visual/numerical annotations near release, not evaluator labels.
- `references/` and `reference_index.jpg`: selected RGB frames, resized contact sheets, and fixed crops from those released videos. No original full videos or original demonstration archives are copied into the package.
- Python helpers: explicit pose arithmetic, interpolation, kinematics, pinhole projection, triangulation, geometric least squares, and optional color thresholding. No model weights were trained or updated.

Preparation used no original Zarr access, privileged state suffix, evaluator source, test seed, test result, or future episode. Validation is limited to released demonstrations, synthetic numerical consistency checks, and package self-containment.
