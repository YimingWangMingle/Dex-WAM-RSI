No hosted web research or external benchmark material was used. No other
learned model, additional agent, weight update, hidden test episode,
privileged state, evaluator, or scoring implementation was used.

Sources and their uses:

- **Supplied task specification (`task.json`).** Task
  `bimanual_unlock_ipad`, instruction to grasp and enter `123`, dual-arm
  layouts, 30 Hz, `rand_full`, and camera-name mapping. The robot README's
  referenced `robot.json` was absent; layouts came from this specification
  and the released array shapes.
- **Trusted released demonstration reader**, restricted to `list`, `arrays`,
  and `video`. All 100 numerical demonstrations, IDs `0000`–`0099`, were
  examined. Arrays contain only robot `state`, `actions`, and `timestamps`.
  Used for control units/order, pointing posture, grasp/hold trajectories,
  timing, and aggregate variation. The package retains only reference
  `0000` (392 frames), `0002` (first 176), and `0003` (first 166), with their
  actual proprioception and actions; no videos or full-corpus copies.
- **Released RGB demonstrations.** Sampled `random_camera` from all 100
  demonstrations for approach, grasp, held posture, and visible final
  appearance. Read `wrist_left` for all 100, examining initial frames and
  pre-grasp motion for the bezel landmark and its geometric reconstruction.
  Read `wrist_right` for `0000`, `0001`, `0002`, `0003`, `0014`, `0026`,
  `0028`, `0030`, `0033`, `0039`, `0056`, `0080`, `0081`, `0088`, `0099`;
  used initial views and selected held/tapping frames, particularly `0000`,
  for digit layout, transient yellow highlights, and the orange home screen.
  Frames were decoded with PyAV; contours, optical flow, triangulation,
  and planar pose fitting used classical numerical/image methods.
- **Pinned robot reference README and MJCF.** Robot commit
  `8d23b0fab23b17a58c4b55f3942e17013aaf8267`, files
  `panda_allegro_bimanual.xml`, `panda_allegro_right.xml`, and
  `panda_allegro_left.xml`, including their robot mesh dependencies when
  loading the robot for a forward-kinematics check. Used only for wrist
  camera intrinsics/extrinsics, attachment-site frame convention, Allegro
  joint ordering, and right-index capsule geometry. No task scene was
  inferred from these files.

Packaged reference images are annotated/resized released frames from
`0000`: `images/initial_corner.jpg` is left frame 0;
`images/grasp_sequence.jpg` uses base/left frames 0, 60, 80, 100, 140, 180;
`images/password_feedback.jpg` uses right frames 180, 270, 330, 391.
Annotations are explanatory and are not additional observations.

The approximate key pitch, screen-corner world positions, motion phase
boundaries, and color heuristics are inferences from those released
observations. The initial probe and closed-loop tapping procedure are
proposed control rules. Their numerical checks do not constitute task
rollout results.
