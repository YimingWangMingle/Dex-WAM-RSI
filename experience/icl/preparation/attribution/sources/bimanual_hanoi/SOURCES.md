# Sources and permitted scope

No external web sources, outside model calls, learned vision models, other agents, or weight updates were used. No benchmark search was performed. The package is based only on the supplied task, released demonstration observations/actions, and the pinned robot reference.

| Preparation source | Use |
|---|---|
| Supplied user instructions and `input/task.json` | Exact task order, hand assignments, 30 Hz control, action/state layouts, camera names, `rand_full` regime, corpus size. |
| Trusted demonstration reader, `list` | Released IDs 0000–0099 and episode lengths. |
| Trusted demonstration reader, `arrays DEMO_ID`, for all 100 released IDs | Numerical corpus survey, initial-state consistency, action semantics and control lag; selection of complete motion references. Returned arrays contained only `state`, `actions`, `timestamps`. |
| Trusted demonstration reader, `video DEMO_ID CAMERA` | Initial/final external views and initial wrist views throughout the corpus; detailed temporal inspection of selected demonstrations. PyAV decoded the released original RGB videos. No raw Zarr files were accessed. |
| Released demonstrations 0044 and 0083 | Exact retained action/state/timestamp arrays, hand posture anchors, phase reference images, and visual peg triangulation. Source-array SHA-256 values and packaged-array hashes are in `reference.json`; sampled-image provenance is in `references/index.json`. |
| Released demonstration 0007, initial `random_camera` frame | Clear semantic layout illustration. |
| `input/robot_reference/README.md`, `mjcf/panda_allegro_bimanual.xml`, `mjcf/panda_allegro_right.xml`, `mjcf/panda_allegro_left.xml` | Robot-only definitions at upstream commit `8d23b0fab23b17a58c4b55f3942e17013aaf8267`: flange-site frame, wrist-camera extrinsics/FOV, hand joint order and ranges. Robot-only forward kinematics checked frame conventions; the files were not treated as task-scene state. |

All peg coordinates in this package are estimates made from selected released RGB frames and their simultaneous public robot poses. They are not privileged object coordinates. Quaternion ordering and absolute action semantics were cross-checked numerically against the public state/action sequences.

No task evaluator, hidden state suffix, test seed, future episode, or test outcome was accessed or inferred from unavailable files. [Offline validation](validation.json) covers package integrity, mathematical helpers, consistency with released observations, and execution from an isolated read-only package copy; it does not establish a held-out task success rate. The original demonstrations and scratch analysis are not required at episode time.
