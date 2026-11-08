This policy uses only task inputs, the authorized demonstration reader, and the supplied robot reference. No hosted web research, external benchmark material, other learned model, or other agent was used. No hidden test episodes, results, evaluator code, privileged state, or original Zarr data were accessed.

| Source | Use |
|---|---|
| Supplied `input/task.json` | Task prompt (`pick_bucket`), single-arm layout, 30 Hz control, `rand_full` regime, 100 demonstrations, public state/action lengths, camera mapping. |
| Trusted reader `list` | Released IDs 0000–0099, lengths, available `random_camera` and `wrist` streams. |
| Trusted reader `arrays` for all 100 IDs | Only `state`, `actions`, `timestamps`: aggregate timing, trajectory and target-tracking statistics, hand postures, phase candidates, and three compact retained trajectories (0036, 0038, 0097). Public state was treated solely as robot proprioception. |
| Trusted reader `video`, `random_camera`, all 100 IDs | All initial/final scene surveys; denser phase review for 0000, 0001, 0003, 0016, 0034, 0036, 0038, 0060, 0070, 0097. Used to identify task sequence, appearance variation, containment/lift cues, and handle-slip recovery. |
| Trusted reader `video`, `wrist`, IDs 0000, 0001, 0003, 0016, 0034, 0036, 0038, 0060, 0070, 0097 | Selected close views of grasps, releases, and handle motion; manual landmark measurements and robot projection checks. Packaged reference JPEGs are selected decoded frames from 0036, with labels/landmark annotations where shown. |
| Supplied `input/robot_reference/README.md` | Established that the pinned MJCF describes only the robot and identified upstream commit `8d23b0fab23b17a58c4b55f3942e17013aaf8267`. |
| Supplied `input/robot_reference/mjcf/panda_allegro_copy.xml` | Robot joint limits/order, hand kinematic chain, flange-site frame, wrist camera mounting and field of view. Only relevant numerical constants were extracted into `robot_geometry.json`. No task scene was inferred from this robot file. |

The reference README mentions `input/robot.json`, but that file was absent. Public layouts came from `task.json`; orientation ordering/frame were checked against the MJCF sensors and visible projections. Supplementary URDF and robot mesh contents were not needed.

`provenance.json` records trusted-reader SHA-256 values for every acquired numerical/base stream and the ten acquired wrist streams. Scene heights, object dimensions, and anchors in `reference.json` are approximate measurements from released RGB paired with public robot pose, not privileged labels. `control.py` is newly written numerical geometry/interpolation and a proposed task procedure; it contains no trained weights. Package validation is offline arithmetic/geometry checking only, not a test-episode success claim.
