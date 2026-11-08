# Sources and use

No external web sources or outside benchmark material were used. No other learned model, agent, hidden scene state, evaluator, test seed, test result, or future episode was accessed.

- Provided `input/task.json`: task prompt, single-arm action/state dimensions, 30 Hz rate, camera mapping, 100 demonstrations, and `rand_full` regime.
- Trusted demonstration reader: only `list`, `arrays`, and `video` were requested. All numerical releases `0000`–`0099` were analyzed (28,212 steps); only `state`, `actions`, and `timestamps` were available. Both released RGB streams were fetched and sampled across every demonstration. Initial and terminal base-view mosaics, representative phase sequences, and selected wrist views were visually inspected. Longer/atypical traces were included in this review.
- Provided pinned robot `README.md` and `mjcf/panda_allegro_copy.xml`, upstream commit `8d23b0fab23b17a58c4b55f3942e17013aaf8267`: wrist/site frames, camera mounting and field of view, finger ordering, and joint limits. This robot reference was not treated as a task-scene definition.

`reference_traces.npz` retains the original numerical arrays for six examples: `0000`, `0019`, `0032`, `0036`, `0058`, `0099`. Their source NPZ SHA-256 values are in `policy_data.json`. Images are annotated or resized frames from released `0000` RGB video. No original video or robot mesh copies are required at test time.

Object geometry and scene anchors in `policy_data.json` are **estimates**, not privileged state. The head landmarks were triangulated from `0000` wrist frames 35/50, with a rectangular-face geometric fit. Alternate resting hammer placements used manually identified head landmarks and camera projection. Board-height statistics used classical SIFT correspondences and stereo geometry in 92 usable released pairs (8,865 points). The canonical nail anchor was checked in later, widely separated pre-strike views; alternate nail anchors are approximate as documented in `API.md`.

Python numerical/image packages were used for array analysis, MP4 decoding, image inspection, and geometric calculations. There was no neural inference or weight update. The geometric exemplar ranking uses explicit hand-set distance scales.

Offline validation passed: six-trace identity preservation, independent grasp/nail transforms, camera projection/triangulation consistency, and rotation interpolation across ±π. Canonical resting-head fitting RMS was 0.65 px; second-view maximum landmark error was 3.95 px; a separately inspected nail landmark in frame 140 was within 2.20 px. Resting-head annotations in all six retained examples fit at 0.48–1.38 px RMS. These checks validate geometry and data handling, **not task success**. Zero simulation task episodes were run.

An isolated fresh-directory check with a read-only package copy also passed: self-test and end-to-end landmark-plan generation required no demonstration or scratch files. Action output had finite 22-value rows and legal finger targets. Auxiliary correction helpers preserved the supplied grip.
