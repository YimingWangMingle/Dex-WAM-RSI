# DexJoCo ICL

Each episode opens the current task's frozen experience package. At each decision,
`get_context_memory(obs)` retrieves historical examples using RGB appearance and
robot state. `build_messages` places their text and images alongside the current
observation in the model request.

## Shared experience release

Set `ICL_CATALOG` in the launch environment to the shared release's
`catalog.json`. Keep the release's relative `packages/` directory, manifests,
payloads and attribution files together. The native `experience.packages`
loader verifies the sealed package. Give each episode a new writable output
directory for its snapshot and access log.

To seal prepared task payloads into a new shared release, run
`python -m experience.icl.seal_catalog --source-root "$ICL_PREPARED_ROOT" --output "$ICL_NEW_RELEASE" --revision icl-v1`.
Each task directory supplies `ICL.md`, `pool/` and `knowledge/`. An existing
native task package with its manifest and payload is also accepted. Use
`--task click_mouse` to publish one task. The resulting catalog lives at
`$ICL_NEW_RELEASE/catalog.json`.

From the repository root, run an offline check with a new output directory:

```sh
python -m pip install -r experience/icl/requirements.txt
python -m experience.icl.example --task click_mouse --catalog "$ICL_CATALOG" --output outputs/icl_click_mouse
```

The example captures the assembled request in `request.json` and shows the
current and historical images in `preview.html`. The capture uses a training
observation and performs no model inference.

## First-group integration

```python
import os
from experience.icl.prompt_builder import get_context_memory, open_icl
from experience.icl.runtime.messages import build_messages

with open_icl(obs["task"], episode_dir=episode_dir,
              catalog_path=os.environ["ICL_CATALOG"]) as icl:
    # Run the existing episode loop inside this block. On each action decision:
    context = get_context_memory(obs)
    messages = build_messages(obs, context, system_prompt=policy_prompt, history=history)
    response = client.complete(messages=messages, tools=action_tools)
```

Use the existing multimodal client, policy prompt, action tools and execution
loop. Supply the latest public `obs` at each decision. Required fields are
`task`, `episode_id`, `instruction`, `action_contract`, `state` and `images`.
If available, pass the layout identifier as `scene_id`. Keep the episode and
scene identifiers consistent across decisions in a single episode.

| Robot | Action contract | Public state | RGB cameras |
|---|---|---|---|
| Single arm | `dexjoco/single-xyz-rotvec-hand16/v1` | XYZ, quaternion WXYZ, 16 hand angles (23 values) | `base`, `wrist` |
| Dual arm | `dexjoco/dual-right-left-xyz-rotvec-hand16/v1` | Right and left poses, then right and left hand angles (46 values) | `wrist_left`, `wrist_right` |

Use metres for XYZ and radians for joint angles. Supply synchronized RGB images,
for example as `uint8` arrays of shape `(H, W, 3)`. For integration with the
native runner, pass `**icl.run_options` to use the same frozen workspace.
