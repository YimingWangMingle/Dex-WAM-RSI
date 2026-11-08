"""Prepare frozen task payloads from complete DexJoCo training demonstrations."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..runtime.memory_features import RGBEncoder, visual_features
from ..runtime.memory_pool import build_pool, digest
from ..runtime.task_knowledge import build_task_knowledge
from .guidance import copy_attribution, task_guidance
from .raw import load_public_episode


ANALYSIS_VERSION = "dexjoco-raw-training-icl/v1"


def _write_json(path, value):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def motion_anchors(state: np.ndarray, *, maximum: int = 8) -> list[int]:
    """Activity quantiles + endpoints; labels remain kinematic, not semantic."""
    if state.ndim != 2 or len(state) < 2 or state.shape[1] not in (23, 46):
        raise ValueError('expected at least two public robot states')
    dual = state.shape[1] == 46
    poses = [0, 7] if dual else [0]
    fingers = 14 if dual else 7
    activity = np.zeros(len(state) - 1)
    for offset in poses:
        activity += np.linalg.norm(np.diff(state[:, offset:offset+3], axis=0), axis=1) / .01
        q = state[:, offset+3:offset+7]
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        angle = 2 * np.arccos(np.clip(np.abs(np.sum(q[1:] * q[:-1], axis=1)), 0, 1))
        activity += angle / .1
    activity += np.mean(np.abs(np.diff(state[:, fingers:], axis=0)), axis=1) / .05
    # A small uniform component covers long stationary/contact intervals too.
    activity += max(float(np.mean(activity)) * .1, 1e-6)
    mass = np.r_[0., np.cumsum(activity)]
    anchors = np.searchsorted(mass, np.linspace(0, mass[-1], maximum)).tolist()
    return sorted(set([0, len(state)-1, *anchors]))


def numeric_descriptor(state: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    dual = state.shape[1] == 46
    pose = 14 if dual else 7
    values = [np.array([np.log1p(timestamps[-1] - timestamps[0])])]
    for offset in ([0, 7] if dual else [0]):
        xyz = state[:, offset:offset+3]
        values.extend([xyz[0], np.quantile(xyz - xyz[0], [.1, .5, .9], axis=0).ravel(),
                       np.array([np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()])])
    values.append(np.quantile(state[:, pose:], [.1, .5, .9], axis=0).ravel())
    return np.concatenate(values)


def diverse_representatives(descriptors: np.ndarray, count: int, *, group_sizes=None) -> tuple[list[int], dict]:
    """Robust-median seed + farthest-point coverage, balancing feature groups."""
    x = np.asarray(descriptors, dtype=float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all() or count < 1:
        raise ValueError('finite nonempty descriptors and positive count required')
    median = np.median(x, axis=0)
    scale = np.quantile(x, .75, axis=0) - np.quantile(x, .25, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.)
    z = np.clip((x - median) / scale, -8, 8)
    weights = None
    if group_sizes is not None:
        if len(group_sizes) != 3 or sum(group_sizes) != x.shape[1] or min(group_sizes) < 1:
            raise ValueError('three positive numerical/visual-mean/visual-std group dimensions required')
        weights = [.5, .25, .25]
        offset = 0
        for size, weight in zip(group_sizes, weights):
            z[:, offset:offset+size] *= np.sqrt(x.shape[1] * weight / size)
            offset += size
    chosen = [int(np.argmin(np.mean(z * z, axis=1)))]
    distances = np.mean((z-z[chosen[0]])**2, axis=1)
    while len(chosen) < min(count, len(z)):
        distances[chosen] = -1
        index = int(np.argmax(distances))
        chosen.append(index)
        distances = np.minimum(distances, np.mean((z-z[index])**2, axis=1))
    distances[chosen] = 0
    return chosen, dict(method='robust_median_seed_then_farthest_point',
                        requested=count, selected=len(chosen),
                        feature_group_sizes=group_sizes, feature_group_weights=weights,
                        max_squared_distance=float(distances.max()),
                        mean_squared_distance=float(distances.mean()),
                        feature_scope='training-only kinematics and RGB appearance; not learned semantics')


def _sample_video(path, indices, count):
    """Decode in chronological order and retain only the requested RGB frames."""
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open source video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        declared = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if declared != count or not np.isclose(fps, 30., atol=.05):
            raise ValueError(f"{path}: expected {count} frames at 30 Hz, got {declared} at {fps}")
        wanted = set(indices)
        if not wanted or min(wanted) < 0 or max(wanted) >= count:
            raise ValueError(f"{path}: frame indices outside the recording")
        frames = {}
        for i in range(max(wanted) + 1):
            if not cap.grab():
                raise ValueError(f"{path}: failed to decode frame {i}")
            if i in wanted:
                ok, pixels = cap.retrieve()
                if not ok:
                    raise ValueError(f"{path}: no RGB image at frame {i}")
                frames[i] = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
        return frames, dict(frames_declared=declared, fps=fps, decoded_indices=sorted(frames))
    finally:
        cap.release()


def _analyze(task, episode, temp):
    episode_id, path = episode
    data = load_public_episode(path, task)
    state, actions, times = data["state"], data["actions"], data["timestamps"]
    anchors = motion_anchors(state)
    sample_indices = sorted(set(anchors) | set(np.linspace(0, len(state) - 1, 8, dtype=int).tolist()))
    cameras = sorted(data["cameras"])
    camera_frames, videos = {}, {}
    for camera in cameras:
        video = data["cameras"][camera]
        camera_frames[camera], videos[camera] = _sample_video(video, sample_indices, len(state))
        videos[camera]["source_sha256"] = digest(video)
    encoder = RGBEncoder()
    appearance = np.stack([
        visual_features({cam: camera_frames[cam][i] for cam in cameras}, cameras, encoder)
        for i in sample_indices
    ])
    numeric = numeric_descriptor(state, times)
    descriptor = np.r_[numeric, appearance.mean(axis=0), appearance.std(axis=0)]
    archive = temp / (episode_id + ".npz")
    np.savez_compressed(archive, state=state, actions=actions, timestamps=times,
                        anchors=np.asarray(anchors), descriptor=descriptor)
    row = dict(task=task, episode_id=episode_id, analysis_version=ANALYSIS_VERSION,
               steps=len(state), action_contract=data["action_contract"],
               public_state_dim=state.shape[1], action_dim=actions.shape[1],
               duration_seconds=float(times[-1] - times[0]), kinematic_anchors=anchors,
               semantic_alignment="unverified; anchors are not contact or phase labels",
               cameras=cameras, videos=videos, numerical_sha256=digest(archive),
               native_success=None, outcome="unknown")
    return row, descriptor, [len(numeric), appearance.shape[1], appearance.shape[1]], data["cameras"]


def _one_representative(task, rank, row, videos, temp, image_source, knowledge_source, guidance):
    episode = row["episode_id"]
    anchors = row["kinematic_anchors"]
    with np.load(temp / (episode + ".npz"), allow_pickle=False) as data:
        state, actions, times = data["state"], data["actions"], data["timestamps"]
    ref_id = f"reference-{rank:02d}"
    ref_path = f"references/{ref_id}.npz"
    np.savez_compressed(knowledge_source / ref_path, state=state, actions=actions,
                        timestamps=times, kinematic_anchors=np.asarray(anchors))
    reference = dict(id=ref_id, path=ref_path, kind="numpy_trajectory",
                     description=f"Full public demonstration {episode}; {len(state)} steps, kinematic anchors only; adapt before execution.",
                     episode_id=episode, steps=len(state), action_contract=guidance["action_contract"],
                     source_split="training", outcome="unknown", semantic_alignment="unverified")
    windows = [sorted({a, (a + b) // 2, b}) for a, b in zip(anchors, anchors[1:])]
    indices = sorted({i for window in windows for i in window})
    frame_paths = {}
    image_dir = image_source / "images" / episode
    image_dir.mkdir(parents=True)
    for camera, video in sorted(videos.items()):
        if digest(video) != row["videos"][camera]["source_sha256"]:
            raise ValueError(f"{task}/{episode}: {camera} video changed during preparation")
        frames, _ = _sample_video(video, indices, len(state))
        for i, pixels in frames.items():
            dest = image_dir / f"{camera}-{i:06d}.png"
            image = Image.fromarray(pixels)
            image.thumbnail((384, 384))
            image.save(dest)
            frame_paths[camera, i] = dest.relative_to(image_source).as_posix()
    records = []
    for window_id, indices in enumerate(windows):
        records.append(dict(
            record_id=f"{task}-{ref_id}-window-{window_id:02d}", task=task, episode_id=episode,
            scene_id=f"training:rand_full:{task}:{episode}", source_type="human", source_split="training",
            channel="icl", action_contract=guidance["action_contract"], instruction=guidance["instruction"],
            outcome="unknown", outcome_evidence="",
            note=f"Full-corpus diversity-selected {ref_id}; interval {indices[0]}..{indices[-1]}. "
                 "Kinematic change window, NOT an identified grasp/contact/semantic phase. "
                 "Use task guidance and current RGB to identify the stage; adapt absolute coordinates.",
            steps=[dict(step=i, state=state[i].tolist(), action=actions[i].tolist(),
                        images={cam: frame_paths[cam, i] for cam in row["cameras"]}) for i in indices]))
    return reference, records


def _knowledge_documents(root, knowledge):
    sections = {"PHASES": "phases", "RECOVERY": "recovery", "REFERENCES": "references",
                "COVERAGE": "coverage", "SOURCES": "provenance"}
    _write_json(root / "knowledge.json", knowledge)
    for filename, key in sections.items():
        _write_json(root / (filename + ".json"), knowledge[key])
    lines = ["# " + knowledge["task"], "", knowledge["guidance"], "",
             "The phase guide summarizes historical demonstration analysis. "
             "Check applicability against current observations. "
             "Recovery suggestions are hypotheses, not independently validated recovery results.",
             "", "## Task phases"]
    for phase in knowledge["phases"]:
        lines += ["", "### " + phase["name"], phase["instruction"],
                  *["- " + check for check in phase["observable_checks"]]]
    lines += ["", "## Conditional recovery hypotheses"]
    for recovery in knowledge["recovery"]:
        lines += ["", "- Symptom: " + recovery["symptom"], "  Response: " + recovery["response"]]
    lines += ["", "Kinematic anchors in the numerical references have no verified semantic phase labels.",
              "No simulator success rate or recovery effectiveness was measured during this build."]
    (root / "TASK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _task_entry(payload, knowledge):
    with (payload / "pool/records.jsonl").open(encoding="utf-8") as stream:
        record = json.loads(next(stream))
    step = record["steps"][0]
    lines = ["# " + knowledge["task"] + " — ICL task guidance", "",
             "Task: " + record["instruction"],
             f"Action contract: `{knowledge['action_contract']}`; public state/action dimensions: "
             f"{len(step['state'])}/{len(step['action'])}.", "", knowledge["guidance"], "",
             "## Procedure and visible checks"]
    for index, phase in enumerate(knowledge["phases"], 1):
        lines.append(f"{index}. **{phase['name']}.** {phase['instruction']}")
        if phase["observable_checks"]:
            lines.append("   Check: " + "; ".join(phase["observable_checks"]))
    lines += ["", "## Conditional recovery",
              "These are demonstration-derived hypotheses, not confirmed results of the current episode."]
    for recovery in knowledge["recovery"]:
        lines.append("- If " + recovery["symptom"] + " Then " + recovery["response"])
    lines += ["", "## Historical visual reference",
              f"The following images belong to source episode `{record['episode_id']}`, step "
              f"{step['step']}; they are not current observations."]
    for camera, name in sorted(step["images"].items()):
        lines.append(f"- {camera}: [{camera} reference](pool/{name})")
    lines += ["", "Full phase checks and sources: `knowledge/knowledge.json`. "
              "Numerical/image examples: `pool/records.jsonl`. "
              "Use the current observation to select and adapt a compatible example; "
              "historical success does not establish current success."]
    entry = "\n".join(lines) + "\n"
    if len(entry) > 4000 or "https://" in entry or "http://" in entry:
        raise ValueError(f"{knowledge['task']}: task entry exceeds the native text budget or contains external URLs")
    (payload / "ICL.md").write_text(entry, encoding="utf-8")
    (payload / "ENTRY.md").write_text(
        "Read ICL.md for this task's procedure, visible checks, conditional recovery and reference images.\n"
        "Detailed task knowledge is in knowledge/knowledge.json; example trajectories and image paths are "
        "in pool/records.jsonl. These are historical examples, not current observations.\n", encoding="utf-8")


def prepare_task(task, episodes, work_dir, payload_dir, *, workers, revision=ANALYSIS_VERSION):
    """Analyze this task's 100 raw demonstrations, then write a prepared ICL payload."""
    episodes = sorted(episodes, key=lambda item: item[0])
    if len(episodes) != 100 or len({ep for ep, _ in episodes}) != 100:
        raise ValueError(f"{task}: expected 100 distinct training episodes")
    if workers < 1:
        raise ValueError("workers must be positive")
    work_dir, payload_dir = Path(work_dir), Path(payload_dir)
    temp = work_dir / "public"
    temp.mkdir(parents=True)
    knowledge_source = work_dir / "knowledge"
    (knowledge_source / "references").mkdir(parents=True)
    image_source = work_dir / "frames"
    payload_dir.mkdir(parents=True, exist_ok=True)
    guidance = task_guidance(task)

    def analyze(episode):
        try:
            return _analyze(task, episode, temp)
        except Exception as exc:
            raise ValueError(f"{task}/{episode[0]}: {exc}") from exc

    with ThreadPoolExecutor(max_workers=workers) as executor:
        analysis = list(executor.map(analyze, episodes))
    rows, descriptors, sizes, videos = zip(*analysis)
    if any(s != sizes[0] for s in sizes) or any(row["action_contract"] != guidance["action_contract"] for row in rows):
        raise ValueError(f"{task}: inconsistent layouts across demonstrations")
    descriptors = np.stack(descriptors)
    chosen, selection = diverse_representatives(descriptors, 8, group_sizes=sizes[0])
    _write_json(knowledge_source / "corpus_index.json", rows)
    np.savez_compressed(knowledge_source / "corpus_descriptors.npz", descriptors=descriptors,
                        episode_ids=np.asarray([row["episode_id"] for row in rows]))
    references, records = [], []
    for rank, index in enumerate(chosen):
        ref, windows = _one_representative(task, rank, rows[index], videos[index],
                                            temp, image_source, knowledge_source, guidance)
        references.append(ref)
        records.extend(windows)
    coverage = dict(episodes_seen=len(rows), numerical_steps=sum(r["steps"] for r in rows),
                    independent_representatives=len(chosen), retrieved_windows=len(records),
                    selected_episodes=[rows[i]["episode_id"] for i in chosen], selection=selection,
                    regime="rand_full", expected_episodes=100, complete=True,
                    semantic_phase_alignment="not automatically established",
                    recovery_validation="not evaluated; demonstration-derived conditional hypotheses")
    provenance = dict(guidance["provenance"], analysis_version=ANALYSIS_VERSION,
                      task_corpus_sha256=digest(knowledge_source / "corpus_index.json"))
    provenance["guidance_source_dataset_revision"] = provenance.pop("dataset_revision", None)
    provenance["input_verification"] = (
        "Caller-supplied training recordings. The task corpus index binds the exported "
        "public arrays, image features and hashes of both supplied videos for every episode. "
        "The guidance source revision identifies the curated guidance's reference dataset.")
    knowledge = dict(schema="task-icl-knowledge/v1", task=task,
                     action_contract=guidance["action_contract"], guidance=guidance["guidance"],
                     phases=guidance["phases"], recovery=guidance["recovery"], references=references,
                     provenance=provenance, coverage=coverage)
    _knowledge_documents(knowledge_source, knowledge)
    copy_attribution(task, knowledge_source / "attribution")
    build_task_knowledge(knowledge_source, payload_dir / "knowledge")
    is_dual = rows[0]["public_state_dim"] == 46
    state_scales = ([.1] * 3 + [1.] * 4) * (2 if is_dual else 1) + [.5] * (32 if is_dual else 16)
    build_pool(records, image_source, payload_dir / "pool", cameras=rows[0]["cameras"], state_scale=state_scales)
    _task_entry(payload_dir, knowledge)
    _write_json(payload_dir / "CONTEXT_MEMORY.json",
                dict(schema="context-memory-package/v1", revision=revision, stage="initial",
                     task=task, mode="icl", channels=["icl"], record_count=len(records),
                     action_contract=guidance["action_contract"], pool_path="pool", knowledge_path="knowledge",
                     knowledge_revision=digest(payload_dir / "knowledge/knowledge_manifest.json")))
    return coverage
