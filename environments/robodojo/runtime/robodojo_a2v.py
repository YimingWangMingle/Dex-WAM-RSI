"""Review-before-execution RPCs for the existing synchronous RoboDojo policy.

Preview never issues commands. Only select_candidate issues the exact prefix
whose full, postprocessed plan was predicted. Simulator feedback owns history.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import time
from uuid import uuid4

import numpy as np


def segment_frames(frame_steps, decision_step, execute_steps):
    """Mark each predicted frame as the executed prefix or the unexecuted lookahead.

    frame_steps are native control rows. Offsets 1..execute_steps are the prefix
    that select_candidate will issue. Later frames are hypothetical and are not
    compared with the next replan.
    """
    rows = []
    for index, step in enumerate(frame_steps):
        offset = int(step) - int(decision_step)
        if offset <= 0:
            raise ValueError('forecast frame must follow the measured observation')
        rows.append(dict(frame_index=index, control_step=int(step), offset_steps=offset,
                         segment='execute' if offset <= int(execute_steps) else 'lookahead'))
    return rows


def _visual_history(history, decision_step):
    """Accept only measured image records; never pass arbitrary reviewer history."""
    clean = []
    last = -1
    for record in history:
        if not isinstance(record, dict) or set(record) - {'step', 'images', 'source'}:
            raise ValueError('history requires measured step/images records only')
        step = record.get('step')
        images = record.get('images')
        if (type(step) is not int or not last < step < decision_step
                or record.get('source', 'measured') != 'measured'
                or not isinstance(images, (list, tuple)) or not images
                or any(not isinstance(path, str) or not path for path in images)):
            raise ValueError('history must contain ordered past measured images')
        clean.append(dict(step=step, images=list(images), source='measured'))
        last = step
    return clean


def mode3_astra_input(preview, *, images, history, instruction):
    """Strict visual-only view; caller supplies actual measured observation history."""
    if (preview.get('predict_steps') != 24
            or preview.get('control_hz') != 25
            or type(preview.get('decision_step')) is not int
            or preview['decision_step'] < 0
            or len(preview.get('candidates', [])) != 3):
        raise ValueError('mode 3 requires 3 candidates, a 24-step forecast and 25 Hz')
    if (not isinstance(images, (list, tuple)) or not images
            or any(not isinstance(path, str) or not path for path in images)):
        raise ValueError('current RGB image paths required')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('original task instruction required')
    if preview.get('instruction', instruction) != instruction:
        raise ValueError('instruction must match the preview task')
    shown, ids = [], set()
    depth_available = []
    for candidate in preview['candidates']:
        cid = candidate['candidate_id']
        if type(cid) is not int or cid not in range(3) or cid in ids:
            raise ValueError('candidate IDs must be unique integers 0, 1, 2')
        ids.add(cid)
        if {'actions', 'action_hash'} & set(candidate):
            raise ValueError('mode 3 Astra input cannot include candidate actions')
        media = candidate.get('forecast') or {}
        frames, steps = media.get('frames'), media.get('frame_steps')
        expected = [preview['decision_step'] + n for n in range(2, 25, 2)]
        if (not isinstance(frames, list) or len(frames) != 12
                or any(not isinstance(path, str) or not path for path in frames)
                or not isinstance(steps, list) or any(type(step) is not int for step in steps)
                or steps != expected):
            raise ValueError('mode 3 requires 12 A2V frames at offsets 2,4,...,24')
        depth = media.get('depth_frames')
        if depth is not None and (not isinstance(depth, list) or len(depth) != 12
                or any(not isinstance(path, str) or not path for path in depth)):
            raise ValueError('predicted depth must align with all 12 RGB frames')
        depth_available.append(depth is not None)
        marked = segment_frames(steps, preview['decision_step'], 24)
        for row in marked:
            row['segment'] = 'prediction'
        rows = []
        for i, (path, row) in enumerate(zip(frames, marked)):
            rows.append(dict(path=path, **row, offset_s=row['offset_steps'] / 25,
                             **({'depth_path': depth[i]} if depth is not None else {})))
        shown.append(dict(candidate_id=cid, frames=rows))
    if any(depth_available) and not all(depth_available):
        raise ValueError('all candidates must have the same depth availability')
    return dict(mode='mode3_a2v_visual_review', review_id=preview['review_id'],
                instruction=instruction, current_rgb=list(images),
                history=_visual_history(history, preview['decision_step']),
                decision_step=preview['decision_step'], allowed_execute_steps=[24, 16, 8, 4],
                control_hz=25, candidates=shown, response='candidate_id + execute_steps',
                depth_available=all(depth_available),
                depth_description='Model-predicted normalized depth, fixed [0,1] grayscale, bright=near, dark=far/invalid; '
                                  'not measured depth or calibrated metric distance.')


def save_forecast(forecast, directory, control_hz):
    """Export RGB frames without introducing another observation decoder."""
    from PIL import Image
    frames = forecast.frames
    if frames is None:
        raise ValueError('A2V requires decoded RGB frames')
    frames = frames.detach().float().cpu().numpy()
    if (frames.ndim != 5 or frames.shape[0:2] != (1, 3)
            or frames.shape[2] != len(forecast.frame_steps)
            or not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1):
        raise ValueError('expected decoded RGB [1,3,T,H,W] in [0,1]')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rgb = np.rint(frames[0].transpose(1, 2, 3, 0) * 255).astype(np.uint8)
    paths = []
    for index, frame in enumerate(rgb):
        path = directory / f'frame-{index:03d}.png'
        Image.fromarray(frame).save(path)
        paths.append(str(path))
    depth_paths = None
    if forecast.depth is not None:
        depth = forecast.depth.detach().float().cpu().numpy()
        if (depth.ndim != 5 or depth.shape[0] != 1 or depth.shape[1] not in (1, 3)
                or depth.shape[2:] != frames.shape[2:] or not np.isfinite(depth).all()
                or depth.min() < 0 or depth.max() > 1):
            raise ValueError('expected normalized decoded depth aligned with RGB')
        # Preserve the decoded tensor; fixed grayscale avoids per-candidate rescaling.
        np.save(directory / 'predicted-depth.npy', depth, allow_pickle=False)
        gray = np.rint(depth[0].mean(axis=0) * 255).astype(np.uint8)
        depth_paths = []
        for index, frame in enumerate(gray):
            path = directory / f'depth-{index:03d}.png'
            Image.fromarray(frame).save(path)
            depth_paths.append(str(path))
        if forecast.depth_content_mask is not None:
            mask = forecast.depth_content_mask.detach().cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(mask).save(directory / 'depth-content-mask.png')
    # PNGs are authoritative and sufficient for review; MP4 is optional.
    video, error = None, None
    try:
        import imageio.v2 as imageio
        delta = forecast.frame_steps[0] - forecast.decision_step
        if delta <= 0:
            raise ValueError('forecast must start after the measured observation')
        video = directory / 'prediction.mp4'
        imageio.mimwrite(str(video), rgb, fps=control_hz / delta, macro_block_size=1)
        video = str(video)
    except Exception as exc:
        video, error = None, str(exc)
    result = dict(frames=paths, depth_frames=depth_paths, video=video, video_error=error,
                  frame_steps=forecast.frame_steps, action_hash=forecast.action_hash,
                  predicted_only=True)
    (directory / 'forecast.json').write_text(json.dumps(dict(
        **result, metadata=forecast.meta()), indent=2, allow_nan=False))
    return result


class A2VReviewMixin:
    """Place before RoboDojoPolicy in the MRO; uses native XPL custom RPCs."""

    def __init__(self, usr_args):
        # Validate before allocating a second model or starting async workers.
        runtime = usr_args.get('runtime_config')
        if runtime and json.loads(Path(runtime).read_text()).get('async_lead_steps', 0):
            raise ValueError('review requires synchronous runtime: async_lead_steps=0')
        self._review = None
        super().__init__(usr_args)

    def reset(self):
        self._review = None
        return super().reset()

    def finish_episode(self):
        self._review = None
        return super().finish_episode()

    def get_action(self):
        raise RuntimeError('review enabled: use preview_candidates then select_candidate')

    def framework_identity(self):
        return dict(action_contract=self.profile.action_contract, action_dim=self.profile.action_dim,
                    control_hz=self.control_hz, checkpoint=self.inf.checkpoint)

    def framework_prediction_identity(self):
        from dataclasses import asdict
        from tools.prediction import a2v
        import hashlib
        return dict(self.framework_identity(), settings=asdict(self.inf.settings),
                    action_stats_version=getattr(self.inf,'action_stats_version',None),
                    camera_transform=getattr(self.inf,'camera_transform',None),
                    horizon=self.horizon.predict_steps,
                    implementation=hashlib.sha256(Path(a2v.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest())

    def framework_observe(self, obs):
        # Explicit external-skill replay. Only allowed with no pending/issued WAM motion.
        if self._review is not None or (self.observation is not None and
                self.next_decision != self.observation.step):
            raise RuntimeError('resolve WAM proposal before external observation replay')
        expected = 0 if self.observation is None else self.observation.step + 1
        if obs.get('rsi_control_step') != expected:
            raise ValueError('external measured rows must be contiguous')
        self.next_decision = expected
        return self.update_obs(obs)

    def framework_preview(self, request):
        if set(request) - {'mode3', 'predict', 'count', 'plans', 'seed_offset'} or any(type(request.get(k)) is not bool for k in ('mode3','predict')):
            raise ValueError('explicit framework mode/prediction flags required')
        if request['mode3'] != isinstance(self, Mode3ReviewMixin):
            raise ValueError('framework mode differs from the loaded service')
        count = request.get('count', 3 if request['mode3'] else 1)
        if type(count) is not int or not 1 <= count <= 8 or (request['mode3'] and count != 3):
            raise ValueError('invalid candidate count for framework mode')
        if request['mode3'] and ('plans' in request or 'seed_offset' in request):
            raise ValueError('strict visual mode does not allow action revision')
        payload = dict(count=count, predict=request['predict'], reveal_actions=not request['mode3'])
        if 'plans' in request:
            if not isinstance(request['plans'], list) or len(request['plans']) != count:
                raise ValueError('replacement plans must preserve candidate count')
            payload.pop('count')
            payload['plans'] = request['plans']
        if 'seed_offset' in request:
            payload['seed_offset'] = request['seed_offset']
        preview = self.preview_candidates(payload)
        # This internal transport envelope is not passed to the visual reviewer.
        return dict(preview=preview, candidates=copy.deepcopy(self._review[0]['candidates']))

    def framework_select(self, request):
        if set(request) != {'review_id', 'candidate_id', 'execute_steps'}:
            raise ValueError('framework selection requires review, candidate and prefix')
        if isinstance(self, Mode3ReviewMixin):
            return self.select_candidate(request)
        count = request['execute_steps']
        if type(count) is not int or not 1 <= count <= self.horizon.predict_steps:
            raise ValueError('invalid framework execution prefix')
        return self._select_candidate_prefix({k:request[k] for k in ('review_id','candidate_id')},count)

    def framework_feedback(self, request):
        if set(request) != {'executed_actions','control_step'} or self.last_command is None:
            raise ValueError('issued command and exact feedback required')
        start, commands = self.last_command
        step = request['control_step']
        actions = np.asarray(request['executed_actions'], dtype=np.float32)
        count = len(actions)
        if (type(step) is not int or step != self.observation.step or step != start+count
                or count > len(commands) or (count and not np.array_equal(actions,commands[:count]))):
            raise ValueError('feedback differs from selected/measured prefix')
        if count != len(commands) and self.continuity is not None:
            self.continuity.reset()
        self._record(dict(event='framework_feedback',step=step,executed_steps=count,
                          discarded_steps=len(commands)-count))
        self.next_decision = step
        self.last_command = None
        return dict(resolved=True,step=step)

    def _at_boundary(self):
        if self.observation is None or self.observation.step != self.next_decision:
            raise RuntimeError('all issued actions must have measured feedback before preview')
        if getattr(self, 'async_replan', None) is not None:
            raise ValueError('review cannot share asynchronous model inference')

    def preview_candidates(self, request):
        from thinker_wm.models.inference_api import action_hash
        self._at_boundary()
        if self._review is not None:
            raise RuntimeError('select or discard the pending preview first')
        if not isinstance(request, dict) or set(request) - {'plans', 'count', 'predict', 'reveal_actions', 'seed_offset'}:
            raise ValueError('preview accepts plans OR count, plus predict and reveal_actions')
        predict = request.get('predict', True)
        reveal_actions = request.get('reveal_actions', True)
        if type(reveal_actions) is not bool:
            raise ValueError('reveal_actions must be boolean')
        if type(predict) is not bool:
            raise ValueError('predict must be boolean')
        supplied = request.get('plans')
        if supplied is not None and 'count' in request:
            raise ValueError('supply plans or count, not both')
        count = len(supplied) if isinstance(supplied, list) else request.get('count', 1)
        if supplied is not None and not isinstance(supplied, list):
            raise ValueError('plans must be a list of full native joint14 plans')
        if type(count) is not int or not 1 <= count <= 8:
            raise ValueError('candidate count must be between 1 and 8')
        seed_offset = request.get('seed_offset', 0)
        if type(seed_offset) is not int or seed_offset < 0:
            raise ValueError('seed_offset must be a nonnegative integer')
        review_id = uuid4().hex
        private, public, stored = [], [], []
        start = time.monotonic()
        for index in range(count):
            # Share immutable weights, not mutable inference settings or filter state.
            inference = copy.copy(self.inf)
            inference.settings = replace(self.inf.settings,
                seed=self.inf.settings.seed + seed_offset + index, decode_outputs=False)
            raw = (supplied[index] if supplied is not None else
                   inference.generate_action(self.observation, self.history, self.instruction).actions)
            raw = np.asarray(raw, dtype=np.float32)
            if raw.shape != (self.horizon.predict_steps, 14) or not np.isfinite(raw).all():
                raise ValueError(f'candidate must be finite [{self.horizon.predict_steps},14]')
            continuity = copy.deepcopy(self.continuity)
            if supplied is not None:
                # External candidates are FINAL commands; no hidden re-smoothing.
                if continuity is not None:
                    raise ValueError('external final plans require continuity disabled')
                plan = raw.copy()
                if ((plan[:, [6, 13]] < 0) | (plan[:, [6, 13]] > 1)).any():
                    raise ValueError('external gripper commands must be in [0,1]')
            elif continuity is None:
                plan = raw.copy()
                plan[:, [6, 13]] = plan[:, [6, 13]].clip(0, 1)
            else:
                plan, _ = continuity.process(raw, self.observation.state,
                    self.observation.step, self.block_size, self.measured_velocity)
            plan = np.asarray(plan, dtype=np.float32).copy()
            media = None
            if predict:
                from tools.prediction.a2v import forecast_final_plan
                media = forecast_final_plan(self, plan,
                    self.log_dir / 'a2v' / self.episode_id / review_id / str(index))
            private.append((plan, continuity))
            recorded = dict(candidate_id=index, actions=plan.tolist(),
                            action_hash=action_hash(plan), forecast=media)
            visible = dict(recorded)
            if not reveal_actions:
                visible.pop('actions')
                visible.pop('action_hash')
                if media is not None:
                    visible['forecast'] = {k: v for k, v in media.items() if k != 'action_hash'}
            public.append(visible)
            stored.append(recorded)
        result = dict(review_id=review_id, episode_id=self.episode_id,
            decision_step=self.observation.step, instruction=self.instruction,
            predict_steps=self.horizon.predict_steps, execute_steps=self.block_size,
            control_hz=self.control_hz, latency_s=time.monotonic()-start, candidates=public,
            seed_offset=seed_offset, source='revised' if supplied is not None else 'sampled')
        # The audit record keeps the id-action-frame mapping. The returned object
        # omits action values when the caller is a visual reviewer.
        self._record(dict(event='candidate_preview', **{**result, 'candidates': stored}))
        self._review = ({**result, 'candidates': stored}, private)
        return copy.deepcopy(result)

    def framework_predict(self, request):
        """Read-only A2V RPC. Does not select, commit, discard or replace pending actions."""
        from thinker_wm.models.inference_api import action_hash
        from tools.prediction.a2v import forecast_final_plan
        self._at_boundary()
        if isinstance(self, Mode3ReviewMixin):
            raise ValueError('independent prediction tools require a generic review service')
        if not isinstance(request, dict) or set(request) != {'decision_step','plans','candidate_ids'}:
            raise ValueError('prediction requires observation step, final plans and candidate IDs')
        if type(request['decision_step']) is not int or request['decision_step'] != self.observation.step:
            raise ValueError('stale prediction observation')
        plans, ids = request['plans'], request['candidate_ids']
        if (not isinstance(plans,list) or not isinstance(ids,list) or not 1<=len(plans)<=8
                or len(ids)!=len(plans) or any(type(i) is not int or i<0 for i in ids)
                or len(set(ids))!=len(ids)):
            raise ValueError('invalid prediction candidates')
        checked=[]
        for plan in plans:
            plan=np.asarray(plan,dtype=np.float32)
            if (plan.shape!=(self.horizon.predict_steps,14) or not np.isfinite(plan).all()
                    or ((plan[:,[6,13]]<0)|(plan[:,[6,13]]>1)).any()):
                raise ValueError('tool requires full final native plans; no padding or smoothing')
            checked.append(plan.copy())
        request_id=uuid4().hex
        forecasts=[]
        for cid,plan in zip(ids,checked):
            media=forecast_final_plan(self,plan,self.log_dir/'a2v-tools'/self.episode_id/request_id/str(cid))
            forecasts.append(dict(candidate_id=cid,action_hash=action_hash(plan),forecast=media))
        result=dict(decision_step=self.observation.step,candidates=forecasts)
        self._record(dict(event='tool_prediction',request_id=request_id,**result))
        return result

    def select_candidate(self, request):
        return self._select_candidate_prefix(request, self.block_size)

    def _select_candidate_prefix(self, request, execute_steps):
        from XPolicyLab.utils.process_data import unpack_robot_state
        self._at_boundary()
        if not isinstance(request, dict) or set(request) != {'review_id', 'candidate_id'}:
            raise ValueError('selection requires review_id and candidate_id')
        if self._review is None or request['review_id'] != self._review[0]['review_id']:
            raise ValueError('unknown, stale or already consumed review')
        result, candidates = self._review
        index = request['candidate_id']
        if type(index) is not int or not 0 <= index < len(candidates):
            raise ValueError('invalid candidate_id')
        if result['episode_id'] != self.episode_id or result['decision_step'] != self.observation.step:
            raise ValueError('review does not match current observation')
        plan, continuity = candidates[index]
        commanded = plan[:execute_steps].copy()
        packed = unpack_robot_state(commanded, 'joint', self.dims, source_type='obs')
        self._record(dict(event='candidate_selected', review_id=result['review_id'],
            candidate_id=index, step=self.observation.step, execute_steps=execute_steps,
            commanded=commanded.tolist(),
            full_action_hash=result['candidates'][index]['action_hash']))
        self.last_command = (self.observation.step, commanded)
        self.continuity = continuity
        self.next_decision += execute_steps
        self._review = None
        return packed

    def discard_candidates(self, request):
        if (not isinstance(request, dict) or set(request) != {'review_id'}
                or self._review is None or request['review_id'] != self._review[0]['review_id']):
            raise ValueError('unknown review')
        self._record(dict(event='candidate_discarded', review_id=request['review_id']))
        self._review = None
        return {'discarded': True}


class Mode3ReviewMixin(A2VReviewMixin):
    """Dedicated strict server entry point; generic/mode-two behavior stays separate."""

    def __init__(self, usr_args):
        super().__init__(usr_args)
        self._validate_mode3()

    def _validate_mode3(self):
        if (self.horizon.predict_steps != 24 or self.block_size not in (4, 8, 16, 24)
                or self.control_hz != 25 or self.inf.num_frames != 12 or self.inf.ratio != 2):
            raise ValueError('mode 3 requires 24 actions, 4/8/16/24 execution, 12 frames, ratio 2, 25 Hz')
        if self.continuity is not None:
            raise ValueError('adaptive mode 3 requires continuity disabled')
        if not self.inf.settings.return_depth:
            raise ValueError('mode 3 must request available predicted depth')

    def preview_candidates(self, request):
        self._validate_mode3()
        if (not isinstance(request, dict) or set(request) != {'count', 'predict', 'reveal_actions'}
                or type(request['count']) is not int or request['count'] != 3
                or request['predict'] is not True or request['reveal_actions'] is not False):
            raise ValueError('mode 3 requires count=3, predict=True, reveal_actions=False')
        result = super().preview_candidates(request)
        try:
            # Validate server-generated timing/depth before exposing a selectable review.
            packet = mode3_astra_input(result, images=['validation-only'], history=[],
                                       instruction=self.instruction)
            model_config = getattr(getattr(self.inf, 'model', None), 'config', None)
            if getattr(model_config, 'enable_depth', False) and not packet['depth_available']:
                raise ValueError('depth-enabled model returned no decoded future depth')
        except Exception:
            self.discard_candidates({'review_id': result['review_id']})
            raise
        return result

    def select_candidate(self, request):
        self._validate_mode3()
        if (not isinstance(request, dict)
                or set(request) != {'review_id', 'candidate_id', 'execute_steps'}
                or type(request['execute_steps']) is not int
                or request['execute_steps'] not in (4, 8, 16, 24)):
            raise ValueError('mode 3 selection requires candidate_id and execute_steps=4/8/16/24')
        return self._select_candidate_prefix(
            {key: request[key] for key in ('review_id', 'candidate_id')}, request['execute_steps'])


def eval_reviewed_episode(task_env, client, reviewer, *, count=1, predict=True, root=None, max_steps=800):
    """Compatibility entry: all candidate review executes through TaskSkillRunner."""
    import os
    from tempfile import mkdtemp
    from environments.robodojo.framework import run_framework_episode
    root = root or os.environ.get('RSI_LIVE_RUN_DIR') or mkdtemp(prefix='rsi-reviewed-')
    return run_framework_episode(task_env, root=root, client=client, reviewer=reviewer,
        mode3=False, predict=predict, candidate_count=count, max_steps=max_steps)
