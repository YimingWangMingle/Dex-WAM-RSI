"""Adapter for the audited RoboDojo PI0.5 HTTP/NPZ experiment service.

The service owns normalization/delta conversion; its output is absolute joint14.
This module proposes only. TaskSkillRunner owns review, dispatch and feedback.
"""
from __future__ import annotations
import copy
import io
import json
import math
import urllib.error
import urllib.parse
import urllib.request
import numpy as np
from skills.contracts import ActionCandidate, BackendProposal

from environments.robodojo.training import NATIVE_ACTION_CONTRACT as CONTRACT
CONFIG = 'pi05_base_aloha_full_sim_arx-x5_seed_0'
CAMERAS = {'cam_high':'cam_head','cam_left_wrist':'cam_left_wrist','cam_right_wrist':'cam_right_wrist'}


class PI05Error(RuntimeError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        return None


class PI05Client:
    """Existing GET identity + POST /infer NPZ protocol; no automatic retry/reset.

    The historical service lacks checkpoint digests and an action-contract field.
    Its exact configuration name and declared output semantics are therefore
    checked. The adapter's known 25Hz ARX-X5 contract is explicit, not negotiated.
    """
    def __init__(self, base_url: str, *, checkpoint: str, timeout: float = 900):
        url=urllib.parse.urlsplit(base_url)
        if url.scheme not in ('http','https') or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('explicit HTTP(S) PI0.5 base URL required')
        if not checkpoint or isinstance(timeout,bool) or not math.isfinite(timeout) or timeout<=0:
            raise ValueError('checkpoint and positive timeout required')
        self.url,self.checkpoint,self.timeout=base_url.rstrip('/'),checkpoint,timeout
        self._opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),_NoRedirect())
        self.identity=None

    def _request(self, path: str, data: bytes | None = None) -> bytes:
        req=urllib.request.Request(self.url+path,data=data,
            headers={'Content-Type':'application/octet-stream'} if data is not None else {})
        try:
            with self._opener.open(req,timeout=self.timeout) as response:
                body=response.read(12_000_001)
            if len(body)>12_000_000:raise PI05Error('oversized PI0.5 response')
            return body
        except urllib.error.HTTPError as error:
            code=error.code;error.close()
            raise PI05Error(f'PI0.5 HTTP {code}; not retried') from None
        except OSError as error:
            raise PI05Error('PI0.5 transport failure; not retried: '+type(error).__name__) from None

    def describe(self) -> dict:
        try:identity=json.loads(self._request('/'))
        except (ValueError,TypeError):raise PI05Error('invalid PI0.5 identity JSON') from None
        expected=dict(config=CONFIG,checkpoint=self.checkpoint,action_horizon=50,output_dim=14,
            internal_dim=32,normalization='quantile_q01_q99',
            internal_arm_actions='relative_to_current_state',output_actions='absolute_joint14_continuous_grippers')
        if not isinstance(identity,dict) or any(identity.get(k)!=v for k,v in expected.items()):
            raise PI05Error('PI0.5 identity/normalization/action semantics differ from the audited contract')
        self.identity=copy.deepcopy(identity)
        return copy.deepcopy(identity)

    def infer(self, *, state, images: dict, instruction: str, seed: int) -> np.ndarray:
        if self.identity is None:self.describe()
        state=np.asarray(state,dtype=np.float32)
        if state.shape!=(14,) or not np.isfinite(state).all():raise ValueError('finite native joint14 required')
        if type(seed) is not int or not 0<=seed<2**32:raise ValueError('uint32 request seed required')
        if not isinstance(instruction,str) or not instruction.strip():raise ValueError('task instruction required')
        values=dict(state=state,instruction=np.array(instruction),seed=np.array(seed,dtype=np.int64))
        for dst,src in CAMERAS.items():
            rgb=np.asarray(images[src])
            if rgb.ndim!=3 or rgb.shape[-1]!=3 or rgb.dtype!=np.uint8 or min(rgb.shape[:2])<1:
                raise ValueError('native HWC uint8 RGB required: '+src)
            values[dst]=rgb.transpose(2,0,1)
        buf=io.BytesIO();np.savez(buf,**values)
        if buf.tell()>12_000_000:raise ValueError('PI0.5 observation exceeds service limit')
        try:
            with np.load(io.BytesIO(self._request('/infer',buf.getvalue())),allow_pickle=False) as z:
                actual_seed=z['seed']
                if actual_seed.shape!=() or actual_seed.dtype.kind not in 'iu' or actual_seed.item()!=seed:
                    raise PI05Error('PI0.5 returned a mismatched seed')
                actions=np.asarray(z['actions'],dtype=np.float32).copy()
        except (ValueError,KeyError,OSError) as error:
            raise PI05Error('invalid PI0.5 NPZ response: '+type(error).__name__) from None
        if actions.shape!=(50,14) or not np.isfinite(actions).all():raise PI05Error('PI0.5 must return finite 50x14 absolute targets')
        return actions


class PI05Backend:
    def __init__(self, client, session, *, candidate_count=1, seed=0, a2v_client=None, sink=None, action_only=False):
        if candidate_count not in (1,3) or type(candidate_count) is not int:
            raise ValueError('PI0.5 candidate count must be 1 or 3')
        if type(seed) is not int or not 0<=seed<2**32:raise ValueError('uint32 inference seed required')
        if action_only and a2v_client is not None:raise ValueError("action-only mode forbids A2V")
        self.action_only=action_only
        import os
        self.defer_predictions=os.environ.get("RSI_A2V_DEFER")=="1"
        self.pool_size=int(os.environ.get("RSI_PI05_POOL_SIZE",candidate_count))
        if self.pool_size not in (candidate_count,8) or (self.pool_size==8 and candidate_count!=3):raise ValueError("invalid pool size")
        self.client,self.session,self.count,self.seed=client,session,candidate_count,seed
        self.identity=client.describe()
        self.sink=sink or (lambda event:None)
        self.world=None
        if a2v_client is not None:
            from environments.robodojo.framework import WAMCandidates
            self.world=WAMCandidates(a2v_client,session,mode3=False,predict=True)
            if self.world.identity['action_contract']!=CONTRACT:raise ValueError('PI0.5/A2V action contract mismatch')
        self.candidates=()
        self.selected=None
        self.before=None
        self.revision_plans=None
        self.recovery_feedback=None
        self.sampling_step=None
        self.sampling_round=0

    def reset(self, episode_id):
        if episode_id!=self.session.episode_id or self.candidates:raise ValueError('resolve PI0.5 proposal before handoff')
        # Each inference explicitly seeds RNG; do not reset shared remote service.
        self.client.describe()
        if self.world:self.world.reset(episode_id)

    def propose(self, request):
        from environments.robodojo.framework import flat
        if self.candidates:raise RuntimeError('PI0.5 proposal already pending')
        raw=self.session.raw(request.observation.control_step)
        if request.observation.control_step != self.sampling_step:
            self.sampling_step=request.observation.control_step
            self.sampling_round=0
            self.recovery_feedback=None
        plans=[]
        count=min(24,request.max_actions)
        pool_started=__import__("time").perf_counter()
        for cid in range(self.pool_size) if self.revision_plans is None else []:
            seed=self.seed+request.observation.control_step+cid+self.sampling_round*self.pool_size
            if seed>=2**32:raise ValueError('inference seed overflow')
            full=self.client.infer(state=flat(raw['state']),images={k:v['color'] for k,v in raw['vision'].items()},
                instruction=raw['instruction'],seed=seed)
            plan=np.asarray(full[:24],dtype=np.float32).copy()
            if np.asarray(full).shape!=(50,14) or not np.isfinite(full).all():raise ValueError('invalid PI0.5 client output')
            plan[:,[6,13]]=plan[:,[6,13]].clip(0,1)
            plans.append(plan)
            self.sink(dict(event='pi05_proposal',episode_id=self.session.episode_id,
                control_step=request.observation.control_step,candidate_id=cid,seed=seed,
                checkpoint=self.identity['checkpoint'],raw_actions=np.asarray(full).tolist(),
                final_plan=plan.tolist(),postprocess='clip_grippers_0_1_before_forecast'))
        if self.revision_plans is None and self.pool_size==8:
            from environments.robodojo.pi05_diversity import select_native
            ids,details=select_native(plans,self.session.env)
            from execution.trajectory import action_hash
            self.sink(dict(event='pi05_pool_selection',episode_id=self.session.episode_id,
                control_step=request.observation.control_step,original_indices=ids,
                seeds=[self.seed+request.observation.control_step+i+self.sampling_round*self.pool_size for i in range(self.pool_size)],
                action_hashes=[action_hash(p) for p in plans],selected_hashes=[action_hash(plans[i]) for i in ids],
                pool_inference_and_selection_s=__import__('time').perf_counter()-pool_started,**details))
            plans=[plans[i] for i in ids]
        if self.revision_plans is not None:
            plans=[np.asarray(p,dtype=np.float32) for p in self.revision_plans]
            self.revision_plans=None
        self.plans=[p.copy() for p in plans]
        self.sampling_round+=1
        self.offered_steps=count
        forecasts=[None]*self.count
        if self.world and not self.defer_predictions:
            self.world.sync_observation(request.observation)
            preview=self.world.client.call('preview_candidates',dict(plans=[p.tolist() for p in plans],predict=True,reveal_actions=True))
            self.world.preview=preview;self.world.selected=False
            returned=preview['candidates']
            if len(returned)!=self.count:raise ValueError('wrong A2V candidate count')
            from execution.trajectory import action_hash
            for cid,(candidate,plan) in enumerate(zip(returned,plans)):
                if candidate['candidate_id']!=cid or not np.array_equal(np.asarray(candidate['actions'],np.float32),plan):
                    raise ValueError('A2V changed PI0.5 actions')
                if candidate['action_hash'] != action_hash(plan) or candidate['forecast']['action_hash'] != action_hash(plan):
                    raise ValueError('PI0.5 A2V action hash mismatch')
                forecasts[cid]=candidate['forecast']
        self.before=request.observation
        self.candidates=tuple(ActionCandidate(cid,tuple(tuple(float(v) for v in row) for row in p[:count]),
            json.dumps(forecasts[cid],sort_keys=True) if forecasts[cid] is not None else None) for cid,p in enumerate(plans))
        return BackendProposal(self.candidates[0].actions,self.candidates[0].prediction_ref,candidates=self.candidates)

    def predict_for_review(self, packet):
        if self.world is None or self.before is None or self.world.preview is not None:
            raise RuntimeError('prediction request requires pending unpredicted candidates')
        if packet['decision_step']!=self.before.control_step:raise ValueError('stale prediction request')
        plans=self.plans
        forecasts=[None]*self.count
        if self.world:
            self.world.sync_observation(self.before)
            preview=self.world.client.call('preview_candidates',dict(plans=[p.tolist() for p in plans],predict=True,reveal_actions=True))
            self.world.preview=preview;self.world.selected=False
            returned=preview['candidates']
            if len(returned)!=self.count:raise ValueError('wrong A2V candidate count')
            from execution.trajectory import action_hash
            for cid,(candidate,plan) in enumerate(zip(returned,plans)):
                if candidate['candidate_id']!=cid or not np.array_equal(np.asarray(candidate['actions'],np.float32),plan):
                    raise ValueError('A2V changed PI0.5 actions')
                if candidate['action_hash'] != action_hash(plan) or candidate['forecast']['action_hash'] != action_hash(plan):
                    raise ValueError('PI0.5 A2V action hash mismatch')
                forecasts[cid]=candidate['forecast']
        result=copy.deepcopy(packet)
        for candidate,forecast in zip(result['candidates'],forecasts):
            candidate['forecast']=forecast
        result['a2v_enabled']=True
        return result

    def packet(self, ref):
        if ref!=self.before:raise ValueError('stale PI0.5 review')
        images,history=self.session.visual(ref)
        from execution.trajectory import action_hash
        packet = dict(actor='pi05',instruction=self.session.load(ref)['instruction'],
            observation_id=ref.observation_id,decision_step=ref.control_step,
            current_rgb=images,history=history,a2v_enabled=self.world is not None,
            candidates=[dict(candidate_id=c.candidate_id,action_hash=action_hash(self.plans[c.candidate_id]),actions=[list(r) for r in c.actions],
                forecast=json.loads(c.prediction_ref) if c.prediction_ref else None) for c in self.candidates],
            allowed_execute_steps=(list(range(1,len(self.candidates[0].actions)+1)) if getattr(self,'recovery_review',False) else [n for n in (24,16,8,4) if n<=len(self.candidates[0].actions)]),recovery_feedback=self.recovery_feedback)

        if self.action_only:
            packet['review_mode']='action_only'
            for candidate in packet['candidates']:
                candidate.pop('forecast',None)
        return packet

    def revise(self, choice):
        from environments.robodojo.action_revision import revised_plans
        if choice['operation']=='edit':
            self.revision_plans=revised_plans(
                [dict(candidate_id=i,actions=p.tolist()) for i,p in enumerate(self.plans)],
                choice['candidate_id'],choice['edits'])
        self.recovery_feedback=dict(operation=choice['operation'],reason=choice['reason'],
            edits=copy.deepcopy(choice['edits']),candidate_id=choice['candidate_id'],
            at_step=self.sampling_step,sampling_round=self.sampling_round,
            robot_moved=False,requires_fresh_review=True)

    def prepare_execution(self,candidate_id,rows):
        if type(candidate_id) is not int or not 0<=candidate_id<len(self.candidates):raise ValueError('unknown PI0.5 candidate')
        candidate=self.candidates[candidate_id]
        if not rows or tuple(rows)!=candidate.actions[:len(rows)]:raise ValueError('PI0.5 requires exact approved prefix')
        if self.world and self.world.preview is not None:self.world.prepare_execution(candidate_id,rows)
        self.selected=tuple(rows)

    def observe(self,feedback):
        if not self.candidates:raise ValueError('no pending PI0.5 proposal')
        rows=feedback.executed_actions
        if rows and (self.selected is None or tuple(rows)!=self.selected[:len(rows)]):raise ValueError('PI0.5 feedback differs from approval')
        if self.world:self.world.observe(feedback)
        self.sink(dict(event='pi05_feedback',episode_id=self.session.episode_id,
            executed_steps=len(rows),control_step=feedback.final_observation.control_step,
            unexecuted_steps=(len(self.selected)-len(rows)) if self.selected else 0))
        self.candidates=();self.selected=None;self.before=None
