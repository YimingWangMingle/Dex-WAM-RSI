"""Layout3 matrix compatibility bridge. Never starts a simulator or model service.

Use a separate run directory and caller-owned services. The fixed experiment
protocol does not opt into SelectionPolicy or independent TaskTool orchestration.
"""
import json
import os
from pathlib import Path
import time
import uuid
from contextlib import contextmanager
from environments.robodojo.framework import run_framework_episode

GROUPS = {
    'P0':'pi05_baseline', 'V0':'pi05_action3', 'V1':'pi05_a2v',
    'R6':'pi05_a2v', 'ADAPT':'pi05_a2v', 'D6M':'pi05_a2v', 'S83':'pi05_a2v',
    'W0':'wam_medium_action3', 'W1':'wam_a2v', 'WBASE':'wam_baseline',
}


def group_environment(group: str, root, max_steps: int = 800) -> dict:
    if group not in GROUPS or type(max_steps) is not int or max_steps < 1:
        raise ValueError('known experiment group and positive step limit required')
    return dict(RSI_MATRIX_MODE=GROUPS[group], RSI_EPISODE_STEPS=str(max_steps),
        RSI_LIVE_RUN_DIR=str(Path(root).resolve()), RSI_PI05_POOL_SIZE='8' if group=='S83' else '1' if group=='P0' else '3',
        RSI_A2V_RGB_FRAMES='6', RSI_A2V_DEPTH='1' if group in ('V1','D6M') else '0',
        RSI_A2V_METRIC_DEPTH='1' if group=='D6M' else '0', RSI_A2V_DEFER='1' if group=='ADAPT' else '0')


@contextmanager
def experiment_environment(group: str, root, max_steps: int = 800):
    """One episode per process: inherited experiment flags are process-global."""
    values=group_environment(group,root,max_steps)
    previous={k:os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k,v in previous.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v


def bridge_episode(TASK_ENV, model_client, *, action_spec=None, exchange=None, pi05_client=None):
    import numpy as np
    from environments.robodojo.framework import flat
    if action_spec is None:
        from third_party.RoboProbe.policy.RoboDojo_Agent_L3_Inspect import deploy
        action_spec = deploy._action_spec(TASK_ENV)
    spec = action_spec
    mode=os.environ['RSI_MATRIX_MODE']
    from execution.review import no_checks
    # WAM emits native joint targets. Do not apply EEF path-discretization
    # heuristics as an additional rejection gate for this action expert.
    opts=dict(review_protocol="experiment-v3",max_model_calls=60,action_spec=spec,max_steps=int(os.environ.get('RSI_EPISODE_STEPS','800')),row_checkers={'vla_control': no_checks,'wam_control': no_checks})
    if mode=='pi05_baseline':
        from skills.pi05.backend import PI05Client
        opts.update(pi05_client=pi05_client or PI05Client(os.environ['RSI_PI05_URL'],checkpoint='/pi05-checkpoint'),pi05_candidates=1,pi05_seed=20261102)
    elif mode=='wam_baseline': opts.update(client=model_client,mode3=False,predict=False,candidate_count=1)
    elif mode in ('pi05_action3','pi05_a2v','wam_medium_action3','wam_a2v'):
        review_runtime={}
        def send_review(raw, phase):
            from environments.robodojo.review_context import build_review_packet
            packet=build_review_packet(raw,os.environ['RSI_LIVE_RUN_DIR'])
            packet['prediction_request_allowed']=phase=='optional'
            packet['review_phase']=phase
            packet['request_id']=uuid.uuid4().hex
            if exchange is not None:
                return exchange(packet)
            path=Path(os.environ['RSI_LIVE_RUN_DIR'])/('review-%s-%s.json'%(packet.get('decision_step',0),packet['request_id']))
            temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(packet));temporary.replace(path)
            response=path.with_suffix('.response.json');deadline=time.monotonic()+600
            while not response.exists():
                if time.monotonic()>deadline:raise TimeoutError('review timed out')
                time.sleep(1)
            reply=json.loads(response.read_text())
            if reply.get('request_id')!=packet['request_id']:raise ValueError('stale review response')
            return reply['result']
        def visual(packet):
            adaptive=os.environ.get('RSI_A2V_DEFER')=='1'
            choice=send_review(packet,'optional' if adaptive else 'prediction')
            if choice.get('operation')=='request_prediction':
                if not adaptive or choice.get('candidate_id') is not None or choice.get('execute_steps')!=0 or choice.get('edits'):
                    raise ValueError('invalid prediction request')
                review_runtime['budget'].charge('vla_prediction_review')
                predicted=review_runtime['backend'].predict_for_review(packet)
                with (Path(os.environ['RSI_LIVE_RUN_DIR'])/'prediction-requests.jsonl').open('a') as f:
                    f.write(json.dumps(dict(step=packet['decision_step'],reason=choice['reason'],action_hashes=[c['action_hash'] for c in packet['candidates']]))+'\n')
                choice=send_review(predicted,'prediction')
                if choice.get('operation')=='request_prediction':raise ValueError('recursive prediction request')
            return choice
        visual.bind_runtime=lambda backend,budget:review_runtime.update(backend=backend,budget=budget)
        if mode.startswith('pi05'):
            from skills.pi05.backend import PI05Client
            opts.update(pi05_client=pi05_client or PI05Client(os.environ['RSI_PI05_URL'],checkpoint='/pi05-checkpoint'),
                        pi05_candidates=3,pi05_seed=20261102,vla_reviewer=visual)
            if mode=='pi05_action3': opts['vla_review_mode']='action_only'
            else: opts['pi05_a2v_client']=model_client
        elif mode=='wam_medium_action3':
            opts.update(client=model_client,wam_review_mode='action_only',candidate_count=3,reviewer=visual)
        else:
            opts.update(client=model_client,mode3=False,predict=True,candidate_count=3,reviewer=visual)
    else:raise ValueError(mode)
    # Measure native execution only; observation/log export happens outside it.
    original_take_action = TASK_ENV.take_action
    def timed_take_action(*args, **kwargs):
        started=time.perf_counter()
        try:
            return original_take_action(*args, **kwargs)
        finally:
            elapsed=time.perf_counter()-started
            with (Path(os.environ['RSI_LIVE_RUN_DIR'])/'action-timing.jsonl').open('a') as f:
                f.write(json.dumps(dict(native_action_wall_s=elapsed))+'\n')
    TASK_ENV.take_action=timed_take_action
    result = run_framework_episode(TASK_ENV, root=os.environ['RSI_LIVE_RUN_DIR'], **opts)
    # Native exporter initializes success=True; a bounded unfinished smoke run
    # is not task success. Framework summary already retains the native state.
    for index, ended in enumerate(TASK_ENV.end_flag):
        if not ended:
            TASK_ENV.success[index] = False
            TASK_ENV.end_flag[index] = True
    return result


def run_experiment(env, *, group: str, root, model_client=None, pi05_client=None,
                   action_spec=None, exchange=None, max_steps: int = 800):
    """Run an already-reset native env. Caller owns env/service teardown.

    exchange(public_packet) returns the legacy review operation, or None uses
    the existing atomic request/response file bridge with a 600-second timeout.
    No supervisor, experience injection or outer RSI is enabled by this profile.
    """
    root=Path(root).resolve()
    if root.exists():raise FileExistsError('use a fresh experiment directory')
    with experiment_environment(group,root,max_steps):
        return bridge_episode(env,model_client,action_spec=action_spec,
                              exchange=exchange,pi05_client=pi05_client)
