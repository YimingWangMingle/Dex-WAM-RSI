"""Interactive Codex controller using the pinned RoboProbe EEF runtime.

No external model API. The active conversation supplies one response per fresh
request, via an atomic file. Images/state are public RGB/proprioception only.
"""
import base64
import json
import os
from pathlib import Path
import time
import uuid
from dataclasses import replace


class FileController:
    def __init__(self, root, *, retain_policy_images=False):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.count=0;self.session=uuid.uuid4().hex
        self.retain_policy_images=retain_policy_images

    def complete(self,messages,tools):
        self.count+=1
        request_id=f'{self.session}-{self.count:05d}'
        directory=self.root/f'{self.count:05d}';directory.mkdir()
        readable=[];images=[]
        # Keep a complete text history; export only the latest observation's
        # image payloads. Earlier images already live in earlier requests.
        latest=max((i for i,m in enumerate(messages) if isinstance(m.get('content'),list)
                    and any(p.get('type')=='image_url' for p in m['content'])),default=-1)
        for index,message in enumerate(messages):
            content=message.get('content')
            if isinstance(content,list):
                parts=[]
                for part in content:
                    if part.get('type')=='image_url':
                        if self.retain_policy_images or index==latest:
                            data=part['image_url']['url'];header,payload=data.split(',',1)
                            if not header.startswith('data:image/'):raise ValueError('local_image_required')
                            path=directory/f'image-{len(images)}.jpg'
                            path.write_bytes(base64.b64decode(payload,validate=True));images.append(str(path))
                            parts.append({'type':'image_file','path':str(path)})
                        else:parts.append({'type':'text','text':'[image in earlier request]'})
                    else:parts.append(part)
                readable.append(dict(message,content=parts))
            else:readable.append(message)
        request=dict(request_id=request_id,sequence=self.count,controller='current_codex_conversation',
                     external_api_calls=0,messages=readable,tools=tools,images=images)
        (directory/'request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2))
        temporary=self.root/'pending.tmp';temporary.write_text(json.dumps({'request_id':request_id,'directory':str(directory),'sequence':self.count}));temporary.replace(self.root/'pending.json')
        print('CODEX_WAITING',request_id,str(directory),flush=True)
        response_path=directory/'response.json'
        while True:
            if response_path.exists():
                response=json.loads(response_path.read_text())
                if response.get('request_id')!=request_id:raise ValueError('response_request_id_mismatch')
                name=response['name'];arguments=response['arguments']
                if name not in {t['function']['name'] for t in tools}:raise ValueError('unknown_tool')
                if not isinstance(arguments,dict):raise ValueError('arguments_must_be_object')
                return {'id':request_id,'model':'current-codex-conversation',
                        'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[{
                            'id':'call-'+request_id,'type':'function','function':{'name':name,'arguments':json.dumps(arguments)}}]},'finish_reason':'tool_calls'}]}
            time.sleep(.2)


def controller_clients(root, env):
    """One conversation per API client; file mode retains the shared sequence."""
    if env.get('RSI_LLM_PROVIDER','file').strip().lower() == 'file':
        controller=FileController(Path(root)/'bridge',
            retain_policy_images=env.get('RSI_EVAL_PROTOCOL') == 'official_eef')
        return controller,controller
    from agents.llm_clients import client_from_env
    return client_from_env(env),client_from_env(env)


def official_eef_config(env, native_step_limit):
    """Pin public baseline controls; reject inherited ablation settings."""
    required = {
        'RSI_CONTROL_MODE': 'direct',
        'L3_INSPECT_MAX_LLM_CALLS': '170',
        'L3_INSPECT_USE_RECIPE': '1',
        'L3_INSPECT_KEEP_ALL_IMAGES': '0',
        'L3_INSPECT_IMAGE_HORIZON': '2',
        'L3_INSPECT_REASONING_EFFORT': 'medium',
        'L3_INSPECT_DEPTH': 'off',
        'L3_INSPECT_ACTION_TYPE': 'joint',
        'ROBODOJO_ACTION_TYPE': 'joint',
        'L3_INSPECT_DISABLE_GIVE_UP': '0',
    }
    if int(native_step_limit) < 1:
        raise ValueError('positive native task step limit required')
    required['RSI_MAX_CONTROL_STEPS'] = str(int(native_step_limit))
    for key, value in required.items():
        if key in env and str(env[key]) != value:
            raise ValueError(f'official_eef requires {key}={value}')
    for key in ('RSI_EXPERIENCE_CATALOG', 'RSI_BOWL_MODEL',
                'L3_INSPECT_ENV_STEP_LIMIT', 'ROBODOJO_ENABLE_METRIC_DEPTH'):
        if env.get(key):
            raise ValueError(f'official_eef forbids {key}')
    # Unknown L3 overrides include motion jitter, timing and task-specific ablations.
    allowed = set(required) | {'L3_INSPECT_TRACE_DIR', 'L3_INSPECT_TRACE_NAMESPACE',
                              'L3_INSPECT_PLANNER', 'L3_INSPECT_BASE_URL'}
    extra = sorted(k for k in env if k.startswith('L3_INSPECT_') and k not in allowed)
    if extra:
        raise ValueError('unreviewed official_eef overrides: '+', '.join(extra))
    return dict(env, **required)


def eval_one_episode(TASK_ENV,model_client):
    from third_party.RoboProbe.policy.RoboDojo_Agent_L3_Inspect import deploy
    from third_party.RoboProbe.policy.RoboDojo_Agent_L3_Inspect_EEF.deploy import _planner
    from third_party.RoboProbe.policy.RoboDojo_Agent_L3_Inspect_EEF.docs import eef_docs_from_joint_docs
    from third_party.RoboProbe.policy.RoboDojo_Agent_L3_Inspect_EEF.policy import EefAgentPolicy
    from environments.robodojo.framework import ContextClient, run_framework_episode
    root=Path(os.environ['RSI_LIVE_RUN_DIR'])
    protocol=os.environ.get('RSI_EVAL_PROTOCOL','framework')
    if protocol not in ('framework', 'official_eef'):
        raise ValueError('RSI_EVAL_PROTOCOL must be framework or official_eef')
    official=protocol == 'official_eef'
    env=official_eef_config(os.environ, TASK_ENV.step_lim) if official else dict(os.environ)
    controller,eef_client=controller_clients(root,env)
    client=eef_client if official else ContextClient(eef_client)
    mode=env.get('RSI_CONTROL_MODE','direct')
    if mode not in ('direct','hybrid'):raise ValueError('RSI_CONTROL_MODE must be direct or hybrid')
    spec=deploy._action_spec(TASK_ENV)
    cfg=dict(env,L3_INSPECT_PLANNER='astra',L3_INSPECT_BASE_URL='https://unused.invalid/v1')
    # Preserve the official EEF default and its native budget-exhaustion stop.
    cfg.setdefault('L3_INSPECT_MAX_LLM_CALLS', '170')
    policy=EefAgentPolicy(action_spec=replace(spec,docs=eef_docs_from_joint_docs(spec.docs)),
                          env=cfg,planner=_planner(TASK_ENV),client=client)
    root.mkdir(parents=True,exist_ok=True)
    (root/'evaluation-protocol.json').write_text(json.dumps(dict(
        protocol=protocol, official_replication_verified=False,
        model_identity_verified=False, provider=env.get('RSI_LLM_PROVIDER','file'),
        native_step_limit=int(TASK_ENV.step_lim), eef_call_limit=policy._max_llm_calls,
        supervisor=not official, framework_context=not official, geometry=not official,
        policy_settings={k:v for k,v in cfg.items() if k in (
            'L3_INSPECT_USE_RECIPE','L3_INSPECT_KEEP_ALL_IMAGES',
            'L3_INSPECT_IMAGE_HORIZON','L3_INSPECT_REASONING_EFFORT')},
        unresolved=['controller model/version and inference settings',
                    'framework/native execution and termination parity',
                    'official layout manifest and simulator/assets revision']),indent=2))
    perception=None
    if os.environ.get('RSI_BOWL_MODEL'):
        from environments.robodojo.runtime.robodojo_bowl_perception import BowlPerception
        perception=BowlPerception(os.environ['RSI_BOWL_MODEL'])
    return run_framework_episode(TASK_ENV,root=root,direct_policy=policy,context_client=None if official else client,
        max_steps=int(env.get('RSI_MAX_CONTROL_STEPS','800')),
        client=model_client if mode=='hybrid' else None, action_spec=spec, enable_geometry=not official,
        bowl_perception=perception,
        supervisor_client=None if official else controller, experience_catalog=env.get('RSI_EXPERIENCE_CATALOG'))
