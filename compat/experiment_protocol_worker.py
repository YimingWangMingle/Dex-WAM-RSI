"""Subprocess half of compare_experiment_protocol.py; intentionally no live I/O."""
import argparse,ast,copy,hashlib,json,os,re,sys,types
from pathlib import Path
from unittest.mock import patch

parser=argparse.ArgumentParser();parser.add_argument('--code',type=Path,required=True)
parser.add_argument('--launchers',type=Path,required=True);parser.add_argument('--side',required=True)
parser.add_argument('--group',required=True);parser.add_argument('--scenario',required=True)
parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(args.code))
if args.side == 'new':
    from compat import enable
    enable()
else:
    # Frozen runtime snapshot omits model sources. As before, both sides use
    # the delivered model contracts; do not alias any reference rsi/runtime code.
    from compat.imports import install_legacy_aliases
    install_legacy_aliases({name: 'thinker_wm.' + name
                           for name in ('models', 'data_prepare', 'utils')})
import numpy as np
import torch
from types import SimpleNamespace as NS
from models.inference_api import GenerationSettings,Forecast,action_hash
from runtime.robodojo_a2v import A2VReviewMixin
from runtime.roboprobe.policy.RoboDojo_Agent_L3_Inspect.types import ActionSpace
from rsi.robodojo_framework import run_framework_episode,flat
from rsi.robodojo_training import NATIVE_ACTION_CONTRACT

# Same public native-shaped fixtures, independent of either implementation.
class Inference:
    settings=GenerationSettings(seed=12,return_depth=True)
    checkpoint='fixture-only'
    def generate_action(self,*_):return NS(actions=np.full((24,14),.1+self.settings.seed/1000,np.float32))
    def predict_outcome(self,obs,history,instruction,plan):
        assert self.settings.decode_outputs
        return Forecast('action_conditioned',obs.step,obs.step/25,plan.copy(),action_hash(plan),
            'stats','camera','fixture-only',{},[obs.step+2*i for i in range(1,13)],[obs.step+24],
            frames=torch.full((1,3,12,64,64),float(plan[0,0])),
            depth=torch.full((1,1,12,64,64),.5),depth_content_mask=torch.ones((64,64)))

class Base:
    def reset(self):self.episode_id='reset-episode';self.next_decision=0;self.observation=None
    def finish_episode(self):pass

class Policy(A2VReviewMixin,Base):
    def __init__(self,root):
        self._review=None;self.inf=Inference();self.observation=None;self.next_decision=0
        self.horizon=NS(predict_steps=24);self.block_size=8;self.control_hz=25
        self.log_dir=Path(root);self.episode_id='episode';self.instruction='stack bowls'
        self.history=NS(last_step=0);self.continuity=self.async_replan=self.measured_velocity=self.last_command=self.dims=None
        self.profile=NS(action_contract=NATIVE_ACTION_CONTRACT,action_dim=14)
        self.events=[];self._record=self.events.append;self.measured=[]
    def update_obs(self,raw):
        step=raw['rsi_control_step'];expected=0 if self.observation is None else self.observation.step+1
        if step!=expected or step>self.next_decision:raise ValueError('bad measured row')
        self.observation=NS(step=step,state=torch.tensor(flat(raw['state'])))
        self.history=NS(last_step=step);self.measured.append(step)

class Env:
    task_name='stack_bowls'
    def __init__(self):
        self.steps=0;self.success=[False];self.end_flag=[False];self.actions=[]
        self.state=dict(ActionSpace().decode([0.]*6+[.5]+[0.]*6+[.5]).data)
        self.robot_manager=NS(robot_list=[NS(arm_name=a+'_arm',type='target') for a in ('left','right')])
    def get_obs(self):
        return dict(instruction='stack bowls',state=self.state,
            vision={k:{'color':np.full((8,9,3),self.steps,np.uint8)} for k in ('cam_head','cam_left_wrist','cam_right_wrist')},
            additional_info=dict(frequency=25))
    def take_action(self,action):
        self.actions.append(flat(action));self.state=action;self.steps+=1;self.end_flag[0]=self.steps>=10
    def is_episode_end(self):return self.steps>=10

rpc=[];inference=[];packets=[];prompts=[];decisions=[]
class Client:
    def __init__(self,p):self.p=p
    def call(self,name,payload=None):
        rpc.append(dict(method=name,payload=copy.deepcopy(payload)))
        result=getattr(self.p,name)() if payload is None else getattr(self.p,name)(payload)
        if args.scenario=='bad_hash' and name=='preview_candidates':
            result['candidates'][0]['forecast']['action_hash']='corrupted'
        return result

class Pi:
    def describe(self):
        from rsi.pi05_backend import CONFIG,CONTRACT
        return dict(config=CONFIG,checkpoint='/pi05-checkpoint',action_horizon=50,output_dim=14,internal_dim=32,
            normalization='quantile_q01_q99',internal_arm_actions='relative_to_current_state',
            output_actions='absolute_joint14_continuous_grippers')
    def infer(self,**kw):
        inference.append(dict(seed=kw['seed'],state=kw['state'],instruction=kw['instruction'],
            image_hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in kw['images'].items()}))
        a=np.full((50,14),.1+(kw['seed']%1000)*.0001,np.float32);a[:,6]=1.1;a[:,13]=-.1
        return a

class FK:
    def __init__(self,*args):pass
    def flange_world(self,row):return [float(row[0]),float(row[1]),float(row[2]),1.,0.,0.,0.]

# Parse the frozen launcher, execute only bridge_episode, never its top-level launch code.
source=(args.launchers/'run_episode.py').read_text();tree=ast.parse(source)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='bridge_episode')
legacy_bridge=ast.get_source_segment(source,fn)
worker_tree=ast.parse((args.launchers/'review_worker.py').read_text())
worker_nodes=list(ast.walk(worker_tree))
schema_node=next(n for n in worker_nodes if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='schema' for t in n.targets))
prompt_node=next(n for n in worker_nodes if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prompt' for t in n.targets))
metric_node=next(n for n in worker_nodes if isinstance(n,ast.If) and ast.unparse(n.test)=="group == 'D6M'")
optional_node=next(n for n in worker_nodes if isinstance(n,ast.If) and ast.unparse(n.test)=="packet.get('prediction_request_allowed')" and any(isinstance(x,ast.AugAssign) for x in n.body))
completion=(args.launchers/'completion-requirements.txt').read_text()

def original_payload(packet):
    packet=copy.deepcopy(packet);packet['attachment_order']=[{k:v for k,v in i.items() if k!='path'} for i in packet['images']]
    ns=dict(packet=packet,group=args.group,json=json)
    for node in (schema_node,prompt_node,metric_node,optional_node):
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<original-review-worker>','exec'),ns)
    return ns['schema'],ns['prompt']+'\n'+completion,packet['images']

# Import the new pure protocol by explicit file, without changing either rsi/runtime package.
import importlib.util
spec=importlib.util.spec_from_file_location('new_protocol',ROOT/'environments/robodojo/experiments/review_protocol.py')
protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)

def exchange(packet):
    packets.append(copy.deepcopy(packet))
    expected=original_payload(packet)
    actual=protocol.reviewer_payload(packet,args.group)
    assert actual==expected,'reviewer schema/prompt/attachment ordering differs from frozen worker'
    prompts.append(dict(schema=actual[0],prompt=actual[1],attachments=actual[2]))
    n=len(packets);scenario=args.scenario
    op='execute';cid=0;steps=7;edits=[]
    if scenario=='stop' or scenario=='budget':op='stop' if scenario=='stop' else 'resample';cid=None;steps=0
    elif scenario=='resample' and n==1:op='resample';cid=None;steps=0
    elif scenario=='edit' and n==1 or scenario=='predict_then_edit' and n==2:
        op='edit';steps=0;edits=[dict(start=0,end=7,channel=6,value=.75)]
    elif scenario in ('request_prediction','predict_then_edit') and packet['prediction_request_allowed']:
        op='request_prediction';cid=None;steps=0
    result=dict(operation=op,candidate_id=cid,execute_steps=steps,edits=edits,reason='deterministic comparison')
    decisions.append(result)
    return result

args.output.mkdir(parents=True)
mode={'V0':'pi05_action3','W0':'wam_medium_action3','W1':'wam_a2v'}.get(args.group,'pi05_a2v')
os.environ.update(RSI_MATRIX_MODE=mode,RSI_LIVE_RUN_DIR=str(args.output/'run'),RSI_EPISODE_STEPS='32',
    RSI_PI05_POOL_SIZE='8' if args.group=='S83' else '3',RSI_A2V_RGB_FRAMES='6',
    RSI_A2V_DEPTH='1' if args.group in ('V1','D6M') else '0',RSI_A2V_METRIC_DEPTH='1' if args.group=='D6M' else '0',
    RSI_A2V_DEFER='1' if args.group=='ADAPT' else '0',RSI_PI05_URL='http://invalid.test')
policy=Policy(args.output/'service');env=Env();client=Client(policy)
# The fixture's no_checks configuration matches the frozen experiment's choice.
def runner(*a,**kw):
    if args.scenario=='budget':kw['max_model_calls']=3
    if args.side=='new':kw['review_protocol']='experiment-v3'
    return run_framework_episode(*a,**kw)

if args.side=='reference':
    # Replace only the file transport with the same direct reviewer callback.
    start=legacy_bridge.index("            path=Path(os.environ['RSI_LIVE_RUN_DIR'])")
    end=legacy_bridge.index('        def visual(packet):',start)
    legacy_bridge=legacy_bridge[:start]+'            return exchange(packet)\n'+legacy_bridge[end:]
    ns=dict(os=os,json=json,Path=Path,time=__import__('time'),uuid=__import__('uuid'),exchange=exchange,
        framework_deploy=NS(_action_spec=lambda _:None),run_framework_episode=runner)
    exec(compile(legacy_bridge,'<reference-bridge>','exec'),ns)
    bridge=ns['bridge_episode']
else:
    from adapters.robodojo.experiments import layout3
    layout3.run_framework_episode=runner
    bridge=lambda env,client:layout3.bridge_episode(env,client,action_spec=NS(max_step=None),exchange=exchange,pi05_client=Pi())
    # No native bounds are applied by either side; no_checks covers both skills.
    original=bridge
    layout3.bridge_episode.__globals__['run_framework_episode']=lambda *a,**kw:runner(*a,**dict(kw,action_spec=None))

module=types.ModuleType('XPolicyLab.utils.process_data');module.unpack_robot_state=lambda a,*args,**kw:a.copy()
error=None
with patch.dict(sys.modules,{'XPolicyLab.utils.process_data':module}), patch('rsi.pi05_backend.PI05Client',lambda *a,**kw:Pi()), patch('runtime.robodojo_motion_plan.NativeMotionPlanner',FK):
    try:bridge(env,client)
    except Exception as exc:
        if args.scenario!='bad_hash':raise
        error=dict(type=type(exc).__name__,message=str(exc))

summary=json.loads((args.output/'run/episode-summary.json').read_text())
events=[json.loads(s) for s in (args.output/'run/framework-events.jsonl').read_text().splitlines()]
selected_events=[]
for e in events:
    if e.get('event')=='execution':
        selected_events.append({k:e[k] for k in ('event','skill','approved_actions','executed_actions','stop_reason')})
    if e.get('event')=='pi05_pool_selection':
        selected_events.append({k:v for k,v in e.items() if k not in ('seq','wall_time','pool_inference_and_selection_s','episode_id')})

def normalize(value):
    if isinstance(value,np.ndarray):return normalize(value.tolist())
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:normalize(v) for k,v in value.items() if k not in ('latency_s','request_id')}
    if isinstance(value,list):return [normalize(v) for v in value]
    if isinstance(value,str):
        value=value.replace(str(args.output),'<OUTPUT>')
        return re.sub(r'(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])','<ID>',value)
    return value
result=normalize(dict(rpc=rpc,inference=inference,packets=packets,prompts=prompts,decisions=decisions,
    actions=env.actions,events=selected_events,measured=policy.measured,error=error,
    budget=summary['model_budget'],stop_reason=summary['stop_reason'],executed_steps=summary['executed_steps']))
(args.output/'comparison.json').write_text(json.dumps(result,sort_keys=True,indent=2))
