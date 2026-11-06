"""CPU verification only; runs inside the independent A6000 source copy."""
from pathlib import Path
import subprocess,sys,json,os,argparse
root=Path.cwd();results=[]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output",type=Path,required=True)
output=parser.parse_args().output
output.parent.mkdir(parents=True,exist_ok=True)
commands=[
 ['-m','unittest','discover','-s','tests','-v'],
 ['scripts/test_framework_native.py'],
 ['scripts/test_framework_campaign.py'],
 ['scripts/test_model_budget.py'],
 ['-m','rsi','platforms'],
 ['-m','rsi','campaign','--help'],
 ['-c', 'import sys; import execution.episode, rsi.framework_campaign; assert "torch" not in sys.modules; print("core imports without torch or simulator")'],
 ['-c', 'import thinker_wm.runtime.robodojo_policy as d; from thinker_wm.models.inference_api import WAMInference; from thinker_wm.scripts.eval_video_ckpt import create_eval_model; from thinker_wm.utils.output_paths import repo_root; from pathlib import Path; assert repo_root()==Path.cwd(); assert "third_party/Thinker-WM" in d.__file__; print("model implementation imports and legacy output root OK")'],
]
for args in commands:
 p=subprocess.run([sys.executable,*args],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=120)
 results.append(dict(command=args,returncode=p.returncode,output=p.stdout+p.stderr))
 print(args[:3],p.returncode,flush=True)
output.write_text(json.dumps(results,indent=2))
raise SystemExit(any(x['returncode'] for x in results))
