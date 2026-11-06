"""Isolated CPU differential check against a read-only experiment snapshot.

Never connects to an RPC endpoint or imports a simulator. The same deterministic
model doubles and scripted reviewer replies exercise both actual implementations.
"""
import argparse
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-code',type=Path,required=True)
    parser.add_argument('--reference-launchers',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--groups',nargs='+',default=['V0','V1','R6','ADAPT','D6M','S83','W0','W1'])
    args=parser.parse_args()
    manifest=json.loads((ROOT/'environments/robodojo/experiments/source_manifest.json').read_text())
    for category,base in (('files',args.reference_code),('launchers',args.reference_launchers)):
        for name,expected in manifest[category].items():
            if hashlib.sha256((base/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('reference differs from frozen source: '+name)
    if args.output.exists():raise ValueError('use a fresh differential output directory')
    args.output.mkdir(parents=True)
    cases=[(g,s) for g in args.groups
           for s in ('execute','resample','edit','stop','budget')]
    cases += [(g,s) for g,s in [('ADAPT','request_prediction'),('ADAPT','predict_then_edit'),('R6','bad_hash')] if g in args.groups]
    results=[]
    for group,scenario in cases:
        outputs=[]
        for side,code in (('reference',args.reference_code),('new',ROOT)):
            out=args.output/f'{group}-{scenario}-{side}'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            cmd=[sys.executable,str(ROOT/'compat/experiment_protocol_worker.py'),
                 '--code',str(code.resolve()),'--launchers',str(args.reference_launchers.resolve()),
                 '--side',side,'--group',group,'--scenario',scenario,'--output',str(out.resolve())]
            result=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=90)
            if result.returncode:
                raise RuntimeError(f'{group}/{scenario}/{side}: {result.stdout}\n{result.stderr}')
            outputs.append(json.loads((out/'comparison.json').read_text()))
        equal=outputs[0]==outputs[1]
        results.append(dict(group=group,scenario=scenario,equal=equal))
        print(group,scenario,'MATCH' if equal else 'DIFFERENT',flush=True)
        (args.output/'report.json').write_text(json.dumps(dict(
            scope='CPU differential protocol validation; no simulator or live model calls',
            reference=str(args.reference_code.resolve()),cases=results),indent=2))
    return int(not all(r['equal'] for r in results))

if __name__=='__main__':raise SystemExit(main())
