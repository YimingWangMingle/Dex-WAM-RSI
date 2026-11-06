"""Run geometry tools on a recorded calibrated RGB bundle; no robot execution.

python scripts/measure_geometry.py --bundle DIR --requests requests.json
Requests are [{tool, arguments, optional save_as}]. Reference arguments may use
$alias to refer to an earlier successful result from this same observation.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.perception.tools import GeometryTools, load_bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--requests',required=True)
    args=parser.parse_args()
    bundle=load_bundle(args.bundle)
    tools=GeometryTools(lambda obs:bundle)
    aliases={}
    for request in json.loads(Path(args.requests).read_text()):
        parameters={k:(aliases[v[1:]] if isinstance(v,str) and v.startswith('$') else v)
                    for k,v in request.get('arguments',{}).items()}
        result=tools.measure(request['tool'],bundle.observation,parameters)
        if result.data['status']=='ok' and request.get('save_as'):
            if request['save_as'] in aliases:
                raise ValueError('measurement aliases must be unique')
            aliases[request['save_as']]=result.data['measurement_id']
        print(json.dumps(dict(operation=result.report.to_dict(),data=dict(result.data)),
                         ensure_ascii=False,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
