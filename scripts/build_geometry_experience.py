"""Seal geometry interfaces, notes, source and tests using existing experience format.

This creates a candidate resource package, never a task mapping or release.
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experience.packages import build_package

FILES=('tools/perception/interface.md','tools/perception/notes.md','tools/perception/geometry.py',
       'tools/perception/tools.py','tools/perception/direct_target.py','tools/measurement.py',
       'tools/perception/camera.py','scripts/measure_geometry.py','scripts/test_geometry_perception.py')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--package-id',required=True)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        for name in FILES:
            target=Path(temp)/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
        package=build_package(temp,args.output,args.package_id,metadata=dict(
            source='ThinkerWAM calibrated RGB tools',method_revision='calibrated-rgb/v1',
            scope='algorithm contracts and analytic fixtures; no robot success claims',
            publication_status='candidate',code_usage='reference only; runtime registry owns execution'))
    print(json.dumps(package.identity(),sort_keys=True))


if __name__=='__main__':
    main()
