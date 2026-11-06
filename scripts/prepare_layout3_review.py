"""Export the original reviewer schema/prompt and verify locally available images.

This prepares a single isolated review. It does not call a model or submit a
response to any experiment. Transfer images before using when paths are remote.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from environments.robodojo.experiments.review_protocol import reviewer_payload, REVIEWER_SETTINGS
from environments.robodojo.experiments.layout3 import GROUPS


def prepare(request: Path, output: Path, group: str):
    packet=json.loads(request.read_text())
    schema,prompt,attachments=reviewer_payload(packet,group)
    for image in attachments:
        if hashlib.sha256(Path(image['path']).read_bytes()).hexdigest()!=image['sha256']:
            raise ValueError('review attachment changed: '+image['path'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'review-settings.json').write_text(json.dumps(REVIEWER_SETTINGS,indent=2))
    (output/'schema.json').write_text(json.dumps(schema,indent=2))
    (output/'prompt.txt').write_text(prompt)
    (output/'attachments.json').write_text(json.dumps(attachments,indent=2))
    (output/'request.json').write_text(json.dumps(packet,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--group',choices=GROUPS,required=True)
    args=parser.parse_args();prepare(args.request,args.output,args.group)
