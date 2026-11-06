"""Read-only API preflight using a recorded native observation; no execution."""
import base64
import getpass
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.deepseek import DeepSeekClient
from environments.robodojo.runtime.robodojo_deepseek_ik import SYSTEM, specs, CAMERAS
p=Path(sys.argv[1])
content=[dict(type='text',text=(p/'state.json').read_text())]
for camera in CAMERAS:
    content.extend([dict(type='text',text='CURRENT '+camera),dict(type='image_url',image_url=dict(
        url='data:image/png;base64,'+base64.b64encode((p/(camera+'.png')).read_bytes()).decode(),detail='original'))])
client=DeepSeekClient(getpass.getpass('DeepSeek credential (memory only): '),thinking=False,max_tokens=4096,timeout=120)
response=client.complete([dict(role='system',content=SYSTEM),dict(role='user',content=content)],specs())
message=response['choices'][0]['message']
result=dict(status='API observation-only preflight; zero execution',metadata=client.last_metadata,
            finish_reason=response['choices'][0].get('finish_reason'),content=message.get('content'),tool_calls=message.get('tool_calls'))
Path(sys.argv[2]).write_text(json.dumps(result,indent=2,ensure_ascii=False))
print(json.dumps(result,ensure_ascii=False))
