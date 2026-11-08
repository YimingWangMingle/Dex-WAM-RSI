"""Check the project's native experience workspace and actual multimodal request."""
from __future__ import annotations

import argparse
from copy import deepcopy
from html import escape
import json
from pathlib import Path

from .prompt_builder import open_icl
from .runtime.messages import build_messages
from .runtime.render import ContextMemory


class CaptureClient:
    last_metadata = {'mode': 'offline_request_capture'}

    def complete(self, messages, tools):
        self.messages, self.tools = deepcopy(messages), deepcopy(tools)
        return {'inference_performed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='click_mouse')
    parser.add_argument('--catalog', type=Path, help='catalog.json for the shared ICL release')
    parser.add_argument('--output', type=Path, required=True, help='new directory outside the experience packages')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with open_icl(task=args.task, episode_dir=args.output / 'icl-workspace', catalog_path=args.catalog) as icl:
        record = json.loads(icl.workspace.read_text('pool/records.jsonl').splitlines()[0])
        step = record['steps'][0]
        obs = dict(task=args.task, episode_id='offline-source-copy', scene_id='offline-source-copy',
            instruction=record['instruction'], action_contract=record['action_contract'], state=step['state'],
            images={camera: icl.workspace.read_image('pool/' + name)[0]
                    for camera, name in step['images'].items()})
        current_messages = build_messages(obs, ContextMemory('', (), (), {}), system_prompt=(
            'Use the measured observation and task references under the declared control contract. '
            'This request is captured locally from a training observation copy; no model or robot is run.'))
        raw = CaptureClient()
        client = icl.wrap_client(raw, lambda: obs)
        result = client.complete(current_messages, [])
        assert result == {'inference_performed': False}
        context = client.last_context
        request = dict(messages=raw.messages, tools=raw.tools)
        (args.output / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding='utf-8')
        (args.output / 'icl_text.txt').write_text(context.text, encoding='utf-8')
        report = dict(task=args.task, native_workspace_identity=icl.workspace.identity(),
                      source_episode=record['episode_id'], source_step=step['step'],
                      observation_kind='offline_training_source_copy', inference_performed=False,
                      historical_images=len(context.images), current_images=len(obs['images']),
                      selected=list(context.hits), audit=context.audit)
        (args.output / 'selection.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        rendered, text_parts = [], []
        for message in raw.messages:
            rendered.append('<section><h2>' + escape(message['role']) + '</h2>')
            text_parts.append('ROLE: ' + message['role'])
            blocks = message['content']
            if isinstance(blocks, str):
                blocks = [dict(type='text', text=blocks)]
            for block in blocks:
                if block['type'] == 'text':
                    rendered.append('<pre>' + escape(block['text']) + '</pre>')
                    text_parts.append(block['text'])
                elif block['type'] == 'image_url':
                    url = block['image_url']['url']
                    assert url.startswith('data:image/')
                    rendered.append('<img alt="实际请求图片" src="' + escape(url, quote=True) + '">')
                    text_parts.append('[INLINE IMAGE: full PNG data is in request.json and preview.html]')
            rendered.append('</section>')
        page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>项目 ICL 输入预览</title>
<style>body{font:16px system-ui;max-width:1100px;margin:24px auto;padding:20px}
pre{white-space:pre-wrap;overflow-wrap:anywhere}section{border-top:1px solid #ccc;margin-top:24px}
img{max-width:min(100%,512px);margin:12px 0}</style><h1>项目 ICL 输入预览</h1>
<p>由项目原生 catalog 和冻结工作区读取；本次仅捕获模型输入，没有执行推理或机器人动作。</p>'''
        page += '<p>历史示范按 BEFORE / AFTER 成对展示，区间命令与时长由完整源轨迹核对。当前观测标为 CURRENT。</p>'
        (args.output / 'preview.html').write_text(page + '\n'.join(rendered) + '</html>', encoding='utf-8')
        (args.output / 'model_text.txt').write_text('\n\n'.join(text_parts), encoding='utf-8')
    print(json.dumps(dict(task=args.task, output=str(args.output.resolve()),
                         historical_images=report['historical_images'], current_images=report['current_images'],
                         inference_performed=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
