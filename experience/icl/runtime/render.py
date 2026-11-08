"""Evidence-grounded multimodal ICL: named controls and paired historical observations."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import re
from string import Template

from .contracts import contract_description, describe_action, describe_state, measured_change
from .memory_features import rgb_image
from .memory_pool import Hit, inside
from .sequence_evidence import interval_evidence
from .task_knowledge import TaskKnowledge


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def _render_knowledge(sections):
    groups = []
    guidance = [s['text'] for s in sections if s['section'] == 'guidance']
    if guidance:
        groups.append('Task-specific guidance\n' + '\n'.join(guidance))

    phases = []
    for section in sections:
        if section['section'] == 'phase':
            lines = [f"{section['phase_number']}. {section['name']} [phase_id={section['id']}]",
                     '   Do: ' + section['instruction']]
            lines.extend('   Observe/check: ' + item for item in section['observable_checks'])
            phases.append('\n'.join(lines))
    if phases:
        groups.append('Phase guide: choose from CURRENT visual evidence; retrieval does not label the phase.\n'
                      + '\n'.join(phases))
    recovery = []
    for section in sections:
        if section['section'] == 'recovery':
            recovery.append('If visible: ' + section['symptom'] + '\nThen: ' + section['response'])
    if recovery:
        groups.append('Conditional recovery: use only when its symptom is observed.\n' + '\n'.join(recovery))
    return '\n\n'.join(groups)


def _named_groups(groups):
    return '\n'.join(arm + '.' + key + '=' + _json(value)
                     for arm, fields in groups.items() for key, value in fields.items())


@dataclass(frozen=True)
class ContextMemory:
    text: str
    images: tuple[dict, ...]
    hits: tuple[dict, ...]
    audit: dict
    resources: tuple[dict, ...] = ()

    def to_content(self) -> list[dict]:
        if not self.text:
            return []
        content = [dict(type='text', text=self.text)]
        for item in self.images:
            content.extend([dict(type='text', text=item['label']),
                            dict(type='image_url', image_url=dict(url=item['data_url'], detail='auto'))])
        return content


class PromptBuilder:
    def __init__(self, *, max_chars=16000, max_images=8, image_side=384,
                 max_image_bytes=4 * 1024 * 1024, template_path=None, image_reader=None):
        if (any(type(x) is not int for x in (max_chars, max_images, image_side, max_image_bytes))
                or max_chars < 1000 or max_images < 0 or not 32 <= image_side <= 2048 or max_image_bytes < 1):
            raise ValueError('invalid prompt/image limits')
        self.max_chars, self.max_images = max_chars, max_images
        self.image_side, self.max_image_bytes = image_side, max_image_bytes
        self.image_reader = image_reader
        path = Path(template_path) if template_path else Path(__file__).resolve().parent.parent / 'icl_template.txt'
        self.template = Template(path.read_text(encoding='utf-8'))

    def build(self, obs: dict, hits: list[Hit], root: Path, *, audit: dict,
              knowledge: TaskKnowledge | None = None, max_examples: int | None = None) -> ContextMemory:
        if max_examples is not None and (type(max_examples) is not int or max_examples < 0):
            raise ValueError('max_examples must be None or a nonnegative integer')
        contract = obs['action_contract']
        control = contract_description(contract)
        current = 'Task: ' + obs['task'] + '\nGoal: ' + obs['instruction'] + '\nProtocol: ' + contract
        cards, images, selected, skipped = [], [], [], []
        sections, knowledge_skipped, knowledge_truncated, card_audits = [], [], [], []
        text_sources, record_sources = {}, {}

        def model_text(value):
            def substitute(match):
                url = match.group()
                if url not in text_sources:
                    text_sources[url] = f'R{len(text_sources) + 1}'
                return '[' + text_sources[url] + ']'
            return re.sub(r'https?://[^\s<>"\[\]]+', substitute, value)

        def render(items):
            return self.template.substitute(
                current=model_text(current), contract=control,
                knowledge=_render_knowledge(sections) or 'No task-specific guidance supplied.',
                examples='\n\n'.join(items) or 'No complete compatible example fits this request; do not invent one.')

        if len(render([])) > self.max_chars:
            raise ValueError('max_chars cannot fit the essential task/control instructions; increase the text budget')

        if knowledge is not None:
            if (knowledge.task, knowledge.action_contract) != (obs['task'], contract):
                raise ValueError('task knowledge differs from observation task/action contract')
            budget = self.max_chars if not hits else max(len(render([])), self.max_chars // 2)
            phase_number = 0
            for section in knowledge.context_sections():
                if section['section'] == 'reference':
                    continue  # No unused file index in the model input; retained in resources/audit.
                if section['section'] == 'phase':
                    phase_number += 1
                    section = dict(section, phase_number=phase_number)
                if section['section'] == 'guidance' and len(section['text']) > 2000:
                    section = dict(section, text=section['text'][:2000] + '\n[guidance shortened]', truncated=True)
                    knowledge_truncated.append('guidance')
                sections.append(section)
                if len(render([])) > budget:
                    sections.pop()
                    knowledge_skipped.append(dict(section=section['section'], id=section.get('id'), reason='text_budget'))

        for hit in hits:
            record = hit.record
            if (record.task, record.action_contract) != (obs['task'], contract):
                raise ValueError('retrieved example differs from current task/action contract')
            if max_examples is not None and len(cards) >= max_examples:
                skipped.append(dict(record_id=record.record_id, reason='example_limit'))
                continue
            first, last = record.steps[0], record.steps[-1]
            if set(first.images) != set(last.images):
                raise ValueError('before/after camera names differ')
            example_id = f'H{len(cards) + 1}'
            endpoints = [('before', first)]
            if last.step != first.step:
                endpoints.append(('after', last))
            attached = []
            for temporal_role, step in endpoints:
                for camera, name in sorted(step.images.items()):
                    image_id = f'{example_id}_{temporal_role}_{camera}'
                    label = (f'HISTORICAL icl image_id={image_id} example={example_id} '
                             f'step={step.step} camera={camera} temporal_role={temporal_role}')
                    attached.append(dict(label=label, path=name, image_id=image_id,
                        record_id=record.record_id, episode_id=record.episode_id,
                        step=step.step, camera=camera, temporal_role=temporal_role))
            if len(images) + len(attached) > self.max_images:
                skipped.append(dict(record_id=record.record_id, reason='image_budget_complete_pair'))
                continue

            if knowledge is not None:
                interval = interval_evidence(record, knowledge)
            else:
                samples = [dict(step=s.step, action=list(s.action)) for s in record.steps[:-1]]
                interval = dict(before_step=first.step, after_step=last.step, elapsed_seconds=None,
                    commands_count=last.step - first.step, command_samples=samples,
                    omitted_command_count=last.step - first.step - len(samples),
                    source_reference_id=None, source_reference_sha256=None)

            before_ids = {i['camera']: i['image_id'] for i in attached if i['temporal_role'] == 'before'}
            after_ids = {i['camera']: i['image_id'] for i in attached if i['temporal_role'] == 'after'}
            lines = [
                example_id + ' | source=' + record.source_type + '/' + record.source_split
                    + ' | historical_episode_outcome=' + record.outcome,
                'Record: ' + record.record_id + '; source episode: ' + record.episode_id,
                'Applicability: compare the BEFORE views with CURRENT object placement and hand contact. '
                    'These windows have no verified semantic phase label; use the phase checks above.',
                f'BEFORE observation at source step {first.step}; images=' + _json(before_ids),
                _named_groups(describe_state(first.state, contract)),
            ]
            duration = ('unknown (timestamps unavailable)' if interval['elapsed_seconds'] is None
                        else str(round(interval['elapsed_seconds'], 6)) + ' seconds from source timestamps')
            lines.append(f"RECORDED INTERVAL: steps {first.step}..{last.step}; elapsed={duration}.")
            lines.append(f"{interval['commands_count']} command rows precede the AFTER observation. "
                         f"{len(interval['command_samples'])} boundary command samples follow; "
                         f"{interval['omitted_command_count']} intervening command rows are omitted from this display. "
                         'These samples are not a continuous action queue.')
            for sample in interval['command_samples']:
                lines.append('Recorded absolute command at source step ' + str(sample['step']) + ':\n'
                             + _named_groups(describe_action(sample['action'], contract)))
            if after_ids:
                lines.extend([
                    f'AFTER observation at source step {last.step}; images=' + _json(after_ids),
                    _named_groups(describe_state(last.state, contract)),
                    'MEASURED ROBOT CHANGE (AFTER minus BEFORE; rotation is sign-invariant):\n'
                        + _named_groups(measured_change(first.state, last.state, contract)),
                    'Visual evidence: inspect the paired views for object displacement, support and contact. '
                        'Robot displacement alone does not establish those facts or task success.',
                ])
            else:
                lines.append('AFTER evidence unavailable: a one-step record has no later observation.')
            lines.append('Timing: each row contains an observation and its command. The AFTER image is before '
                         'its own row command; that later command is not presented as a cause of this image. '
                         'Do not attribute the interval result to the displayed samples alone.')
            card = '\n'.join(lines)
            if len(render(cards + [card])) + sum(len(i['label']) for i in images + attached) > self.max_chars:
                skipped.append(dict(record_id=record.record_id, reason='text_budget_complete_pair'))
                continue

            encoded = []
            for item in attached:
                source = self.image_reader(item['path']) if self.image_reader else inside(root, item['path'])
                pixels = rgb_image(source)
                pixels.thumbnail((self.image_side, self.image_side))
                stream = BytesIO()
                pixels.save(stream, format='PNG')
                data = stream.getvalue()
                if len(data) > self.max_image_bytes:
                    raise ValueError('encoded historical image exceeds byte limit')
                encoded.append(dict(**item, data_url='data:image/png;base64,' + base64.b64encode(data).decode('ascii')))
            cards.append(card)
            images.extend(encoded)
            selected.append(dict(record_id=record.record_id, episode_id=record.episode_id,
                channel=record.channel, outcome=record.outcome, score=hit.score,
                visual_score=hit.visual_score, state_score=hit.state_score))
            card_audits.append(dict(example_id=example_id, record_id=record.record_id,
                **interval, states_shown=[step.step for _, step in endpoints],
                images=[dict(image_id=i['image_id'], step=i['step'], camera=i['camera'],
                             temporal_role=i['temporal_role']) for i in attached]))
            record_sources[example_id] = dict(record_id=record.record_id,
                outcome_evidence=record.outcome_evidence, evidence_files=record.evidence_files,
                scene_id=record.scene_id)

        text = render(cards)
        payload_chars = len(text) + sum(len(item['label']) for item in images)
        if payload_chars > self.max_chars:
            raise ValueError('rendered context exceeds the text and image-label budget')
        resources = knowledge.references() if knowledge is not None else ()
        return ContextMemory(text, tuple(images), tuple(selected), dict(
            audit, prompt_revision='paired-observations-named-controls/v2', skipped=skipped,
            text_chars=len(text), payload_text_chars=payload_chars, image_count=len(images),
            selected=selected, example_cards=card_audits,
            retrieval_scope='appearance/state similarity; no automatic semantic phase recognition',
            knowledge_revision=knowledge.revision if knowledge else None,
            knowledge_sources=knowledge.context_sources() if knowledge else {},
            record_sources=record_sources, text_sources={value: key for key, value in text_sources.items()},
            knowledge_sections=len(sections), knowledge_skipped=knowledge_skipped,
            knowledge_truncated=knowledge_truncated,
            knowledge_selected=[dict(section=s['section'], id=s.get('id')) for s in sections]),
            resources=resources)
