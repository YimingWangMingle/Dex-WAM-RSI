"""Evidence-backed episode notes; never auto-promote a hypothesis to a rule."""
import json
from pathlib import Path


class EpisodeMemory:
    def __init__(self, root, serializer):
        self.root=Path(root); self.serializer=serializer
        self.entries=[]

    def add(self, step, tool, arguments, result):
        entry=dict(evidence_id=f'feedback-{len(self.entries)+1}',control_step=step,
                   tool=tool,arguments=arguments,result=result,
                   scope='this_episode',cross_episode_status='candidate_not_validated')
        # Snapshot arrays instead of retaining references to changing state.
        entry=json.loads(json.dumps(entry,default=self.serializer))
        self.entries.append(entry)
        with (self.root/'experience-evidence.jsonl').open('a') as f:
            f.write(json.dumps(entry,ensure_ascii=False)+'\n')
        self.save()

    def context(self):
        # Preserve early rejected attempts instead of only the last ten turns.
        significant=[e for e in self.entries if e['result'].get('status') not in ('measured','scene_registered')]
        def compact(e):
            fields=('status','reason','executed_steps','follows_lift','actual_tcp','target_tcp',
                    'position_error_m','next_action','source_measurement_id')
            return dict(evidence_id=e['evidence_id'],control_step=e['control_step'],tool=e['tool'],
                        arguments=e['arguments'],result={k:v for k,v in e['result'].items() if k in fields})
        return dict(recent=[compact(e) for e in self.entries[-4:]],earlier_evidence=[compact(e) for e in significant[-20:]],
                    scope='episode only; outcomes are observations, reasons are Agent hypotheses')

    def save(self, native_success=None):
        payload=dict(task='stack_bowls',status='candidate_not_validated',
                     native_success=native_success,entries=self.entries)
        target=self.root/'experience-candidate.json'
        temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2))
        temporary.replace(target)
        lines=['# 本局经验记录','', '仅记录执行证据；跨局规则仍待独立布局验证。', '']
        for e in self.entries:
            r=e['result']; status=r.get('status','unknown')
            lines.append(f"- {e['evidence_id']}，step {e['control_step']}，{e['tool']}：{status}；实际执行 {r.get('executed_steps',0)} 步。")
            if r.get('reason'): lines.append('  原因：'+str(r['reason']))
            if 'follows_lift' in r: lines.append('  碗随抬升移动：'+str(r['follows_lift']))
        if native_success is not None: lines.extend(['',f'本局原生评测成功：{native_success}。单局结果不自动成为通用规则。'])
        (self.root/'experience-live.md').write_text('\n'.join(lines)+'\n')
