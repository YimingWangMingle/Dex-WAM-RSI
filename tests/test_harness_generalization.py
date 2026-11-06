"""Dependency and contract regressions for the standalone harness."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

class DecouplingTests(unittest.TestCase):
    def test_demo_runs_without_site_packages(self):
        result=subprocess.run([sys.executable,'-S','-m','examples.modular.harness_core_demo'],
            cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        data=json.loads(result.stdout)
        self.assertEqual(data['mode'],'synthetic-contract-demo')
        self.assertEqual(data['result']['counters']['executed_steps'],3)
        self.assertEqual(data['result']['activated_skills'],['wholebody_motion','contact_recovery'])
        self.assertNotIn('task_success',data)

    def test_wholebody_and_code_recovery_skill_identity(self):
        from skills.contracts import SkillIdentity
        identity=SkillIdentity('wholebody_motion','ardy-gmr-sonic/v1','wholebody-reference/v1',29,'model-revision')
        self.assertEqual(identity.action_dim,29)
        SkillIdentity('contact_recovery','code/v1','wholebody-reference/v1',29)
        with self.assertRaises(ValueError):
            SkillIdentity(' ','v1','wholebody-reference/v1',29)

    def test_core_contract_modules_import_without_site_packages(self):
        modules=['rsi.version_manifest','rsi.proposal_adapter','skills.contracts','skills.chunk_backend','execution.runner','execution.review','agents.context','experience.packages']
        code='import sys,importlib;sys.path.insert(0,'+repr(str(ROOT))+');'+''.join(
            'importlib.import_module('+repr(name)+');' for name in modules)
        result=subprocess.run([sys.executable,'-S','-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_model_backed_skill_requires_explicit_checkpoint(self):
        from skills.contracts import ModelControlSkill,SkillIdentity
        with self.assertRaises(ValueError):
            ModelControlSkill(SkillIdentity('motion','v1','reference/v1',29),object())

    def parent(self,root):
        from rsi.version_manifest import CLOSURE,write_backend_version,build_harness
        files={}
        for kind in CLOSURE:
            path=root/kind/'metadata.json';path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(dict(component=kind,source='unit-test')))
            files[kind]=[str(path.relative_to(root))]
        backend=write_backend_version(root,backend_revision='ardy-gmr-sonic/v1',
            action_contract='wholebody-reference/v1',control_hz=50,runtime_dependencies={'sonic':'sha256:test'})
        files['configuration'].append(str(backend.relative_to(root)))
        return build_harness(root,'parent',files)

    def test_candidate_version_and_frozen_inputs(self):
        from rsi.version_manifest import verify_harness,backend_version
        from rsi.proposal_adapter import materialize_candidate
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);parent=self.parent(base/'parent')
            candidate=materialize_candidate(parent,{'control/metadata.json':{'contact_threshold':15}},
                new_id='candidate',out_root=base/'candidates')
            self.assertNotEqual(verify_harness(parent),verify_harness(candidate))
            self.assertEqual(backend_version(candidate)['control_hz'],50)
            for name in ('weights/metadata.json','configuration/metadata.json'):
                with self.assertRaises(ValueError):
                    materialize_candidate(parent,{name:'changed'},new_id='frozen',out_root=base/'candidates')

    def test_candidate_path_escape_is_rejected_before_writes(self):
        from rsi.proposal_adapter import materialize_candidate
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);parent=self.parent(base/'parent')
            with self.assertRaises(ValueError):
                materialize_candidate(parent,{'../escaped.txt':'bad'},new_id='candidate',out_root=base/'candidates',
                    new_files={'../escaped.txt':'control'})
            self.assertFalse((base/'candidates/escaped.txt').exists())

    def test_candidate_identifier_cannot_escape_output_root(self):
        from rsi.proposal_adapter import materialize_candidate
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);parent=self.parent(base/'parent')
            with self.assertRaises(ValueError):
                materialize_candidate(parent,{},new_id='../escaped',out_root=base/'candidates')
            self.assertFalse((base/'escaped').exists())

    def test_path_alias_cannot_reassign_a_frozen_file(self):
        from rsi.proposal_adapter import materialize_candidate
        from rsi.version_manifest import verify_harness
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);parent=self.parent(base/'parent');before=verify_harness(parent)
            for i,alias in enumerate(('configuration/./backend_version.json',
                                      'configuration//backend_version.json',
                                      './configuration/backend_version.json')):
                with self.subTest(alias=alias),self.assertRaises(ValueError):
                    materialize_candidate(parent,{alias:{'backend_revision':'changed'}},
                        new_id=f'alias{i}',out_root=base/'candidates',new_files={alias:'control'})
            self.assertEqual(verify_harness(parent),before)

    def test_evidence_captures_actual_runner_handoffs_and_reviews(self):
        from examples.modular.harness_core_demo import ScriptedBackend,ScriptedAgent,ScriptedExecutor,CONTRACT
        from agents.context import AgentContext,TaskState
        from skills.contracts import ControlSkill,SkillIdentity,SkillRegistry
        from execution.runner import TaskSkillRunner
        from rsi.proposal_adapter import evidence_bundle
        registry=SkillRegistry()
        for name,rows in [('wholebody_motion',((.01,)*29,)*3),('contact_recovery',((0.,)*29,))]:
            registry.register(ControlSkill(SkillIdentity(name,'synthetic/v1',CONTRACT,29),ScriptedBackend(rows)))
        events=[]
        TaskSkillRunner(registry,ScriptedExecutor(),ScriptedAgent(),
            AgentContext(TaskState('synthetic event test')),sink=events.append).run('synthetic-episode')
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'events.jsonl';path.write_text(''.join(json.dumps(e)+'\n' for e in events))
            result=evidence_bundle({'episodes':[],'evaluator_revision':'fixed/v1'},[path])
        self.assertEqual(result['skill_handoffs'],{'contact_recovery':1})
        self.assertEqual(result['review_decisions'],{'accept':2})

    def test_evidence_uses_generic_measured_events(self):
        from rsi.proposal_adapter import evidence_bundle
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'events.jsonl'
            p.write_text(json.dumps({'event':'execution','stop_reason':'interrupted:contact'})+'\n')
            result=evidence_bundle({'episodes':[{'task':'S09','case':'seed0','outcome':'failure'}],
                                    'evaluator_revision':'fixed/v1'},[p])
            self.assertEqual(result['execution_stop_reasons'],{'interrupted:contact':1})
            self.assertIsNone(result['visual_mse'])

    def test_aliases_and_duplicate_ownership_rejected_before_upstream(self):
        import copy
        import types
        from unittest.mock import patch
        from rsi.version_manifest import verify_harness
        calls=[]
        upstream=types.ModuleType('PhysicalRSI_core.self_harness.artifacts')
        upstream.verify_harness=lambda h: calls.append(h) or 'upstream-result'
        with tempfile.TemporaryDirectory() as temp:
            parent=self.parent(Path(temp)/'parent')
            with patch.dict(sys.modules,{'PhysicalRSI_core.self_harness.artifacts':upstream}):
                self.assertEqual(verify_harness(parent),'upstream-result')
                for alias in ('./configuration/backend_version.json','configuration//backend_version.json'):
                    malformed=copy.deepcopy(parent)
                    value=malformed['components']['configuration'].pop('configuration/backend_version.json')
                    malformed['components']['configuration'][alias]=value
                    with self.assertRaises(ValueError):verify_harness(malformed)
                duplicate=copy.deepcopy(parent)
                duplicate['components']['control']['configuration/backend_version.json']=parent['components']['configuration']['configuration/backend_version.json']
                with self.assertRaises(ValueError):verify_harness(duplicate)
                self.assertEqual(len(calls),1)

    def test_new_files_must_be_editable_explicit_and_noncolliding(self):
        from rsi.proposal_adapter import materialize_candidate
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);parent=self.parent(base/'parent')
            cases=[({'weights/new.txt':'x'},{'weights/new.txt':'weights'}),
                   ({},{'control/new.txt':'control'}),
                   ({'control/metadata.json/child':'x'},{'control/metadata.json/child':'control'})]
            for i,(edits,new_files) in enumerate(cases):
                with self.subTest(i=i),self.assertRaises(ValueError):
                    materialize_candidate(parent,edits,new_id=f'bad{i}',out_root=base/'out',new_files=new_files)
            self.assertFalse((base/'out').exists())
            child=materialize_candidate(parent,{'control/new.txt':'new'},new_id='good',
                                        out_root=base/'out',new_files={'control/new.txt':'control'})
            self.assertIn('control/new.txt',child['components']['control'])

    def test_legacy_wam_identity_is_preserved_and_generic_schema_validated(self):
        from rsi.version_manifest import (CLOSURE,build_harness,write_wam_version,write_backend_version,
                                         wam_version,verify_harness,backend_version,WAM_VERSION_FILE)
        from rsi.proposal_adapter import materialize_candidate
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);parent=self.parent(base/'parent')
            root=Path(parent['root'])
            fields=dict(checkpoint='test-only',checkpoint_sha256='0'*64,action_stats_version='v1',
                        history={},camera_transform='fixture',data_version='fixture',model_config_sha256='1'*64)
            write_wam_version(root,**fields)
            files={k:list(v) for k,v in parent['components'].items()}
            files['configuration']=[n for n in files['configuration'] if not n.endswith('backend_version.json')]+[WAM_VERSION_FILE]
            legacy=build_harness(root,'legacy',files)
            before=verify_harness(legacy);raw=(root/WAM_VERSION_FILE).read_bytes()
            child=materialize_candidate(legacy,{'control/metadata.json':'edit'},new_id='child',out_root=base/'out')
            self.assertEqual(wam_version(child),fields)
            self.assertEqual(child['components']['configuration'],legacy['components']['configuration'])
            self.assertEqual(verify_harness(legacy),before)
            self.assertEqual((root/WAM_VERSION_FILE).read_bytes(),raw)
            with self.assertRaises(ValueError):backend_version(legacy)
            for hz in (0,-1,True,float('nan'),float('inf'),'50'):
                with self.subTest(hz=hz),self.assertRaises(ValueError):
                    write_backend_version(base/'invalid',backend_revision='v1',action_contract='v1',control_hz=hz,runtime_dependencies={})
            self.assertFalse((base/'invalid').exists())

    def test_generic_evidence_keeps_legacy_prediction_statistics(self):
        from rsi.proposal_adapter import evidence_bundle
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'events.jsonl'
            records=[{'role':'error','features':{'valid_visual':True,'visual_mse_max':x}} for x in (1,2,4)]
            records += [{'role':'decision','action':'recover'},{'event':'review','decision':'accept'},
                        {'event':'handoff','from_skill':'motion','to_skill':'recover'},
                        {'event':'execution','stop_reason':'completed'}]
            path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            result=evidence_bundle({'episodes':[],'evaluator_revision':'test'},[path])
            self.assertAlmostEqual(result['visual_mse']['mean'],7/3)
            self.assertAlmostEqual(result['visual_mse']['p90'],3.6)
            self.assertEqual(result['recovery_decisions'],{'recover':1})
            self.assertEqual(result['review_decisions'],{'accept':1})
            self.assertEqual(result['skill_handoffs'],{'recover':1})

    def test_legacy_chunk_backend_is_same_generic_implementation(self):
        from skills.chunk_backend import CommittedChunkBackend
        from skills.wam.backend import WAMControlBackend
        self.assertIs(WAMControlBackend,CommittedChunkBackend)
