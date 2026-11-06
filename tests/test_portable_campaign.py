import tempfile
from pathlib import Path
import unittest
from examples.modular.campaign_demo import run

class CampaignTests(unittest.TestCase):
    def test_policy_only_candidate_inherits_with_fixed_tool_requirements(self):
        from examples.modular.selection_campaign_demo import run as run_selection
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'campaign'
            result=run_selection(root)
            self.assertEqual([r['outcome'] for r in result['rounds']],['inherited','retained'])
            summaries=[json.loads(p.read_text()) for p in (root/'episodes').rglob('episode-summary.json')]
            self.assertTrue(summaries)
            self.assertEqual({tuple(s['selection_policy']['config']['skills']) for s in summaries},{('small',),('large',)})
            self.assertTrue(all(s['tool_usage']['calls']==2 for s in summaries))

    def test_actual_core_inherits_and_retains_without_wam_or_robodojo(self):
        with tempfile.TemporaryDirectory() as root:
            result=run(Path(root)/'campaign')
            self.assertEqual([r['outcome'] for r in result['rounds']],['inherited','retained'])
            self.assertEqual(result['current']['harness']['id'],'child')

if __name__=='__main__':unittest.main()
