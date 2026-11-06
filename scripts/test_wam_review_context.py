import sys,json,hashlib,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from environments.robodojo.review_context import build_review_packet

class Tests(unittest.TestCase):
 def setUp(self):
  from unittest.mock import patch
  patcher=patch.dict("os.environ", {"RSI_A2V_DEPTH":"1","RSI_A2V_RGB_FRAMES":"6","RSI_A2V_METRIC_DEPTH":"0"});patcher.start();self.addCleanup(patcher.stop)
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.paths={}
  for step in (0,8):
   p=self.root/'observations'/'episode'/f'{step:06d}';p.mkdir(parents=True)
   images={};hashes={}
   for camera in ('head','left','right'):
    image=p/(camera+'.png');image.write_bytes((camera+str(step)).encode());images[camera]=str(image);hashes[str(image)]=hashlib.sha256(image.read_bytes()).hexdigest()
   d=dict(step=step,instruction='stack bowls',images=images,image_hashes=hashes,state={'joint':step})
   f=p/'observation.json';f.write_text(json.dumps(d));self.paths[step]=str(f)
  events=[dict(event='review',episode_id='episode',proposal_id='p',decision='accept',candidate_id=2),dict(event='execution',episode_id='episode',proposal_id='p',before=dict(payload_ref=self.paths[0]),after=dict(control_step=8),approved_actions=[[1]]*16,executed_actions=[[1]]*8,stop_reason='event_interrupted')]
  (self.root/'framework-events.jsonl').write_text('\n'.join(map(json.dumps,events)))
  self.packet=dict(decision_step=8,instruction='stack bowls',candidates=[dict(candidate_id=i,actions=[[i]*14]*24) for i in range(3)],allowed_execute_steps=[24,16,8,4],framework_context='internal trajectories'*1000)
 def test_real_before_after_and_concise_feedback(self):
  p=build_review_packet(self.packet,self.root)
  self.assertEqual(len(p['images']),6);self.assertEqual([i['control_step'] for i in p['images']],[0]*3+[8]*3)
  self.assertEqual(p['execution_feedback'][-1]['candidate_id'],2)
  self.assertEqual(p['execution_feedback'][-1]['approved_steps'],16);self.assertEqual(p['execution_feedback'][-1]['executed_steps'],8)
  self.assertNotIn('framework_context',p);self.assertEqual(p['candidates'],self.packet['candidates'])
 def test_first_request_only_current_views(self):
  self.packet['decision_step']=0;p=build_review_packet(self.packet,self.root)
  self.assertEqual(len(p['images']),3);self.assertEqual(p['execution_feedback'],[])
 def test_image_tamper_fails(self):
  Path(self.paths[0]).with_name('head.png').write_bytes(b'changed')
  with self.assertRaises(ValueError):build_review_packet(self.packet,self.root)
 def test_forecast_binding_rejected(self):
  for c in self.packet['candidates']:c.update(action_hash='a',forecast={'action_hash':'b'})
  with self.assertRaises(ValueError):build_review_packet(self.packet,self.root)
 def test_cross_episode_history_fails(self):
  p=self.root/'framework-events.jsonl';s=p.read_text().replace(self.paths[0],str(self.root/'observations'/'foreign'/'000000'/'observation.json'));p.write_text(s)
  with self.assertRaises(ValueError):build_review_packet(self.packet,self.root)
 def test_W1_only_adds_bound_predictions(self):
  baseline=build_review_packet(self.packet,self.root)
  for c in self.packet['candidates']:
   c['action_hash']='hash-'+str(c['candidate_id'])
   c['forecast']=dict(action_hash=c['action_hash'],frames=[str(Path(self.paths[8]).with_name('head.png'))]*12,depth_frames=[str(Path(self.paths[8]).with_name('left.png'))]*12,frame_steps=list(range(10,33,2)))
  p=build_review_packet(self.packet,self.root)
  self.assertEqual(len(p['images']),42)
  self.assertEqual(sum(i['kind']=='predicted_RGB' for i in p['images']),18)
  self.assertEqual(sum(i['kind']=='predicted_depth' for i in p['images']),18)
  self.assertEqual(p['candidates'][0]['forecast']['reviewed_offsets'],[2,6,10,16,20,24])
  self.assertEqual(p['images'][:6],baseline['images']);self.assertEqual(p['execution_feedback'],baseline['execution_feedback'])

 def test_only_last_two_decision_observations(self):
  for step in (16,24):
   folder=self.root/'observations'/'episode'/f'{step:06d}';folder.mkdir()
   d=json.loads(Path(self.paths[8]).read_text());d['step']=step
   f=folder/'observation.json';f.write_text(json.dumps(d));self.paths[step]=str(f)
  events=[]
  for before,after in [(0,8),(8,16),(16,24)]:
   events.append(dict(event='execution',episode_id='episode',proposal_id=str(before),before=dict(payload_ref=self.paths[before]),after=dict(control_step=after),approved_actions=[[1]]*8,executed_actions=[[1]]*8,stop_reason='completed'))
  (self.root/'framework-events.jsonl').write_text('\n'.join(map(json.dumps,events)))
  self.packet['decision_step']=24
  p=build_review_packet(self.packet,self.root)
  self.assertEqual([i['control_step'] for i in p['images']],[8]*3+[16]*3+[24]*3)
  self.assertEqual(len(p['execution_feedback']),2)

if __name__=='__main__':unittest.main()
