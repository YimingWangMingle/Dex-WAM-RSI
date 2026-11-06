"""Production-length sparse attention tail and gradient regression."""
import argparse,importlib.util,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
from thinker_wm.models.temporal_layout import Segment,TemporalLayout,TemporalRequest,TimeGrid,entries_video_segment,history_segment,packed_latent_entries
ap=argparse.ArgumentParser();ap.add_argument('--module',default=str(ROOT/'third_party/Thinker-WM/models/temporal_attention.py'));ap.add_argument('--heads',type=int,default=4);ap.add_argument('--batch',type=int,default=2);ap.add_argument('--reference',action='store_true');a=ap.parse_args()
spec=importlib.util.spec_from_file_location('attention_under_test',a.module);TA=importlib.util.module_from_spec(spec);spec.loader.exec_module(TA)
torch.manual_seed(20261021)
grid=TimeGrid(video_action_freq_ratio=2,action_t_downsample=1)
req=TemporalRequest(grid,target_reads_earlier_targets=False,packed=True,clean_blocks=2,first_target=1,last_target=3,depth_first_frame_conditioned=False)
for slots in (10,9,1):
 times=-torch.arange(slots,0,-1,device='cuda').float().repeat(a.batch,1)
 valid=torch.ones(a.batch,slots,device='cuda',dtype=torch.bool)
 for b in range(a.batch):valid[b,:min(b+1,slots)]=False
 hs=history_segment(times,valid,grid,8,10,pool=2,source_hw=(15,20))
 layout=TemporalLayout(req);layout.video=Segment.cat([hs,entries_video_segment(packed_latent_entries(req),15,20,'cuda')],a.batch)
 layout.history_logit_bias=torch.tensor(-6.,device='cuda',requires_grad=True)
 mask,bias=TA.self_mask(layout,'video',a.batch);lq,lk=mask.shape[-2:]
 print(json.dumps(dict(event='begin',slots=slots,shape=[a.batch,a.heads,lq,lk])),flush=True)
 q,k,v=[torch.randn(a.batch,n,a.heads,128,device='cuda',dtype=torch.bfloat16).requires_grad_() for n in (lq,lk,lk)]
 results={}
 for impl in (('dense','flex') if a.reference else ('flex',)):
  TA.ATTN_IMPL=impl
  out=TA.masked_attention(q,k,v,*TA.self_mask(layout,'video',a.batch))
  grads=torch.autograd.grad(out.float().square().mean(),(q,k,v,layout.history_logit_bias))
  torch.cuda.synchronize()
  assert all(torch.isfinite(x).all() for x in (out,*grads))
  results[impl]=[x.detach().float() for x in (out,*grads)]
 if a.reference:
  errors=[float((f-d).norm()/d.norm().clamp_min(1e-12)) for f,d in zip(results['flex'],results['dense'])]
  absolute_errors=[float((f-d).abs().max()) for f,d in zip(results['flex'],results['dense'])]
  assert max(errors[:4])<.02,errors
  # With all history masked the scalar bias gradient is zero analytically;
  # BF16 reduction roundoff must be checked with an absolute tolerance.
  assert errors[4]<.02 or absolute_errors[4]<1e-8,(errors,absolute_errors)
 else:errors=[]
 print(json.dumps(dict(event='passed',slots=slots,shape=[a.batch,a.heads,lq,lk],relative_errors=errors)),flush=True)
 del q,k,v,out,grads,results,layout,mask,bias
