#!/usr/bin/env python3
"""Single-GPU, no-KV-cache AR/history latency; not a policy-quality evaluation.

Example: CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_history_latency.py \
 --config <V15.yaml> --checkpoint <step52000> --catalog <catalog.jsonl> --output <result.json>
Each measured call rebuilds history VAE features. Existing base weights are
required; only newly introduced history adapter parameters may be absent.
"""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from thinker_wm.scripts.eval_video_ckpt import create_eval_model, load_config
from thinker_wm.data_prepare.history_attach import RobotwinHistoryReader, HistoryAttacher
from thinker_wm.data_prepare.temporal_windows import WindowSpec
from thinker_wm.data_prepare.history_memory import EventSpec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','checkpoint','catalog','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--steps',type=int,default=10)
    p.add_argument('--action-steps',type=int,default=10)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--cases',default='none,r5,r9,r13,r5s8,r5s8b4')
    args=p.parse_args()
    cfg=load_config(args.config)
    cfg.common.control_hz=50
    cfg.common.video_action_freq_ratio=4
    cfg.common.num_video_frames=16
    cfg.common.action_chunk_size=64
    cfg.model.load_pretrained_backbones=False
    for k,v in dict(temporal_mode='block_causal',ar_packed_teacher_forcing=True,
                    ar_target_reads_earlier_targets=False,ar_latents_per_block=1,
                    history_enable=True,history_spatial_pool=2,depth_condition_mode='rgb_only',
                    depth_share_video_noise=False,temporal_und_reads='observed').items():
        cfg.model.wm[k]=v
    cfg.model.wm.ar_cut_probs=None
    print('Building model',flush=True)
    model=create_eval_model(cfg, load_pretrained_backbones=False)
    ckpt=Path(args.checkpoint)
    if ckpt.is_dir():ckpt=ckpt/'pytorch_model_0.bin'
    print('Loading checkpoint',str(ckpt),flush=True)
    payload=torch.load(ckpt,map_location='cpu',weights_only=False)
    sd=payload.get('module',payload.get('state_dict',payload))
    if all(k.startswith('module.') for k in sd):sd={k[7:]:v for k,v in sd.items()}
    missing,unexpected=model.load_state_dict(sd,strict=False)
    if unexpected or any(not k.startswith('world_model.history_encoder.') for k in missing):
        raise RuntimeError(dict(missing=missing,unexpected=unexpected))
    del sd,payload
    model.to('cuda').eval()
    reader=RobotwinHistoryReader(args.catalog)
    ep=next(iter(reader.records))
    d=min(400,int(reader.records[ep]['state']['count'])-66)
    if d<60:raise ValueError('benchmark needs a longer episode')
    h,w=int(cfg.common.video_height),int(cfg.common.video_width)
    px=reader.rgb(ep,[d],(h,w))
    st=reader.state(ep,[d])
    names=list(cfg.dataset.robot_names)
    if len(names)!=1:raise ValueError('select one robot for this benchmark')
    # A real RGB/state input and a fixed task message; success is not evaluated.
    prompt='Pick up the object.'
    pil=Image.fromarray((px[0].permute(1,2,0).numpy()*255).round().astype('uint8'))
    batch=dict(video_frames=torch.zeros(1,16,3,h,w),episode_ids=[ep],
               frame_indices=[list(range(d,d+65,4))],source_datasets=['robotwin_3d'])
    result=dict(checkpoint=str(ckpt),gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
                episode=ep,decision=d,steps=args.steps,action_steps=args.action_steps,
                guidance_scale=5,action_guidance_scale=1,resolution=[h,w],control_hz=50,
                new_untrained_parameters=list(missing),kv_cache=False,decode_outputs=True,
                benchmark_sparse_stride_s=.08,
                note='Latency only. History/RGB-only/AR are not trained in V15. Sparse frames benchmark memory size, not recovery success.',cases={})
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    for case in args.cases.split(','):
        recent=13 if case=='r13' else 9 if case=='r9' else 5
        sparse=8 if 's8' in case else 0
        blocks=4 if case.endswith('b4') else 1
        spec=WindowSpec(control_hz=50,num_video_frames=16,recent_frames=recent,
                        sparse_slots=sparse,sparse_stride_s=.08,sparse_jitter=0,
                        include_episode_start=False,require_full_future=False)
        attach=HistoryAttacher(spec,{'robotwin_3d':reader},seed=20261015,
                                events=EventSpec(enabled=False),strict_sources=True)
        samples=[]
        for i in range(args.warmup+args.repeats):
            torch.manual_seed(20261015+i)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode():
                raw=None if case=='none' else attach.attach(batch,0)
                read=time.perf_counter()
                ctx=None if raw is None else model._prepare_history(raw,names)
                torch.cuda.synchronize();encoded=time.perf_counter()
                if ctx is not None and int(ctx['valid'].sum()) != (recent-1)//4+1+sparse:
                    raise ValueError('benchmark requires every requested history slot to be valid')
                out=model.inference_step(first_frame=px.to('cuda',model.dtype),state=st.to('cuda'),
                    first_frame_pils=[pil],vlm_input_messages=[prompt],t5_input_messages=[prompt],
                    robot_names=names,predict_action=True,num_inference_steps=args.steps,
                    action_num_inference_steps=args.action_steps,guidance_scale=5,action_guidance_scale=1,
                    history_context=ctx,num_blocks=blocks,return_depth=True,decode_outputs=True)
                torch.cuda.synchronize();end=time.perf_counter()
            if any(torch.is_tensor(x) and not torch.isfinite(x).all() for x in out):
                raise ValueError('nonfinite inference output')
            record=dict(read_s=read-start,history_vae_s=encoded-read,inference_s=end-encoded,
                        total_s=end-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        history_latents=0 if ctx is None else int(ctx['valid'].sum()),
                        shapes=[list(x.shape) if torch.is_tensor(x) else None for x in out])
            print(case,'warmup' if i<args.warmup else 'measured',json.dumps(record),flush=True)
            if i>=args.warmup:samples.append(record)
            del out,ctx,raw
        result['cases'][case]=dict(blocks=blocks,recent=0 if case=='none' else recent,sparse=sparse,
            samples=samples,median_s=float(np.median([s['total_s'] for s in samples])),
            median_inference_s=float(np.median([s['inference_s'] for s in samples])))
        output.write_text(json.dumps(result,indent=2))
    print('RESULT',str(output),flush=True)

if __name__=='__main__':main()
