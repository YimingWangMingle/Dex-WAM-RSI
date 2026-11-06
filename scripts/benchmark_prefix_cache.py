#!/usr/bin/env python3
"""Compare fixed-prefix cache against recomputation on a real checkpoint.

Uses existing weights and measured RoboTwin data; performs no optimizer updates.
New history parameters alone may be missing. Synthetic CPU tests cover training.
"""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from thinker_wm.scripts.eval_video_ckpt import create_eval_model, load_config
from thinker_wm.data_prepare.history_attach import RobotwinHistoryReader, HistoryAttacher
from thinker_wm.data_prepare.temporal_windows import WindowSpec
from thinker_wm.data_prepare.history_memory import EventSpec
from thinker_wm.models.inference_api import WAMInference, Observation, GenerationSettings, make_message_builder


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','checkpoint','catalog','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--steps',type=int,default=2)
    p.add_argument('--blocks', type=int, nargs='+', default=[1,4])
    p.add_argument('--modes', nargs='+', choices=['joint','idm','fdm'], default=['joint','idm','fdm'])
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--cache-mb', type=float, default=1024.)
    args=p.parse_args()
    cfg=load_config(args.config)
    cfg.common.control_hz=50
    cfg.common.video_action_freq_ratio=4
    cfg.common.num_video_frames=16
    cfg.common.action_chunk_size=64
    cfg.model.load_pretrained_backbones=False
    for k,v in dict(temporal_mode='block_causal',ar_packed_teacher_forcing=True,
                    ar_target_reads_earlier_targets=False,ar_latents_per_block=1,
                    ar_cut_probs=None,action_t_downsample=1,history_enable=True,
                    history_spatial_pool=2,depth_condition_mode='rgb_only',
                    depth_share_video_noise=False,temporal_und_reads='observed',
                    world_mode_prob=0.,training_mode_probs=dict(joint=.6,idm=.2,fdm=.2)).items():
        cfg.model.wm[k]=v
    print('Building model',flush=True)
    model=create_eval_model(cfg,load_pretrained_backbones=False)
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
    rd=RobotwinHistoryReader(args.catalog)
    ep=max(rd.records,key=lambda k:int(rd.records[k]['state']['count']))
    decision=min(400,int(rd.records[ep]['state']['count'])-66)
    if decision<60:raise ValueError('need a longer episode')
    hw=(int(cfg.common.video_height),int(cfg.common.video_width))
    px=rd.rgb(ep,list(range(decision,decision+65,4)),hw)
    state=rd.state(ep,[decision])[0]
    names=list(cfg.dataset.robot_names)
    if len(names)!=1:raise ValueError('select one robot')
    obs=Observation(px[0],state,decision,decision/50,names[0])
    batch=dict(video_frames=torch.zeros(1,16,3,*hw),episode_ids=[ep],
               frame_indices=[list(range(decision,decision+65,4))],source_datasets=['robotwin_3d'])
    spec=WindowSpec(control_hz=50,num_video_frames=16,recent_frames=5,sparse_slots=8,
                    sparse_stride_s=1.,sparse_jitter=0.,include_episode_start=False,require_full_future=False)
    attach=HistoryAttacher(spec,{'robotwin_3d':rd},seed=20261015,events=EventSpec(),strict_sources=True)
    history=attach.attach(batch,0)
    from dataclasses import replace
    from thinker_wm.models import temporal_training as TT
    import statistics
    with torch.inference_mode():
        hctx=model._prepare_history(history,names)
    req=TT.sample_step_decision(model.config,0,False,False).request
    message=make_message_builder(cfg,'robotwin_3d')('Pick up the object.')
    import PIL.Image
    pil=PIL.Image.fromarray((px[0].permute(1,2,0).numpy()*255).round().astype(np.uint8))
    commands=rd.reader.read_action(rd.records[ep],list(range(decision,decision+64))).float()
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(checkpoint=str(ckpt),gpu=torch.cuda.get_device_name(0),steps=args.steps,
        timing_boundary='history VAE pre-encoded; includes current image VAE/VLM and denoising, no pixel decode',
        note='Numerical/latency check only; history/AR not trained, no task-success claims.',
        new_untrained_parameters=list(missing),history_metrics=attach.last_metrics,cases=[])
    def call(mode, blocks, cache):
        cap={}
        kw={}
        r=req
        if mode=='idm':
            r=replace(req,video_role='goal');kw['condition_video_frames']=px[1:1+blocks*4].unsqueeze(0)
        if mode=='fdm':
            r=replace(req,action_role='world');kw['condition_actions']=commands[:blocks*16].unsqueeze(0)
        torch.manual_seed(20261015)
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            out=model.inference_step(first_frame=px[:1].to('cuda'),state=state[None].to('cuda'),
                first_frame_pils=[pil],vlm_input_messages=[message],robot_names=names,
                temporal_request=r,history_context=hctx,num_blocks=blocks,
                num_inference_steps=args.steps,action_num_inference_steps=4,
                decode_outputs=False,latent_capture=cap,prefix_cache_mode=cache,
                prefix_cache_max_mb=args.cache_mb,**kw)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        arrays={k:cap[k].float().cpu() for k in ('video_latent','depth_latent','action_latent')}
        arrays['actions']=out[1].float().cpu()
        return arrays, elapsed, cap['prefix_cache']
    print('Warmup',flush=True)
    call('joint',1,'off');call('joint',1,'fixed')
    for mode in args.modes:
        for blocks in args.blocks:
            timings={'off':[],'fixed':[]};records={};stats=None
            for repeat in range(args.repeats):
                for cache in (('off','fixed') if repeat%2==0 else ('fixed','off')):
                    arrays,elapsed,st=call(mode,blocks,cache)
                    timings[cache].append(elapsed);records[cache]=arrays
                    if cache=='fixed':stats=st
            errors={}
            for k,ref in records['off'].items():
                got=records['fixed'][k];diff=(got-ref).abs()
                errors[k]=dict(max_abs=float(diff.max()),
                    relative_rms=float(diff.square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-8)),
                    allclose=bool(torch.allclose(got,ref,atol=0.01,rtol=0.01)),
                    finite=bool(torch.isfinite(got).all()))
            case=dict(mode=mode,blocks=blocks,timings_s=timings,
                median_s={k:statistics.median(v) for k,v in timings.items()},errors=errors,cache=stats)
            case['speedup']=case['median_s']['off']/case['median_s']['fixed']
            result['cases'].append(case)
            output.write_text(json.dumps(result,indent=2))
            print(json.dumps(case),flush=True)
    result['passed']=all(e['allclose'] and e['finite'] for c in result['cases'] for e in c['errors'].values())
    output.write_text(json.dumps(result,indent=2))
    print('PASS' if result['passed'] else 'FAIL',str(output),flush=True)
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
