#!/usr/bin/env python3
"""Real-checkpoint single-GPU Joint/IDM/FDM interface smoke, not a quality benchmark.

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
    ep=next(iter(rd.records))
    decision=min(400,int(rd.records[ep]['state']['count'])-66)
    if decision<60:raise ValueError('need a longer episode')
    hw=(int(cfg.common.video_height),int(cfg.common.video_width))
    px=rd.rgb(ep,list(range(decision,decision+17,4)),hw)
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
    api=WAMInference(model,checkpoint=str(ckpt),action_stats_version='checkpoint_config',camera_transform='robotwin_3camera_pad',
        settings=GenerationSettings(num_inference_steps=args.steps,seed=20261015,decode_outputs=True,return_depth=True),
        message_builder=make_message_builder(cfg,'robotwin_3d'))
    result=dict(checkpoint=str(ckpt),gpu=torch.cuda.get_device_name(0),steps=args.steps,
        note='Interface/finite-output smoke only; not adapted training, latency benchmark or task success.',
        new_untrained_parameters=list(missing),history_metrics=attach.last_metrics,cases={})
    prompt='Pick up the object.'
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    commands=rd.reader.read_action(rd.records[ep],list(range(decision,decision+16))).float().numpy()
    for mode in ('joint','idm','fdm'):
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            if mode=='joint':
                out=api.generate_action(obs,history,prompt)
                forecast=out.forecast
            elif mode=='idm':
                out=api.generate_action_for_goal(obs,history,prompt,px[1:])
                assert not hasattr(out,'forecast')
                forecast=None
            else:
                out=api.predict_outcome(obs,history,prompt,commands)
                forecast=out
        torch.cuda.synchronize()
        assert out.actions.shape==(16,state.numel()) and np.isfinite(out.actions).all()
        if forecast is not None:
            for value in (forecast.video_latent,forecast.depth_latent,forecast.frames,forecast.depth):
                assert value is not None and torch.isfinite(value).all()
        case=dict(elapsed_s=time.perf_counter()-start,action_shape=list(out.actions.shape),
                  finite=True,has_forecast=forecast is not None,
                  frame_shape=None if forecast is None else list(forecast.frames.shape),
                  depth_shape=None if forecast is None else list(forecast.depth.shape))
        if mode=='fdm':
            case['fixed_action_max_error']=float(np.abs(out.actions-commands).max())
        result['cases'][mode]=case
        print(mode,json.dumps(case),flush=True)
        output.write_text(json.dumps(result,indent=2))
    print('PASS',str(output),flush=True)

if __name__=='__main__':main()
