"""Bowl instances and conditional metric geometry from RGB, independent of Agent.

No depth/segmentation/object-pose simulator APIs. Same-observation multi-view
fits only. Stable IDs are tracked in the fixed head image with a uniqueness
gate; an ambiguous assignment fails closed instead of silently changing IDs.
"""
from __future__ import annotations

from dataclasses import asdict
from itertools import permutations
from typing import Any
import time
import copy
import uuid
import numpy as np

from environments.robodojo.runtime.robodojo_bowl_vision import BowlDetector, extract_rim
from environments.robodojo.runtime.robodojo_bowl_geometry import GeometryConfig, RimCandidate, RimObservation, fit_bowl_rim


class BowlPerception:
    def __init__(self, model_path: str, device: str = 'cuda:0', detector: Any = None) -> None:
        self.detector = detector if detector is not None else BowlDetector(model_path, device)
        self.episode_id=uuid.uuid4().hex
        self.observation_id=0
        self.lost_ids=set()
        self.moving_ids=set()
        self.recovery_hits={}
        self.identity_recovery={}
        self.previous: dict[str, np.ndarray] = {}
        self.latest: dict[str, dict] = {}
        self.step = -1
        self.inventory: dict[str, dict] = {}

    def observe(self, frames: dict, step: int) -> dict:
        start = time.monotonic()
        self.latest={};self.step=-1
        self.observation_id+=1
        detections = {}
        for name in sorted(frames,key=lambda n:n!='cam_head'):
            frame=frames[name]
            rows = []
            for box in self.detector.detect(frame.rgb):
                rim = extract_rim(frame.rgb, box['box_xyxy'])
                if name!='cam_head' and rim['status']=='rim_observed':
                    centre,axes,angle=rim['ellipse_cv']
                    theta=np.deg2rad(angle)
                    rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                    extent=np.sqrt((rot**2)@(np.asarray(axes)/2)**2)
                    h,w=frame.rgb.shape[:2]
                    border=float(min(np.min(np.asarray(centre)-extent),np.min(np.array([w-1,h-1])-np.asarray(centre)-extent)))
                    box_margin=float(min(box['box_xyxy'][0],box['box_xyxy'][1],w-1-box['box_xyxy'][2],h-1-box['box_xyxy'][3]))
                    visibility=dict(coverage_deg=rim['coverage_deg'],occupied_bins=rim['occupied_bins'],total_bins=36,rim_border_margin_px=border,box_border_margin_px=box_margin)
                    rim['visibility']=visibility
                    if (box_margin<4 or rim['coverage_deg']<300 or rim['occupied_bins']<30
                            or np.any(np.asarray(centre)-extent<4)
                            or np.any(np.asarray(centre)+extent>np.array([w-1,h-1])-4)):
                        rim=dict(status='unknown',reason='wrist_rim_not_sufficiently_visible',
                                 coverage_deg=rim['coverage_deg'],visibility=visibility)
                row = dict(**box, rim=rim)
                # Detector often proposes nested boxes for the same rim. Merge
                # only near-identical observed contours, never distinct bowls.
                if rim['status'] == 'rim_observed':
                    centre, axes, _ = rim['ellipse_cv']
                    if any(r['rim']['status']=='rim_observed'
                           and np.linalg.norm(np.asarray(centre)-r['rim']['ellipse_cv'][0]) < 2
                           and np.max(np.abs(np.asarray(axes)/r['rim']['ellipse_cv'][1]-1)) < .06
                           for r in rows):
                        continue
                rows.append(row)
            detections[name] = rows
        head=detections.get('cam_head',[])
        if self.previous and len(head)>3:head=[]
        if not self.previous and len(head)!=3:
            self.latest={};self.step=step
            return dict(status='unknown',reason='initial_head_inventory_requires_three_distinct_detections',
                        detections=detections,instances=[],control_step=step,seconds=time.monotonic()-start)
        def centre(row):
            return np.asarray(row['rim']['ellipse_cv'][0]) if row['rim']['status']=='rim_observed' else (np.asarray(row['box_xyxy'][:2])+row['box_xyxy'][2:])/2
        head.sort(key=lambda r:centre(r)[0])
        if not self.previous:
            assignments={f'bowl_{i+1}':row for i,row in enumerate(head)}
        elif not head:
            assignments={}
        else:
            labels=list(self.previous)
            centres=np.array([centre(r) for r in head])
            prior=np.array([self.previous[k] for k in labels])
            cost=np.linalg.norm(centres[:,None]-prior[None],axis=-1)
            ranked=sorted((sum(cost[i,j] for i,j in enumerate(order)),order)
                          for order in permutations(range(3),len(head)))
            score,order=ranked[0]
            if (len(ranked)>1 and ranked[1][0]-score<15) or max(cost[i,j] for i,j in enumerate(order))>100:
                assignments={}
            else:
                assignments={labels[j]:row for j,row in zip(order,head)}
        if self.previous:
            for label,row in list(assignments.items()):
                displacement=np.linalg.norm(centre(row)-self.previous[label])
                if label in self.lost_ids:
                    # Recover an untouched stationary bowl only at its old head
                    # location, after two unambiguous detections. Never recover
                    # a held/moved bowl by assuming its old location is current.
                    old=self.inventory.get(label,{}).get('head_box_xyxy')
                    current_box=np.asarray(row['box_xyxy'],float)
                    if old is not None:
                        old=np.asarray(old,float)
                        intersection=np.prod(np.maximum(0,np.minimum(old[2:],current_box[2:])-np.maximum(old[:2],current_box[:2])))
                        union=np.prod(old[2:]-old[:2])+np.prod(current_box[2:]-current_box[:2])-intersection
                        overlap=intersection/max(union,1e-9)
                    else:overlap=0.
                    eligible=label not in self.moving_ids and displacement<=8 and overlap>=.7
                    self.recovery_hits[label]=self.recovery_hits.get(label,0)+1 if eligible else 0
                    if self.recovery_hits[label]>=2:
                        self.lost_ids.remove(label)
                        self.identity_recovery[label]='stationary_head_reacquired_twice; geometry_must_be_remeasured'
                    else:assignments.pop(label)
                elif label not in self.moving_ids and displacement>8:
                    assignments.pop(label)
            missing=set(self.previous)-set(assignments)
            for label in missing:
                if not any(label==k for k in assignments) and label not in self.lost_ids:
                    self.recovery_hits[label]=0
            self.lost_ids.update(missing)
        for label,row in assignments.items():self.previous[label]=centre(row).copy()
        instances=[]; evidence={}; accepted_pairs=[]
        config=GeometryConfig(allow_partial=True,z_bounds_m=(.765,1.2))
        from tools.perception.geometry import pixel_ray, intersect_horizontal
        for label in self.previous:
            if label not in assignments:
                old=self.inventory.get(label,{})
                instances.append(dict(bowl_id=label,status='stale',reason='head_missing_or_identity_continuity_lost; ID_not_reassigned',
                    head_box_xyxy=old.get('head_box_xyxy'),obstacle_bounds_m=old.get('obstacle_bounds_m'),
                    obstacle_observation_step=old.get('obstacle_observation_step'),
                    warning='ID retained; previous obstacle envelope retained, never a current grasp target'))
                continue
            row=assignments[label];rim=row['rim']
            box=row['box_xyxy']
            coarse=[]
            for u,v in [(box[0],box[1]),(box[2],box[1]),(box[2],box[3]),(box[0],box[3])]:
                origin,direction=pixel_ray(frames['cam_head'],[u,v])
                for z in [.765,.925]:coarse.append(intersect_horizontal(origin,direction,z))
            coarse=np.asarray(coarse)
            obstacle_bounds=np.array([coarse.min(axis=0)-.01,coarse.max(axis=0)+.01]).tolist()
            if rim['status']!='rim_observed':
                instances.append(dict(bowl_id=label,head_box_xyxy=box,detector_score=row['score'],
                    status='unknown',reason='head_rim_not_resolved',obstacle_bounds_m=obstacle_bounds,
                    obstacle_observation_step=step))
                continue
            h=RimObservation('cam_head',frames['cam_head'], RimCandidate(
                label,np.asarray(rim['contour_uv']),complete=rim['coverage_deg']>=340,clipped=rim.get('clipped',True)))
            fits=[]; rejected=[]
            for camera,rows in detections.items():
                if camera=='cam_head': continue
                for index,other in enumerate(rows):
                    r=other['rim']
                    if r['status']!='rim_observed': continue
                    candidate=RimObservation(camera,frames[camera],RimCandidate(
                        f'{camera}:{index}',np.asarray(r['contour_uv']),complete=r['coverage_deg']>=340,clipped=r.get('clipped',True)))
                    fit=fit_bowl_rim([h,candidate],config=config)
                    if fit.status=='accepted': fits.append((camera,index,fit)); evidence[(camera,index)]=candidate
                    else: rejected.append(dict(camera=camera,candidate=index,reason=fit.reason))
            item=dict(bowl_id=label,head_box_xyxy=row['box_xyxy'],head_rim=rim,
                      detector_score=row['score'],status='unknown',rejections=rejected,
                      obstacle_bounds_m=obstacle_bounds,obstacle_observation_step=step)
            # More than one feasible target in the same view is ambiguous.
            if fits and len({c for c,_,_ in fits}) == len(fits):
                locations=np.array([f.center_xyz_m for _,_,f in fits])
                radii=np.array([f.radius_m for _,_,f in fits])
                if np.max(np.linalg.norm(locations-locations[0],axis=1))<=.005 and np.ptp(radii)<=.003:
                    joint=fit_bowl_rim([h]+[evidence[(c,i)] for c,i,_ in fits],config=config) if len(fits)>1 else fits[0][2]
                    if joint.status=='accepted':
                        fitted=asdict(joint)
                        fitted.update(bowl_id=label,episode_id=self.episode_id,observation_id=self.observation_id,
                                      control_step=step,identity_continuous=True)
                        item.update(status='localized',geometry=fitted)
                        accepted_pairs.extend((label,c,i) for c,i,_ in fits)
            if fits and item['status']=='unknown': item['reason']='ambiguous_cross_view_identity'
            elif not fits: item['reason']='no_valid_cross_view_rim_fit; obtain_a_clearer_wrist_view'
            instances.append(item)
        # The same wrist candidate must not be assigned to two head instances.
        for label,c,i in accepted_pairs:
            if sum(c==c2 and i==i2 for _,c2,i2 in accepted_pairs)>1:
                item=next(r for r in instances if r['bowl_id']==label)
                item.pop('geometry',None); item.update(status='unknown',reason='shared_wrist_candidate_ambiguity')
        self.latest={r['bowl_id']:r for r in instances}; self.step=step
        for row in instances:
            if row['status']!='stale':self.inventory[row['bowl_id']]=row.copy()
        return dict(status='localized' if all(r['status']=='localized' for r in instances) else 'partial',
                    control_step=step,instances=instances,detections=detections,
                    seconds=time.monotonic()-start,
                    assumptions=['upright circular white bowl rims','calibrated simultaneous RGB',
                                 'bounds exclude calibration and shape-model error',
                                 'unknown obstacle envelope assumes table-supported bowl height <=16cm; not a grasp pose'],
                    model=self.detector.provenance)

    def current(self, label: str, step: int) -> dict:
        row=self.latest.get(label)
        if self.step != step or row is None or row['status']!='localized':
            raise ValueError('bowl_geometry_missing_or_stale; observe_bowls_from_current_views')
        return copy.deepcopy(row['geometry'])


def verify_lift(before: dict, after: dict, tcp_before: Any, tcp_after: Any) -> dict:
    """Independent observed rim-centre displacement; TCP never predicts after."""
    required={'bowl_id','episode_id','observation_id','control_step','identity_continuous'}
    if (not required<=set(before) or not required<=set(after)
            or before.get('status')!='accepted' or after.get('status')!='accepted'
            or before['bowl_id']!=after['bowl_id'] or before['episode_id']!=after['episode_id']
            or not before['identity_continuous'] or not after['identity_continuous']
            or after['observation_id']<=before['observation_id'] or after['control_step']<=before['control_step']):
        raise ValueError('fresh_same_episode_same_bowl_independent_observations_required')
    a=np.asarray(before['center_xyz_m']); b=np.asarray(after['center_xyz_m'])
    robot=np.asarray(tcp_after)-np.asarray(tcp_before)
    delta=b-a
    uncertainty=float(before['uncertainty']['center_bound_m']+after['uncertainty']['center_bound_m'])
    residual=float(np.linalg.norm(delta-robot))
    radius_error=abs(before['radius_m']-after['radius_m'])
    follows=bool(robot[2]>=.01 and delta[2]-uncertainty>=.005
                 and residual+uncertainty<=.015 and radius_error<=.003)
    return dict(follows_lift=follows,object_displacement_m=delta.tolist(),
                tcp_displacement_m=robot.tolist(),following_residual_m=residual,
                combined_conditional_uncertainty_m=uncertainty,radius_change_m=radius_error,
                native_task_success=False,
                caveat='Observed rim motion under upright-circle and identity assumptions; not stable-grasp proof.')
