"""Task-independent diversity using public robot FK and gripper targets."""
import itertools
import numpy as np

def select_features(positions,quaternions,grippers):
    # [candidate,time,arm,xyz/wxyz]; scale distances by fixed interpretable units.
    n=positions.shape[0];dist=np.zeros((n,n),float)
    for i,j in itertools.combinations(range(n),2):
        dp=np.linalg.norm(positions[i]-positions[j],axis=-1)/.01
        qi=quaternions[i]/np.linalg.norm(quaternions[i],axis=-1,keepdims=True)
        qj=quaternions[j]/np.linalg.norm(quaternions[j],axis=-1,keepdims=True)
        da=2*np.arccos(np.clip(np.abs(np.sum(qi*qj,axis=-1)),0,1))/.1
        dg=(grippers[i]-grippers[j])/.1
        dist[i,j]=dist[j,i]=float(np.sqrt(np.mean(dp**2)+np.mean(da**2)+np.mean(dg**2)))
    # Keep candidate0 as reference, select two by farthest-point coverage.
    selected=[0]
    while len(selected)<3:
        selected.append(max((i for i in range(n) if i not in selected),key=lambda i:(min(dist[i,j] for j in selected),-i)))
    if not np.isfinite(dist).all():raise ValueError('nonfinite diversity')
    return selected,dist

def select_native(plans,env):
    from environments.robodojo.runtime.robodojo_motion_plan import NativeMotionPlanner
    robots={r.arm_name.split('_')[0]:r for r in env.robot_manager.robot_list if r.type=='target'}
    if set(robots)!={'left','right'}:raise ValueError('expected left/right native arms')
    planners={a:NativeMotionPlanner(env,r) for a,r in robots.items()}
    poses=[]
    for plan in plans:
        candidate=[]
        for row in plan:
            candidate.append([planners[a].flange_world(row[start:start+6]) for a,start in [('left',0),('right',7)]])
        poses.append(candidate)
    poses=np.asarray(poses)
    if poses.shape!=(8,24,2,7) or not np.isfinite(poses).all():raise ValueError('invalid native FK')
    grippers=np.asarray(plans)[:,:,[6,13]]
    selected,distance=select_features(poses[...,:3],poses[...,3:],grippers)
    return selected,dict(selection='candidate0 plus deterministic farthest-point, all24times',
        metric_scales=dict(position_m=.01,orientation_rad=.1,gripper=.1),
        pairwise_distance=distance.tolist(),fk_source='native public robot FK, flange poses',
        task_outcomes_used=False)
