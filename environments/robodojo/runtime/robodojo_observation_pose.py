"""Compute empty-wrist viewing proposals from calibrated head evidence.

The Agent chooses a bowl ID, arm and named side. Approximate scene envelopes
are used ONLY to aim an observing camera, never as contact measurements.
Every returned TCP pose still requires the native motion/collision planner.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation


def head_approach_pose(bounds, current_tcp, quaternion_wxyz):
    """A high, non-contact waypoint; the envelope is NOT a measured rim point.

    Raise first if below clearance; otherwise move horizontally in <=30cm
    increments. Full robot/finger path checking is still mandatory at execution.
    """
    box=np.asarray(bounds,float); current=np.asarray(current_tcp,float)
    q=np.asarray(quaternion_wxyz,float)
    if (box.shape!=(2,3) or current.shape!=(3,) or q.shape!=(4,)
            or not all(np.isfinite(x).all() for x in (box,current,q))
            or np.any(box[1]<box[0]) or np.linalg.norm(q)<1e-8):
        raise ValueError('valid_head_envelope_and_measured_tcp_required')
    height=max(.95,float(box[1,2])+.08,float(current[2]))
    if height>1.4:raise ValueError('head_envelope_has_no_high_approach_clearance')
    destination=np.r_[box.mean(axis=0)[:2],height]
    if current[2]<height-.001:
        target=current.copy();target[2]=height
        stage='raise_before_horizontal_approach'
    else:
        target=destination.copy();stage='high_approach'
    delta=target-current; distance=np.linalg.norm(delta)
    if distance>.30:target=current+delta*(.30/distance)
    if target[2]<.95:raise ValueError('cannot_reach_clear_air_in_one_bounded_move')
    return dict(tcp_world_m=target.tolist(),quaternion_wxyz=(q/np.linalg.norm(q)).tolist(),
                stage=stage,contact_authorized=False,
                source='head_calibrated_conservative_envelope_not_contact_geometry',
                next_action='observe_bowls before another approach or descent; use wrist evidence near contact')


def observation_pose(bounds, camera_world, flange_world, tool_offset, viewpoint):
    offsets={'front':[0,-.14], 'left':[-.14,0], 'right':[.14,0], 'overhead':[0,-.04]}
    if viewpoint not in offsets:raise ValueError('unknown_named_viewpoint')
    box=np.asarray(bounds,float);f=np.asarray(flange_world,float);cam=np.asarray(camera_world,float)
    if box.shape!=(2,3) or not np.isfinite(box).all() or f.shape!=(7,) or cam.shape!=(4,4):
        raise ValueError('bounded_observed_envelope_and_current_camera_mount_required')
    target=box.mean(axis=0)
    rotation=Rotation.from_quat(f[[4,5,6,3]]).as_matrix()
    flange=np.eye(4);flange[:3,:3]=rotation;flange[:3,3]=f[:3]
    mount=np.linalg.inv(flange)@cam
    for height in [1.02,1.06,1.10,1.14]:
        origin=np.r_[target[:2]+offsets[viewpoint],max(height,target[2]+.18)]
        forward=target-origin;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
        down=np.cross(forward,right)
        desired=np.eye(4);desired[:3,:3]=np.column_stack([right,down,forward]);desired[:3,3]=origin
        desired_flange=desired@np.linalg.inv(mount)
        tcp=desired_flange[:3,3]+tool_offset*desired_flange[:3,0]
        if tcp[2]>=.95:
            q=Rotation.from_matrix(desired_flange[:3,:3]).as_quat()
            return dict(tcp_world_m=tcp.tolist(),quaternion_wxyz=q[[3,0,1,2]].tolist(),
                        camera_target_world_m=target.tolist(),camera_origin_world_m=origin.tolist(),
                        viewpoint=viewpoint,contact_authorized=False,
                        source='calibrated_head_obstacle_envelope_and_measured_camera_mount')
    raise ValueError('no_clear_air_viewpoint_in_bounded_height_search')
