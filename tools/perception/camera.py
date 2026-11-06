"""Explicit raw-camera adapter for geometry, separate from WAM tiling/resizing.

Bind only allowed RGB, calibration, camera poses derived from robot kinematics,
and acquisition timestamps. No object state or benchmark task assets are read.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import numpy as np
from tools.perception.geometry import CameraFrame
from tools.perception.tools import CalibratedObservation


@dataclass(frozen=True)
class GeometryCameraBinding:
    rgb_key: str
    intrinsics_key: str
    extrinsics_key: str
    timestamp_key: str
    calibration_revision: str
    extrinsics_convention: str  # camera_to_world | world_to_camera
    translation_unit_to_m: float
    rectified_native: bool
    calibration_image_hw: tuple
    channel_order: str = 'rgb'

    def __post_init__(self):
        if not all((self.rgb_key,self.intrinsics_key,self.extrinsics_key,self.timestamp_key,self.calibration_revision)):
            raise ValueError('all geometry bindings require explicit field identities')
        if self.extrinsics_convention not in ('camera_to_world','world_to_camera') or self.channel_order not in ('rgb','bgr'):
            raise ValueError('unsupported transform convention or RGB channel order')
        if not np.isfinite(self.translation_unit_to_m) or self.translation_unit_to_m <= 0:
            raise ValueError('translation unit scale must be positive and finite')
        if self.rectified_native is not True:
            raise ValueError('rectify images and intrinsics before using geometry adapter')


def _get(live, key):
    current=live
    for part in key.split('.'):
        if not isinstance(current,Mapping) or part not in current:
            raise KeyError('missing configured camera field: '+key)
        current=current[part]
    return current


class CalibratedRGBAdapter:
    def __init__(self, bindings: Mapping[str,GeometryCameraBinding], *, world_frame: str):
        if not bindings or not world_frame:
            raise ValueError('camera bindings and a common world frame are required')
        self.bindings,self.world_frame=dict(bindings),world_frame

    def adapt(self, live, observation):
        cameras={}
        for name,binding in self.bindings.items():
            rgb=np.asarray(_get(live,binding.rgb_key))
            if binding.channel_order=='bgr':
                rgb=rgb[...,::-1]
            transform=np.array(_get(live,binding.extrinsics_key),dtype=float,copy=True)
            if transform.shape!=(4,4):
                raise ValueError('camera extrinsics must be 4x4')
            transform[:3,3]*=binding.translation_unit_to_m
            if binding.extrinsics_convention=='world_to_camera':
                transform=np.linalg.inv(transform)
            cameras[name]=CameraFrame(rgb,_get(live,binding.intrinsics_key),transform,
                float(_get(live,binding.timestamp_key)),binding.calibration_revision,binding.rectified_native,
                binding.calibration_image_hw)
        return CalibratedObservation(observation,cameras,self.world_frame,{})
