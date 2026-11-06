"""Parameterized geometry tools over one immutable calibrated observation bundle.

The loader is an explicit adapter; these tools never read env/task assets, depth
predictions, simulator object poses, robot executors or native success/score.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping
from types import MappingProxyType
import numpy as np

from skills.contracts import MeasuredObservation
from tools.measurement import MeasurementResult, MeasurementTool
from execution.feedback import ObservationEvidence, OperationCheck, OperationReport
from tools.perception.geometry import (CameraFrame, GeometryUnavailable, color_regions,
    intersect_horizontal, pixel_ray, plane_sweep, triangulate)

REVISION = 'calibrated-rgb/v1'


@dataclass(frozen=True)
class CalibratedObservation:
    observation: MeasuredObservation
    cameras: Mapping[str, CameraFrame]
    world_frame: str
    masks: Mapping[str, np.ndarray]  # camera/region -> image-derived native mask

    def __post_init__(self):
        if not self.world_frame or not self.cameras:
            raise ValueError('world frame and cameras required')
        object.__setattr__(self, 'cameras', MappingProxyType(dict(self.cameras)))
        masks = {}
        for key, raw in self.masks.items():
            camera = key.split('/', 1)[0]
            value = np.array(raw, copy=True)
            if camera not in self.cameras or value.dtype != bool or value.shape != self.cameras[camera].rgb.shape[:2]:
                raise ValueError('mask must match its native camera image')
            value.setflags(write=False)
            masks[key] = value
        object.__setattr__(self, 'masks', MappingProxyType(masks))


def load_bundle(directory: str | Path) -> CalibratedObservation:
    """Read geometry.json plus native RGB images and optional .npy masks.

Paths must stay within this immutable bundle; .npy is loaded with pickle disabled.
PNG/JPEG reading requires Pillow. Metadata schema is documented in interface.md.
"""
    root = Path(directory).resolve()
    metadata = json.loads((root / 'geometry.json').read_text())
    if metadata.get('schema') != 'thinkerwam.calibrated-rgb/v1':
        raise ValueError('unsupported calibrated RGB bundle')
    def local(name):
        p = (root / name).resolve()
        if not p.is_relative_to(root):
            raise ValueError('bundle file escapes bundle directory')
        return p
    def image(name):
        p = local(name)
        if p.suffix == '.npy':
            return np.load(p, allow_pickle=False)
        from PIL import Image
        with Image.open(p) as im:
            if im.mode != 'RGB':
                raise ValueError('native RGB images must have RGB mode')
            return np.asarray(im).copy()
    cameras = {}
    for name, record in metadata['cameras'].items():
        cameras[name] = CameraFrame(image(record['rgb']), record['K'], record['T_world_camera'],
                                   record['time_s'], record['calibration_revision'], record['rectified_native'],
                                   tuple(record['calibration_image_hw']))
    masks = {key: np.load(local(value), allow_pickle=False) for key, value in metadata.get('masks', {}).items()}
    return CalibratedObservation(MeasuredObservation(**metadata['observation']), cameras,
                                 metadata['world_frame'], masks)


class GeometryTools:
    def __init__(self, loader: Callable[[MeasuredObservation], CalibratedObservation], *, max_camera_skew_s=0.02):
        if not np.isfinite(max_camera_skew_s) or max_camera_skew_s < 0:
            raise ValueError('nonnegative finite camera skew required')
        self.loader, self.max_skew = loader, max_camera_skew_s
        self.identity, self.results, self.bundle_fingerprint = None, {}, None

    def registry(self):
        specs = {
            'rgb_ray': ('原始像素转换为世界射线；不提供深度。', {'camera':'str', 'uv':'[u,v]'}),
            'triangulate_point': ('同一实测快照中，两相机对应像素三角测量；不自动验证对应关系。',
                {'camera':'str', 'uv':'[u,v]', 'camera2':'str', 'uv2':'[u,v]'}),
            'measure_plane': ('两视角显式表面区域的 RGB 水平平面测高；弱纹理或不确定时返回 unknown。',
                {'camera':'str','camera2':'str','z_min_m':'float','z_max_m':'float','step_m':'float',
                 'roi':'[x0,y0,x1,y1] or omit with mask','roi2':'[x0,y0,x1,y1] or omit with mask2',
                 'mask':'optional bundle mask name','mask2':'optional bundle mask name'}),
            'pixel_world': ('用同一观测中已测得的平面与像素射线求交；像素属于该表面仍需检查。',
                {'camera':'str','uv':'[u,v]','plane_ref':'measurement_id'}),
            'segment_color': ('原始 RGB 阈值和四连通区域；不是语义物体检测器。',
                {'camera':'str','lower_rgb':'[r,g,b]','upper_rgb':'[r,g,b]','min_area':'optional int, default 40'}),
            'region_center_on_plane': ('把颜色区域质心投到实测平面；仅表面投影，不是物体真实中心或抓取点。',
                {'region_ref':'measurement_id of segment_color','region_index':'int','plane_ref':'measurement_id'}),
        }
        return {name: MeasurementTool(name, REVISION, description, params,
                    lambda observation, args, name=name: self.measure(name, observation, args))
                for name, (description, params) in specs.items()}

    def resolve(self, reference, observation):
        """Read a previous result only within the same measured snapshot."""
        if observation != self.identity or reference not in self.results:
            raise GeometryUnavailable('stale_or_unknown_measurement_reference')
        self._bundle(observation)
        # Do not let a caller mutate cache identity or replace stored geometry.
        return json.loads(json.dumps(self.results[reference]))

    def _bundle(self, observation):
        bundle = self.loader(observation)
        if bundle.observation != observation:
            raise GeometryUnavailable('bundle_does_not_match_measured_observation')
        digest = hashlib.sha256(bundle.world_frame.encode())
        for name, camera in sorted(bundle.cameras.items()):
            digest.update(json.dumps([name, camera.calibration_revision, camera.time_s]).encode())
            for array in (camera.rgb, camera.K, camera.T_world_camera):
                digest.update(str(array.shape).encode())
                digest.update(array.tobytes())
        for key, mask in sorted(bundle.masks.items()):
            digest.update(key.encode())
            digest.update(mask.tobytes())
        stamp = digest.hexdigest()
        if self.bundle_fingerprint is not None and stamp != self.bundle_fingerprint:
            self.results.clear()
            raise GeometryUnavailable('immutable_observation_bundle_changed')
        self.bundle_fingerprint = stamp
        return bundle

    def _mask(self, bundle, camera, args, suffix=''):
        mask_key, roi_key = 'mask'+suffix, 'roi'+suffix
        if (mask_key in args) == (roi_key in args):
            raise ValueError('supply exactly one native-image mask or ROI per camera')
        if mask_key in args:
            return bundle.masks[camera+'/'+args[mask_key]]
        roi = args[roi_key]
        h, w = bundle.cameras[camera].rgb.shape[:2]
        if (not isinstance(roi, (list, tuple)) or len(roi) != 4
                or any(type(v) is not int for v in roi)
                or not (0 <= roi[0] < roi[2] <= w and 0 <= roi[1] < roi[3] <= h)):
            raise ValueError('invalid native-image ROI')
        mask = np.zeros((h,w), dtype=bool)
        mask[roi[1]:roi[3], roi[0]:roi[2]] = True
        return mask

    def measure(self, name, observation, arguments):
        args = dict(arguments)
        evidence = ObservationEvidence.from_observation(observation)
        if observation != self.identity:
            self.identity, self.results, self.bundle_fingerprint = observation, {}, None
        try:
            if name not in self.registry():
                raise ValueError('unknown geometry tool')
            allowed = self.registry()[name].parameters
            if any(key not in allowed for key in args):
                raise ValueError('unrecognized geometry argument')
            bundle = self._bundle(observation)
            selected = [args[k] for k in ('camera', 'camera2') if k in args]
            if name == 'region_center_on_plane':
                regions = self.resolve(args['region_ref'], observation)
                if regions['tool'] != 'segment_color':
                    raise ValueError('region_ref must reference segment_color')
                index = args['region_index']
                if type(index) is not int or index < 0 or index >= len(regions['regions']):
                    raise ValueError('invalid region index')
                args['camera'], args['uv'] = regions['camera'], regions['regions'][index]['centroid_px']
                selected = [args['camera']]
            cameras = [bundle.cameras[k] for k in selected]
            if any(abs(c.time_s - observation.time_s) > self.max_skew for c in cameras):
                raise GeometryUnavailable('camera_timestamp_not_aligned')
            if len(cameras) == 2 and (selected[0] == selected[1]
                    or abs(cameras[0].time_s - cameras[1].time_s) > self.max_skew):
                raise GeometryUnavailable('distinct_synchronized_cameras_required')
            payload = self._calculate(name, bundle, observation, args)
            identity = dict(tool=name, revision=REVISION, args=args, observation=vars(observation),
                            calibration={k: bundle.cameras[k].calibration_revision for k in selected})
            reference = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
            payload.update(tool=name, method_revision=REVISION, measurement_id=reference,
                           observation_id=observation.observation_id, status='ok', world_frame=bundle.world_frame,
                           length_unit='m', calibration_revisions=identity['calibration'], action_ready=False)
            self.results[reference] = json.loads(json.dumps(payload))
            # Numerical consistency never asserts target identity or physical grasp success.
            checks = (OperationCheck('measurement_computed', 'passed', REVISION, (evidence,)),
                      OperationCheck('target_surface_association', 'unknown', REVISION,
                                     detail='requires independent image/target verification'))
            return MeasurementResult(OperationReport(name, checks=checks), payload)
        except (GeometryUnavailable, ValueError, KeyError, TypeError, IndexError, OSError, np.linalg.LinAlgError) as error:
            return MeasurementResult(OperationReport(name,
                checks=(OperationCheck('measurement_computed','unknown',REVISION, detail=str(error)),),
                failure_kind='perception', failure_detail=str(error) or type(error).__name__,
                recovery_options=('check_calibration_and_selection','reobserve','request_new_view')),
                dict(tool=name, method_revision=REVISION, status='unknown', reason=str(error),
                     action_ready=False))

    def _calculate(self, name, bundle, observation, args):
        cam = bundle.cameras[args['camera']]
        if name == 'rgb_ray':
            origin, direction = pixel_ray(cam, args['uv'])
            return dict(origin_world_m=origin.tolist(), direction_world=direction.tolist(), depth_known=False)
        if name == 'triangulate_point':
            result = triangulate(cam, args['uv'], bundle.cameras[args['camera2']], args['uv2'])
            return dict(**result, correspondence_verified=False)
        if name == 'measure_plane':
            return plane_sweep(cam, bundle.cameras[args['camera2']],
                self._mask(bundle,args['camera'],args), self._mask(bundle,args['camera2'],args,'2'),
                z_min_m=float(args['z_min_m']),z_max_m=float(args['z_max_m']),step_m=float(args['step_m']))
        if name in ('pixel_world','region_center_on_plane'):
            plane = self.resolve(args['plane_ref'], observation)
            if plane['tool'] != 'measure_plane' or plane['world_frame'] != bundle.world_frame:
                raise ValueError('plane_ref must identify a current measured plane in the same frame')
            origin, direction = pixel_ray(cam, args['uv'])
            return dict(point_world_m=intersect_horizontal(origin, direction, plane['plane_z_m']).tolist(),
                        plane_ref=args['plane_ref'], plane_membership_verified=False,
                        projection_only=True, grasp_point_verified=False)
        if name == 'segment_color':
            return dict(camera=args['camera'],regions=color_regions(cam.rgb,args['lower_rgb'],args['upper_rgb'],
                        args.get('min_area',40)), semantic_identity_verified=False)
        raise ValueError('unsupported tool')


def save_bundle(directory: str | Path, bundle: CalibratedObservation) -> Path:
    """Write a new offline bundle; never overwrite existing observations."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    cameras, masks = {}, {}
    for i, (name, camera) in enumerate(sorted(bundle.cameras.items())):
        filename = f'camera-{i}.npy'
        np.save(root / filename, camera.rgb, allow_pickle=False)
        cameras[name] = dict(rgb=filename, K=camera.K.tolist(), T_world_camera=camera.T_world_camera.tolist(),
                             time_s=camera.time_s, calibration_revision=camera.calibration_revision,
                             rectified_native=camera.rectified_native, calibration_image_hw=camera.calibration_image_hw)
    for i, (name, mask) in enumerate(sorted(bundle.masks.items())):
        filename=f'mask-{i}.npy'
        np.save(root / filename, mask, allow_pickle=False)
        masks[name]=filename
    metadata=dict(schema='thinkerwam.calibrated-rgb/v1', observation=vars(bundle.observation),
                  world_frame=bundle.world_frame,cameras=cameras,masks=masks)
    (root / 'geometry.json').write_text(json.dumps(metadata,sort_keys=True,indent=2)+'\n')
    return root
