"""Observed bowl bounds and public-robot finger geometry, without object GT.

Bowl bounds deliberately enclose the cavity. Only a bounded target rim region
can admit finger contact; the arm never receives that exemption. This is a
conservative rejection check, not a claim to reconstruct a bowl's contact mesh.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from environments.robodojo.runtime.robodojo_motion_plan import PlanRejected


def fit_bowl(points, bottom):
    p, b = np.asarray(points, float), np.asarray(bottom, float)
    if p.ndim != 2 or p.shape[1] != 3 or not 4 <= len(p) <= 12 or b.shape != (3,) or not np.isfinite(p).all() or not np.isfinite(b).all():
        raise ValueError('four_or_more_distributed_outer_rim_points_and_one_visible_bottom_point_required')
    if np.ptp(p[:, 2]) > .008:
        raise ValueError('rim_not_upright_or_correspondences_inconsistent')
    xy = p[:, :2]
    origin = xy.mean(axis=0)
    x = xy-origin
    design = np.column_stack([2*x, np.ones(len(x))])
    if np.linalg.cond(design) > 150:
        raise ValueError('rim_points_insufficiently_distributed')
    sol = np.linalg.lstsq(design, np.sum(x*x, axis=1), rcond=None)[0]
    centre = origin+sol[:2]
    radii = np.linalg.norm(xy-centre, axis=1)
    radius = float(radii.mean())
    angles = np.sort(np.arctan2(xy[:,1]-centre[1], xy[:,0]-centre[0]))
    if np.max(np.diff(np.r_[angles, angles[0]+2*np.pi])) > np.deg2rad(160):
        raise ValueError('rim_points_must_cover_full_circumference')
    top = float(np.mean(p[:,2]))
    if not .02 <= radius <= .18 or np.max(np.abs(radii-radius)) > .004 or not .015 <= top-b[2] <= .16 or np.linalg.norm(b[:2]-centre) > radius+.005:
        raise ValueError('inconsistent_bowl_dimensions_or_correspondences')
    margin = .005
    low = np.r_[centre-radius-margin, b[2]-margin]
    high = np.r_[centre+radius+margin, top+margin]
    return dict(centre_xy=centre.tolist(), radius_m=radius, top_z_m=top,
                bottom_z_m=float(b[2]), bounds=[low.tolist(),high.tolist()],
                margin_m=margin, circular_upright_bowl_assumption=True,
                warning='Conservative observed bounds, not a verified bowl surface mesh or metric accuracy guarantee.')


class ObservedBowlScene:
    def __init__(self):
        self.bowls = {}
        self.revision = 0

    def register(self, label, rim_ids, bottom_id, geometry, step):
        if not isinstance(label,str) or not label.strip() or len(set(rim_ids)) != len(rim_ids) or bottom_id in rim_ids:
            raise ValueError('unique_named_bowl_and_distinct_measurements_required')
        if label not in self.bowls and len(self.bowls) >= 3:
            raise ValueError('three_bowl_inventory_only; update_existing_label')
        rows = [geometry.current(i,step) for i in [*rim_ids,bottom_id]]
        record=fit_bowl([r['point_world_m'] for r in rows[:-1]],rows[-1]['point_world_m'])
        record.update(label=label, measurement_ids=[*rim_ids,bottom_id], measured_step=step,
                      stale=False, source='current_multiview_RGB_correspondences')
        self.bowls[label]=record
        self.revision+=1
        return record

    def require_complete(self):
        if len(self.bowls)!=3 or any(r['stale'] for r in self.bowls.values()):
            raise ValueError('measure_all_three_bowls_before_contact; missing_or_stale_collision_bounds')

    def register_visual_rims(self, estimates, step, *, support_plane_z=.765):
        """Atomically install three independently fitted rims, with provenance.

        The bottom is a conservative public support-plane bound, NOT a measured
        bottom pixel. Extending every bound to the table also covers elevated
        objects but can block otherwise feasible nesting; never infer cavities.
        """
        if len(estimates)!=3 or len(set(estimates))!=3:
            raise ValueError('three_independently_localized_bowls_required')
        records={}
        for label, estimate in estimates.items():
            if estimate.get('status')!='accepted':
                raise ValueError('accepted_multiview_geometry_required')
            centre=np.asarray(estimate['center_xyz_m'],float)
            radius=float(estimate['radius_m'])
            uncertainty=estimate['uncertainty']
            margin=.005+max(float(uncertainty['center_bound_m']),float(uncertainty['radius_bound_m']))
            if (centre.shape!=(3,) or not np.isfinite(centre).all() or not .02<=radius<=.18
                    or not np.isfinite(margin) or margin>.01
                    or not support_plane_z+.015<=centre[2]<=1.2):
                raise ValueError('visual_bowl_geometry_outside_contract')
            low=np.r_[centre[:2]-radius-margin,support_plane_z-.005]
            high=np.r_[centre[:2]+radius+margin,centre[2]+margin]
            records[label]=dict(label=label,centre_xy=centre[:2].tolist(),radius_m=radius,
                top_z_m=float(centre[2]),bottom_z_m=support_plane_z,bounds=[low.tolist(),high.tolist()],
                margin_m=margin,measured_step=step,stale=False,measurement_ids=[],
                source='independent_multiview_rim_fit_with_public_support_plane_bound',
                bottom_is_observed=False,geometry_evidence=estimate,
                warning='Solid conservative envelope to public table; cavity and supported bottom not observed.')
        for first in records:
            for second in records:
                if first<second:
                    a,b=records[first],records[second]
                    if np.linalg.norm(np.asarray(a['centre_xy'])-b['centre_xy'])<.015 and abs(a['top_z_m']-b['top_z_m'])<.01:
                        raise ValueError('duplicate_visual_bowl_geometry')
        self.bowls=records
        self.revision+=1
        return records

    def target(self, xyz):
        self.require_complete()
        p=np.asarray(xyz,float)
        candidates=[name for name,r in self.bowls.items()
                    if r.get('geometry_status','localized')=='localized'
                    if abs(np.linalg.norm(p[:2]-r['centre_xy'])-r['radius_m'])<=.012
                    and abs(p[2]-r['top_z_m'])<=.012]
        if len(candidates)!=1:
            raise ValueError('measured_point_must_identify_one_registered_bowl_rim')
        return candidates[0]

    def require_target(self, label, step):
        self.require_complete()  # complete OBSTACLE inventory, not three grasp poses
        row=self.bowls.get(label)
        if row is None or row.get('geometry_status')!='localized' or row['measured_step']!=step:
            raise ValueError('selected_bowl_must_be_localized_in_current_observation')

    def update_visual_instances(self, instances, step, *, held_labels=()):
        """Authorize each target separately, while retaining other obstacles.

        Unknown instances get the calibrated head-box frustum envelope over the
        explicit public supported-object height prior. Lost instances retain
        their previous obstacle envelope and stale ID, never contact authority.
        """
        pending={}
        for row in instances:
            label=row['bowl_id']; state=row['status']
            if label in held_labels and label in self.bowls:
                pending[label]=self.bowls[label].copy()
                continue  # current_boxes applies the grasp-relative transform
            bounds=row.get('obstacle_bounds_m')
            if bounds is None:continue
            box=np.asarray(bounds,float)
            if box.shape!=(2,3) or not np.isfinite(box).all() or np.any(box[1]<=box[0]):
                raise ValueError('finite_nonempty_obstacle_envelope_required')
            record=dict(label=label,bounds=box.tolist(),geometry_status=state,
                measured_step=step if state!='stale' else row.get('obstacle_observation_step'),
                stale=False,measurement_ids=[],source='head_detection_frustum_with_public_height_bounds',
                bottom_is_observed=False,obstacle_evidence_stale=state=='stale',
                warning='Conservative box envelope assumes table support and <=16cm height; not a grasp pose')
            if state=='localized':
                fit=row['geometry'];centre=np.asarray(fit['center_xyz_m'],float);radius=float(fit['radius_m'])
                u=fit['uncertainty'];margin=.005+max(u['center_bound_m'],u['radius_bound_m'])
                if fit['status']!='accepted' or not .02<=radius<=.18 or margin>.01:
                    raise ValueError('accepted_bounded_geometry_required')
                low=np.r_[centre[:2]-radius-margin,.760]
                high=np.r_[centre[:2]+radius+margin,centre[2]+margin]
                record.update(centre_xy=centre[:2].tolist(),radius_m=radius,top_z_m=float(centre[2]),
                    bottom_z_m=.765,bounds=[low.tolist(),high.tolist()],margin_m=margin,
                    geometry_evidence=fit,source='independent_multiview_rim_with_public_support_envelope')
            pending[label]=record
        if len(pending)!=3:
            raise ValueError('three_obstacle_envelopes_required; unknown_target_is_allowed_but_missing_obstacle_is_not')
        self.bowls=pending;self.revision+=1
        return pending

    def boxes(self):
        if any(r['stale'] for r in self.bowls.values()):
            raise ValueError('stale_bowl_collision_bounds; remeasure_before_motion')
        return {name:np.asarray(r['bounds']) for name,r in self.bowls.items()}

    def invalidate(self, label):
        self.bowls[label]['stale']=True
        self.revision+=1

    def refresh(self,label,old_ids,new_ids,geometry,step):
        if label not in self.bowls or len(old_ids)!=3 or len(new_ids)!=3 or len(set(old_ids))!=3 or len(set(new_ids))!=3:
            raise ValueError('three_distinct_same_feature_pairs_required')
        old=[geometry.points.get(i) for i in old_ids]
        new=[geometry.current(i,step) for i in new_ids]
        if any(r is None for r in old) or not set(old_ids)<=set(self.bowls[label]['measurement_ids']):
            raise ValueError('reference_points_must_belong_to_registered_bowl')
        if any(a['label']!=b['label'] or a['control_step']>=step for a,b in zip(old,new)):
            raise ValueError('match_the_same_three_named_physical_features_across_time')
        before=np.asarray([r['point_world_m'] for r in old]); after=np.asarray([r['point_world_m'] for r in new])
        if np.linalg.norm(np.cross(before[1]-before[0],before[2]-before[0]))<.001:
            raise ValueError('reference_features_too_close_or_collinear')
        shift=np.mean(after-before,axis=0)
        if np.max(np.linalg.norm(after-before-shift,axis=1))>.004:
            raise ValueError('bowl_rotated_deformed_or_mismatched; full_reregistration_required')
        r=self.bowls[label]
        # Translation is applied from the geometry's own reference epoch.
        if any(a['control_step']!=r['measured_step'] for a in old):
            raise ValueError('reference_epoch_mismatch')
        r['bounds']=(np.asarray(r['bounds'])+shift).tolist()
        r['centre_xy']=(np.asarray(r['centre_xy'])+shift[:2]).tolist()
        r['top_z_m']+=float(shift[2]); r['bottom_z_m']+=float(shift[2])
        r.update(stale=False,measured_step=step,measurement_ids=list(new_ids))
        self.revision+=1
        return r


def sphere_box_hits(spheres, bounds):
    s=np.asarray(spheres,float); lo,hi=np.asarray(bounds,float)
    closest=np.clip(s[:,:3],lo,hi)
    return np.linalg.norm(s[:,:3]-closest,axis=1) <= s[:,3]


def check_fingers(spheres, boxes, *, contact=None, table_boxes=(), other_arm=()):
    s=np.asarray(spheres,float)
    if s.ndim!=2 or s.shape[1]!=4 or not len(s) or not np.isfinite(s).all() or np.any(s[:,3]<=0):
        raise PlanRejected('finger_collision_geometry_unavailable')
    for label,box in boxes.items():
        hits=sphere_box_hits(s,box)
        if contact is not None and contact['label']==label:
            # Exempt only complete small finger cells inside a measured rim region.
            permitted=np.linalg.norm(s[:,:3]-contact['point'],axis=1)+s[:,3]<=.03
            hits &= ~permitted
        if hits.any():
            raise PlanRejected('finger_path_intersects_observed_bowl',dict(bowl=label,contact_region_only=True))
    for box in table_boxes:
        if sphere_box_hits(s,box).any():
            raise PlanRejected('finger_path_intersects_table')
    other=np.asarray(other_arm,float)
    if other.size:
        low=(s[:,:3]-s[:,3:]).min(axis=0)
        high=(s[:,:3]+s[:,3:]).max(axis=0)
        other=other[np.all(other[:,:3]+other[:,3:]>=low,axis=1)&np.all(other[:,:3]-other[:,3:]<=high,axis=1)]
        for chunk in np.array_split(s,max(1,int(np.ceil(len(s)/128)))):
            gap=np.linalg.norm(chunk[:,None,:3]-other[None,:,:3],axis=2)-chunk[:,None,3]-other[None,:,3]
            if np.any(gap<=0):
                raise PlanRejected('finger_path_intersects_other_arm')


class FingerGeometry:
    """Voxel covers from the robot's published URDF collision meshes.

    A cover includes cell half-diagonal plus 1mm. It is intentionally larger
    than the mesh; it may reject feasible paths. No task object is queried.
    """
    def __init__(self, urdf):
        import trimesh
        path=Path(urdf); root=ET.parse(path).getroot()
        self.parts=[]
        for name in ('joint7','joint8'):
            joint=root.find(f"joint[@name='{name}']")
            if joint is None or joint.get('type')!='prismatic' or joint.find('parent').get('link')!='link6':
                raise ValueError('unsupported_finger_kinematics')
            origin=joint.find('origin')
            translation=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            rotation=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' '))
            axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
            link=root.find(f"link[@name='{joint.find('child').get('link')}']")
            collision=link.find('collision'); mesh=collision.find('geometry/mesh')
            asset=Path(mesh.get('filename'))
            if not asset.is_absolute(): asset=path.parent/asset
            shape=trimesh.load(asset,force='mesh')
            if mesh.get('scale'): shape.apply_scale(np.fromstring(mesh.get('scale'),sep=' '))
            co=collision.find('origin')
            cr=Rotation.from_euler('xyz',np.fromstring(co.get('rpy','0 0 0'),sep=' ')) if co is not None else Rotation.identity()
            ct=np.fromstring(co.get('xyz','0 0 0'),sep=' ') if co is not None else np.zeros(3)
            points=cr.apply(shape.voxelized(pitch=.004).fill().points)+ct
            self.parts.append((translation,rotation,axis,points))
        self.radius=float(np.sqrt(3)*.002+.001)

    def spheres(self, flange, finger_joints):
        f,j=np.asarray(flange,float),np.asarray(finger_joints,float)
        if f.shape!=(7,) or j.shape!=(2,) or not np.isfinite(f).all() or not np.isfinite(j).all():
            raise ValueError('measured_finger_joints_and_flange_required')
        r=Rotation.from_quat(f[[4,5,6,3]])
        result=[]
        for index,(t,q,axis,p) in enumerate(self.parts):
            world=r.apply(q.apply(p+axis*j[index])+t)+f[:3]
            result.append(np.column_stack([world,np.full(len(world),self.radius)]))
        return np.concatenate(result)

    def boxes(self, flange, finger_joints):
        s=self.spheres(flange,finger_joints)
        sizes=[len(part[3]) for part in self.parts]
        a,b=np.split(s,[sizes[0]])
        return [np.array([(part[:,:3]-part[:,3:]).min(axis=0),(part[:,:3]+part[:,3:]).max(axis=0)]) for part in (a,b)]
