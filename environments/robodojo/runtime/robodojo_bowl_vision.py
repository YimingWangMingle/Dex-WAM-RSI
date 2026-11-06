"""Independent RGB bowl detection and visible white-rim extraction.

Grounding DINO supplies semantic boxes. The explicit white, upright bowl prior
is task-specific; it is NOT a general grasp perception model. No simulator
depth, segmentation, object transforms, or Agent-picked pixels are consumed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np


class BowlDetector:
    def __init__(self, model_path: str, device: str = "cuda:0") -> None:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.torch = torch
        self.device = device
        root = Path(model_path).resolve(strict=True)
        self.provenance = json.loads((root / "source-manifest.json").read_text())
        self.processor = AutoProcessor.from_pretrained(root, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            root, local_files_only=True).to(device).eval()

    def detect(self, rgb: np.ndarray) -> list[dict[str, Any]]:
        from PIL import Image
        image = Image.fromarray(rgb)
        inputs = self.processor(images=image, text="a bowl.", return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        # transformers 4.53 uses threshold, unlike older box_threshold versions.
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=.18, text_threshold=.18,
            target_sizes=[rgb.shape[:2]])[0]
        rows = []
        for box, score in zip(results["boxes"].cpu().numpy(), results["scores"].cpu().numpy()):
            if not np.isfinite(box).all() or not np.isfinite(score):
                continue
            box[[0, 2]] = np.clip(box[[0, 2]], 0, image.width-1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, image.height-1)
            if np.min(box[2:]-box[:2]) >= 15:
                rows.append(dict(box_xyxy=box.tolist(), score=float(score)))
        kept = []
        for row in sorted(rows, key=lambda r: -r['score']):
            a = np.array(row['box_xyxy'])
            duplicate = False
            for old in kept:
                b = np.array(old['box_xyxy'])
                inter = np.prod(np.maximum(0, np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
                union = np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter
                if inter / max(union, 1) > .55:
                    duplicate = True
            if not duplicate:
                kept.append(row)
        return kept[:8]


def _ellipse_coordinates(points: np.ndarray, ellipse: tuple) -> tuple[np.ndarray, np.ndarray]:
    centre, axes, angle = ellipse
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    local = (points - centre) @ rotation
    normalized = local / (np.asarray(axes)/2)
    radial = np.linalg.norm(normalized, axis=1)
    # Normal-distance approximation in pixels. Multiplying by the minor axis
    # underestimates errors near the ends of elongated wrist-view ellipses.
    gradient = 2*np.linalg.norm(local/(np.asarray(axes)/2)**2,axis=1)
    error = np.abs(radial**2-1)/np.maximum(gradient,1e-9)
    return error, np.arctan2(normalized[:,1], normalized[:,0])


def extract_rim(rgb: np.ndarray, box_xyxy: list[float]) -> dict[str, Any]:
    """Return observed contour samples or unknown; never complete hidden arcs."""
    import cv2
    h, w = rgb.shape[:2]
    box = np.asarray(box_xyxy, float)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        return dict(status='unknown', reason='invalid_detection_box')
    x0,y0 = np.maximum(0,np.floor(box[:2]-3).astype(int))
    x1,y1 = np.minimum([w,h],np.ceil(box[2:]+4).astype(int))
    hsv = cv2.cvtColor(rgb[y0:y1,x0:x1], cv2.COLOR_RGB2HSV)
    # White ceramic lip in this task, explicitly not a semantic segmentation model.
    mask = ((hsv[...,1] < 45) & (hsv[...,2] > 145)).astype(np.uint8)*255
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours:
        return dict(status='unknown',reason='no_visible_white_rim')
    contour = max(contours,key=cv2.contourArea)
    points = contour[:,0,:].astype(float)+[x0,y0]
    if len(points)<40 or cv2.contourArea(contour)<150:
        return dict(status='unknown',reason='insufficient_rim_pixels')
    # Exclude artificial boundaries created by image/crop truncation.
    genuine = (points[:,0]>x0+1)&(points[:,0]<x1-2)&(points[:,1]>y0+1)&(points[:,1]<y1-2)
    points=points[genuine]
    if len(points)<40:
        return dict(status='unknown',reason='rim_clipped_or_occluded')
    sample=points[np.linspace(0,len(points)-1,min(len(points),300)).astype(int)]
    rng=np.random.default_rng(8173)
    candidates=[]
    for subset in [sample,*[sample[rng.choice(len(sample),8,replace=False)] for _ in range(80)]]:
        try:
            ellipse=cv2.fitEllipse(subset.astype(np.float32))
            centre,axes,_=ellipse
            if min(axes)<12 or max(axes)>2*max(x1-x0,y1-y0) or min(axes)/max(axes)<.15:
                continue
            if not (x0<centre[0]<x1 and y0<centre[1]<y1):
                continue
            error,_=_ellipse_coordinates(sample,ellipse)
            candidates.append((int((error<1.8).sum()),ellipse))
        except cv2.error:
            continue
    if not candidates:
        return dict(status='unknown',reason='no_consistent_rim_ellipse')
    _,ellipse=max(candidates,key=lambda x:x[0])
    for _ in range(3):
        error,_=_ellipse_coordinates(sample,ellipse)
        inliers=sample[error<1.8]
        if len(inliers)<30: break
        ellipse=cv2.fitEllipse(inliers.astype(np.float32))
    error,angle=_ellipse_coordinates(sample,ellipse)
    selected=error<1.8
    angles=np.sort(angle[selected])
    coverage=float(360-np.rad2deg(np.max(np.diff(np.r_[angles,angles[:1]+2*np.pi])))) if len(angles) else 0
    # Check occupied bins too: a pair of tiny disjoint arcs is not full coverage.
    occupied=len(np.unique(np.floor((angles+np.pi)/(2*np.pi)*36).astype(int)))
    if selected.sum()<40 or coverage<250 or occupied<25:
        return dict(status='unknown',reason='insufficient_visible_rim_coverage',coverage_deg=coverage,occupied_bins=occupied)
    return dict(status='rim_observed',contour_uv=sample[selected].tolist(),
                raw_mask_boundary=True,clipped=bool(np.any(sample[selected]<2) or np.any(sample[selected]>[w-3,h-3])),
                ellipse_cv=ellipse,coverage_deg=coverage,occupied_bins=occupied,
                contour_fit_p95_px=float(np.percentile(error[selected],95)),
                assumption='visible white outer lip of an upright circular bowl; no hidden arc samples')
