"""Training-free baseline features. This is NOT a learned semantic embedding."""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


def rgb_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert('RGB')
    if isinstance(value, (str, Path)):
        if str(value).startswith('data:image/'):
            value = BytesIO(base64.b64decode(str(value).split(',', 1)[1], validate=True))
        with Image.open(value) as image:
            return image.convert('RGB')
    if isinstance(value, bytes):
        return rgb_image(Image.open(BytesIO(value)))
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError('images must be uint8 HWC RGB, paths or inline image data URLs')
    return Image.fromarray(array)


class RGBEncoder:
    """Small spatial thumbnail + RGB histograms; deterministic and CPU-only.

    An injected frozen encoder must provide revision and encode(image)->1D array.
    Its model/preprocessing identity must be identical when building and querying.
    """
    revision = 'rgb-thumbnail8-hist16/v1'

    def encode(self, image: Any) -> np.ndarray:
        img = rgb_image(image)
        thumb = np.asarray(ImageOps.pad(img, (8, 8)), dtype=np.float32).reshape(-1) / 255
        pixels = np.asarray(img.resize((64, 64)), dtype=np.uint8)
        hist = np.concatenate([np.histogram(pixels[:, :, c], bins=16, range=(0, 256))[0]
                               for c in range(3)]).astype(np.float32) / (64 * 64)
        features = np.concatenate((thumb, hist))
        return normalize(features)


def normalize(features) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 1 or not len(features) or not np.isfinite(features).all():
        raise ValueError('encoder must return a finite nonempty 1D vector')
    norm = np.linalg.norm(features)
    if norm <= 0:
        raise ValueError('zero image embedding')
    return features / norm


def visual_features(images: dict, cameras: tuple[str, ...], encoder) -> np.ndarray:
    if any(camera not in images for camera in cameras):
        raise ValueError(f'required cameras: {cameras}')
    return np.concatenate([normalize(encoder.encode(images[c])) for c in cameras]) / np.sqrt(len(cameras))
