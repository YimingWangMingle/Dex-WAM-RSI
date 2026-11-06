"""Bounded, per-episode cache of immutable sparse RGB frame VAE latents.

No temporal-VAE recent clip, RoPE, time/state embedding, attention or KV is
cached here. Callers clear this cache on episode/model/preprocessing changes.
"""
from collections import OrderedDict

import torch

from thinker_wm.models.temporal_training import encode_history_frames


class SparseVAECache:
    def __init__(self, capacity=32):
        if int(capacity) < 1:
            raise ValueError("sparse VAE cache capacity must be positive")
        self.capacity = int(capacity)
        self.clear()

    def clear(self):
        self.entries = OrderedDict()
        self.template = None
        self.hits = self.misses = self.encoded_images = 0
        self.reused_images = self.full_encode_calls = 0

    def encode(self, model, pixels, valid, rows):
        if model.training:
            raise ValueError("sparse VAE cache requires eval mode")
        if pixels.shape[0] != 1 or valid.shape != pixels.shape[:2]:
            raise ValueError("runtime sparse cache expects one episode")
        slots = pixels.shape[1]
        if slots > self.capacity or len(rows) != slots:
            raise ValueError("sparse cache capacity/row count does not cover the slots")
        positions = valid[0].nonzero().flatten().tolist()
        valid_only = bool(getattr(model.config, "history_encode_valid_only", False))
        # Original-size encoding preserves the deployment's convolution batch
        # shape. Slot position is part of this conservative cache's identity.
        keys = {i: (int(rows[i]), tuple(pixels.shape[2:]), None if valid_only else i)
                for i in positions}
        missing = [i for i in positions if keys[i] not in self.entries]
        self.hits += len(positions) - len(missing)
        self.misses += len(missing)
        self.reused_images += (len(positions) - len(missing) if valid_only
                               else len(positions) if not missing else 0)
        if missing:
            if valid_only:
                px = pixels[:, missing].to(model.device)
                v = torch.ones(1, len(missing), dtype=torch.bool, device=model.device)
                lat = encode_history_frames(model.encode_video, px, v, model.dtype, valid_only=True)
                encoded_positions = missing
                latent_positions = range(len(missing))
            else:
                self.full_encode_calls += 1
                lat = encode_history_frames(model.encode_video, pixels.to(model.device),
                                            valid.to(model.device), model.dtype, valid_only=False)
                encoded_positions = positions
                latent_positions = positions
            self.template = lat[:, :, :1].detach().clone()
            self.encoded_images += len(missing) if valid_only else slots
            for j, i in zip(latent_positions, encoded_positions):
                self.entries[keys[i]] = lat[:, :, j:j + 1].detach().clone()
        if self.template is None:
            # Shape probe only; never interpret the blank frame as evidence.
            px = pixels[:, :1].to(model.device)
            v = torch.zeros(1, 1, dtype=torch.bool, device=model.device)
            self.template = encode_history_frames(model.encode_video, px, v, model.dtype,
                                                  valid_only=True).detach()
            self.encoded_images += 1
        result = self.template.new_zeros((1, self.template.shape[1], slots,
                                          *self.template.shape[3:]))
        for i in positions:
            result[:, :, i:i + 1] = self.entries[keys[i]]
            self.entries.move_to_end(keys[i])
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return result

    def stats(self):
        return dict(hits=self.hits, misses=self.misses, encoded_images=self.encoded_images,
                    reused_images=self.reused_images, full_encode_calls=self.full_encode_calls,
                    entries=len(self.entries), capacity=self.capacity)
