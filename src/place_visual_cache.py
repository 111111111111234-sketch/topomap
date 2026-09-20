"""Bounded, content-addressed reuse of deterministic image processing only."""

from collections import OrderedDict
import hashlib

import numpy as np

from src.eval_utils_gpt_goatbench import encode_tensor2base64, make_labeled_contact_sheet


class PlaceVisualCache:
    def __init__(self, enabled=True, max_bytes=64 * 1024 * 1024):
        self.enabled = enabled
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    def _get(self, key, compute):
        if self.enabled and key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        value = compute()
        if self.enabled and len(value) <= self.max_bytes:
            while self.entries and self.bytes + len(value) > self.max_bytes:
                _, removed = self.entries.popitem(last=False)
                self.bytes -= len(removed)
            self.entries[key] = value
            self.bytes += len(value)
        return value

    def encode(self, pixels):
        pixels = np.ascontiguousarray(pixels)
        # Hash pixels rather than frame name/object identity: in-place updates,
        # replacement crops, changed boxes and changed frontier features miss.
        key = ("image", pixels.dtype.str, pixels.shape,
               hashlib.sha256(pixels.tobytes()).digest()) if self.enabled else None
        return self._get(key, lambda: encode_tensor2base64(pixels))

    def sheet(self, items):
        if self.enabled:
            digest = hashlib.sha256()
            for label, encoded in items:
                for part in (label, encoded):
                    data = part.encode("utf-8")
                    digest.update(len(data).to_bytes(8, "big"))
                    digest.update(data)
            key = ("sheet", digest.digest())
        else:
            key = None
        return self._get(key, lambda: make_labeled_contact_sheet(items))
