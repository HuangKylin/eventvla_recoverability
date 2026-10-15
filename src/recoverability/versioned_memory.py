"""Capacity-bounded images with public version metadata; no hidden cue summaries."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from recoverability.fixed_skills import SLOT_ROIS
from recoverability.cue_reader import read_colour


@dataclass
class Evidence:
    slot: int
    version: int
    ordinal: int
    image: np.ndarray


class VersionedMemory:
    METHODS = {'fifo', 'uniform', 'oldest', 'action_aware', 'archive', 'structured'}
    def __init__(self, method, capacity, seed=42):
        if method not in self.METHODS:
            raise ValueError('Unknown memory method')
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
            raise ValueError('Capacity must be a nonnegative integer')
        self.method, self.capacity, self.seed = method, capacity, seed
        self.entries = []
        self.semantic = {}  # Only the explicitly unbounded structured baseline uses this.
        self.versions = [0] * 4  # Public apparatus state; contains no colour information.
        self.ordinal = 0

    def proposed_advance(self, slots):
        for slot in slots:
            if slot not in range(4) or self.versions[slot] != 0:
                raise ValueError('Invalid public advance command')
        for slot in slots:
            self.versions[slot] = 1

    def observe(self, slot, version, image):
        if slot not in range(4) or version != self.versions[slot]:
            raise ValueError('Observation does not match public apparatus version')
        if np.asarray(image).dtype != np.uint8:
            raise ValueError('Expected uint8 evidence')
        old_keys = set(self.keys())
        if self.method == 'structured':
            self.semantic[(slot, version)] = read_colour(image, roi=SLOT_ROIS[slot]).answer
        else:
            self.entries = [e for e in self.entries if (e.slot, e.version) != (slot, version)]
            self.entries.append(Evidence(slot, version, self.ordinal, np.array(image, copy=True)))
            def priority(e):
                if self.method == 'oldest':
                    return (-e.ordinal,)
                if self.method == 'action_aware':
                    return (e.version < self.versions[e.slot], e.ordinal)
                if self.method == 'uniform':
                    return (hashlib.blake2b(f'{self.seed}:{e.slot}:{e.version}'.encode(), digest_size=16).digest(),)
                return (e.ordinal,)
            if self.method != 'archive':
                self.entries = sorted(self.entries, key=priority, reverse=True)[:self.capacity]
        self.ordinal += 1
        return dict(retained_keys=self.keys(), dropped_keys=sorted(old_keys-set(self.keys())), **self.storage())

    def keys(self):
        return sorted(self.semantic) if self.method == 'structured' else sorted((e.slot,e.version) for e in self.entries)

    def answer(self, slot, version):
        if self.method == 'structured':
            return self.semantic.get((slot,version)), 0
        for e in self.entries:
            if (e.slot,e.version) == (slot,version):
                return read_colour(e.image, roi=SLOT_ROIS[slot]).answer, 1
        return None, 0

    def recovery_possible_from_public_state(self, slot, version):
        return version == self.versions[slot]

    def storage(self):
        metadata = dict(method=self.method, capacity=self.capacity, seed=self.seed,
                        versions=self.versions, ordinal=self.ordinal,
                        entries=[(e.slot,e.version,e.ordinal) for e in self.entries],
                        semantic=[(slot,version,value) for (slot,version),value in sorted(self.semantic.items())])
        return dict(image_count=len(self.entries), image_payload_bytes=sum(e.image.nbytes for e in self.entries),
                    metadata_payload_bytes=len(json.dumps(metadata,separators=(',',':')).encode()),
                    bounded=self.method not in {'archive','structured'})
