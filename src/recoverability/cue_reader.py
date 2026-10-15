"""Fixed RGB-only diagnostic reader, scoped to the calibrated leftmost probe ROI.

It is deliberately simple and shared by K/D/R. No task object, actor pose, oracle,
ground-truth colour, filename, seed or branch outcome is accepted by this API.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ReadResult:
    answer: str | None
    pixels: int
    confidence: float


def read_colour(rgb, roi=(0.04, 0.45, 0.26, 0.95), min_pixels=12, min_confidence=0.75):
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError('Expected a uint8 RGB image')
    h, w = image.shape[:2]
    x0,y0,x1,y1 = roi
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError('Invalid calibrated ROI')
    patch = image[int(y0*h):int(y1*h),int(x0*w):int(x1*w)].astype(np.float32)
    r,g,b = np.moveaxis(patch, -1, 0)
    masks = {
        'red': (r>60)&(r>1.6*g)&(r>1.6*b),
        'green': (g>60)&(g>1.6*r)&(g>1.6*b),
        'blue': (b>60)&(b>1.6*r)&(b>1.6*g),
        'yellow': (r>60)&(g>60)&(r>1.8*b)&(g>1.8*b)&(r<1.6*g)&(g<1.6*r),
    }
    counts = {name:int(mask.sum()) for name,mask in masks.items()}
    answer = max(counts, key=counts.get)
    pixels = counts[answer]
    confidence = pixels / max(sum(counts.values()), 1)
    return ReadResult(answer if pixels>=min_pixels and confidence>=min_confidence else None,
                      pixels, confidence)


class ProbeMemory:
    """The only persistent evidence state used by the diagnostic answer controller."""
    def __init__(self, image):
        self.image = np.array(image, copy=True)
    def drop(self):
        self.image = None
    def answer(self):
        return None if self.image is None else read_colour(self.image).answer
    @property
    def nbytes(self):
        return 0 if self.image is None else self.image.nbytes
