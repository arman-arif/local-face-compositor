import numpy as np
from types import SimpleNamespace

from face_swap_models import Ghost256Swapper


def test_ghost_alignment_returns_256_crop_and_affine():
    source = np.zeros((300, 300, 3), dtype=np.uint8)
    points = np.array([
        [100, 110], [160, 110], [130, 140], [108, 175], [153, 175]
    ], dtype=np.float32)
    face = SimpleNamespace(kps=points)
    crop, affine = Ghost256Swapper.align(source, face)
    assert crop.shape == (256, 256, 3)
    assert affine.shape == (2, 3)
    assert np.isfinite(affine).all()
