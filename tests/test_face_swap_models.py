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


def test_ghost_requires_five_landmarks():
    import pytest
    face = SimpleNamespace(kps=np.zeros((4, 2), dtype=np.float32))
    with pytest.raises(RuntimeError, match="five facial landmarks"):
        Ghost256Swapper.align(np.zeros((128, 128, 3), dtype=np.uint8), face)


def test_source_quality_prefers_clearer_face():
    import cv2
    from app import source_quality_score
    rng = np.random.default_rng(7)
    sharp = rng.integers(0, 256, (240, 240, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (21, 21), 0)
    face = SimpleNamespace(
        bbox=np.array([40, 40, 200, 200], dtype=np.float32),
        det_score=0.95,
    )
    assert source_quality_score(face, sharp) > source_quality_score(face, blurred)
