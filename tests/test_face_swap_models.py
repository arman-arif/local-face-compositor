import cv2
import numpy as np
import pytest
from types import SimpleNamespace

from face_swap_models import ARCFACE_112_V1, Ghost256Swapper


def test_ghost_alignment_uses_facefusion_template():
    target_points = ARCFACE_112_V1 * 256.0
    scale, offset = 1.2, np.array([17.0, 23.0], dtype=np.float32)
    source_points = (target_points - offset) / scale
    source = np.zeros((300, 300, 3), dtype=np.uint8)
    face = SimpleNamespace(kps=source_points.astype(np.float32))
    crop, affine = Ghost256Swapper.align(source, face)
    aligned_points = cv2.transform(source_points.reshape(1, -1, 2), affine).reshape(-1, 2)
    assert crop.shape == (256, 256, 3)
    assert affine.shape == (2, 3)
    assert np.isfinite(affine).all()
    assert np.allclose(aligned_points, target_points, atol=1.0)


def test_ghost_requires_five_landmarks():
    face = SimpleNamespace(kps=np.zeros((4, 2), dtype=np.float32))
    with pytest.raises(RuntimeError, match="five facial landmarks"):
        Ghost256Swapper.align(np.zeros((128, 128, 3), dtype=np.uint8), face)


def test_ghost_passes_raw_converter_output_to_swapper(monkeypatch):
    class FakeSession:
        def __init__(self, result):
            self.result = result
            self.feed = None

        def run(self, _outputs, feed):
            self.feed = feed
            return [self.result]

    adapter = Ghost256Swapper.__new__(Ghost256Swapper)
    converter = FakeSession(np.array([[3.0, 4.0]], dtype=np.float32))
    swap = FakeSession(np.zeros((1, 3, 256, 256), dtype=np.float32))
    adapter.converter = converter
    adapter.swap = swap
    adapter.converter_input = SimpleNamespace(name="input")
    monkeypatch.setattr(
        Ghost256Swapper, "align",
        staticmethod(lambda image, face: (
            np.zeros((256, 256, 3), dtype=np.uint8),
            np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        )),
    )
    target = np.zeros((300, 300, 3), dtype=np.uint8)
    face = SimpleNamespace(embedding=np.ones(512, dtype=np.float32))
    adapter.get(target, SimpleNamespace(), face)
    assert np.array_equal(swap.feed["source"], np.array([[3.0, 4.0]], dtype=np.float32))
    assert swap.feed["target"].shape == (1, 3, 256, 256)
    assert np.allclose(swap.feed["target"], -1.0)


def test_source_quality_prefers_clearer_face():
    from app import source_quality_score
    rng = np.random.default_rng(7)
    sharp = rng.integers(0, 256, (240, 240, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (21, 21), 0)
    face = SimpleNamespace(
        bbox=np.array([40, 40, 200, 200], dtype=np.float32),
        det_score=0.95,
    )
    assert source_quality_score(face, sharp) > source_quality_score(face, blurred)
