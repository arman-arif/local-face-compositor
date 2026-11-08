"""Model-specific ONNX swap adapters. GHOST is not an InSwapper drop-in."""
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

ARCFACE_112 = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float32)


def _session(path: Path, providers: list[str]):
    if not path.is_file():
        raise RuntimeError(f"Missing model file: {path}")
    return ort.InferenceSession(str(path), providers=providers)


class Ghost256Swapper:
    model_name = "ghost_1_256"

    def __init__(self, model_dir: Path, providers: list[str]):
        self.swap = _session(model_dir / "ghost_1_256.onnx", providers)
        self.converter = _session(model_dir / "crossface_ghost.onnx", providers)
        self.providers = providers
        names = {item.name for item in self.swap.get_inputs()}
        if not {"source", "target"}.issubset(names):
            raise RuntimeError(f"Unexpected GHOST input signature: {sorted(names)}")
        if len(self.converter.get_inputs()) != 1:
            raise RuntimeError("Unexpected GHOST embedding-converter signature.")
        self.converter_input = self.converter.get_inputs()[0]

    @staticmethod
    def align(image, face):
        kps = getattr(face, "kps", None)
        if kps is None or np.asarray(kps).shape != (5, 2):
            raise RuntimeError("GHOST requires five facial landmarks.")
        affine, _ = cv2.estimateAffinePartial2D(
            np.asarray(kps, dtype=np.float32),
            ARCFACE_112 * (256.0 / 112.0),
            method=cv2.LMEDS,
        )
        if affine is None or not np.isfinite(affine).all():
            raise RuntimeError("Could not align face for GHOST.")
        crop = cv2.warpAffine(
            image, affine, (256, 256), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        return crop, affine.astype(np.float32)

    def get(self, target, target_face, source_face, paste_back=False):
        if paste_back:
            raise ValueError("GHOST adapter returns an aligned crop.")
        raw = getattr(source_face, "embedding", None)
        if raw is None:
            raw = getattr(source_face, "normed_embedding", None)
        if raw is None:
            raise RuntimeError("Source face has no ArcFace embedding.")
        embedding = np.asarray(raw, dtype=np.float32).reshape(1, -1)
        if embedding.shape != (1, 512) or not np.isfinite(embedding).all():
            raise RuntimeError("GHOST expects a finite 512-value ArcFace embedding.")
        converted = self.converter.run(None, {self.converter_input.name: embedding})[0]
        converted = np.asarray(converted, dtype=np.float32).reshape(1, -1)
        norm = float(np.linalg.norm(converted))
        if not np.isfinite(converted).all() or norm < 1e-8:
            raise RuntimeError("GHOST embedding conversion returned an invalid vector.")
        converted /= norm

        aligned, affine = self.align(target, target_face)
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32)
        result = self.swap.run(None, {"source": converted, "target": tensor})[0]
        result = np.asarray(result, dtype=np.float32)
        if result.ndim != 4:
            raise RuntimeError(f"Unexpected GHOST output shape: {result.shape}")
        if result.shape[1] == 3:
            result = result[0].transpose(1, 2, 0)
        elif result.shape[-1] == 3:
            result = result[0]
        else:
            raise RuntimeError(f"GHOST output has no 3-channel axis: {result.shape}")
        if result.shape[:2] != (256, 256):
            result = cv2.resize(result, (256, 256), interpolation=cv2.INTER_CUBIC)
        rgb_out = np.clip((result * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR), affine


def create_swapper(model_name: str, model_dir: Path, providers: list[str]):
    if model_name == "ghost_1_256":
        return Ghost256Swapper(model_dir, providers)
    if model_name == "inswapper_128":
        from insightface.model_zoo import get_model
        path = model_dir / "inswapper_128.onnx"
        if not path.is_file():
            raise RuntimeError(f"Missing model file: {path}")
        return get_model(str(path), providers=providers)
    raise RuntimeError(f"Unsupported face-swap model: {model_name}")
