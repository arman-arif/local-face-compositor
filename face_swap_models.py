"""Model-specific ONNX swap adapters. GHOST is not an InSwapper drop-in."""
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

# FaceFusion's exact arcface_112_v1 warp template (normalized to 112px).
ARCFACE_112_V1 = np.array([
    [0.35473214, 0.45658929],
    [0.64526786, 0.45658929],
    [0.50000000, 0.61154464],
    [0.37913393, 0.77687500],
    [0.62086607, 0.77687500],
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
        swap_inputs = {item.name: item for item in self.swap.get_inputs()}
        if not {"source", "target"}.issubset(swap_inputs):
            raise RuntimeError(f"Unexpected GHOST input signature: {sorted(swap_inputs)}")
        source_shape = swap_inputs["source"].shape
        target_shape = swap_inputs["target"].shape
        if len(source_shape) != 2 or source_shape[-1] not in (512, "512", None):
            raise RuntimeError(f"Unexpected GHOST source tensor shape: {source_shape}")
        if len(target_shape) != 4 or target_shape[1:] != [3, 256, 256]:
            raise RuntimeError(f"Unexpected GHOST target tensor shape: {target_shape}")
        converter_inputs = self.converter.get_inputs()
        if len(converter_inputs) != 1 or converter_inputs[0].name != "input":
            raise RuntimeError("GHOST embedding converter must expose its expected 'input' tensor.")
        if len(converter_inputs[0].shape) != 2 or converter_inputs[0].shape[-1] not in (512, "512", None):
            raise RuntimeError(f"Unexpected GHOST converter input shape: {converter_inputs[0].shape}")
        self.converter_input = converter_inputs[0]

    @staticmethod
    def align(image, face):
        kps = getattr(face, "kps", None)
        if kps is None or np.asarray(kps).shape != (5, 2):
            raise RuntimeError("GHOST requires five facial landmarks.")
        affine, _ = cv2.estimateAffinePartial2D(
            np.asarray(kps, dtype=np.float32),
            ARCFACE_112_V1 * 256.0,
            method=cv2.RANSAC,
            ransacReprojThreshold=100,
        )
        if affine is None or not np.isfinite(affine).all():
            raise RuntimeError("Could not align face for GHOST.")
        crop = cv2.warpAffine(
            image, affine, (256, 256), flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_REPLICATE,
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
        # FaceFusion passes the converter's raw output to GHOST; do not L2-normalize it.

        aligned, affine = self.align(target, target_face)
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32)
        result = self.swap.run(None, {"source": converted, "target": tensor})[0]
        result = np.asarray(result, dtype=np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError("GHOST returned non-finite pixel values.")
        if result.ndim != 4:
            raise RuntimeError(f"Unexpected GHOST output shape: {result.shape}")
        if result.shape[1] == 3:
            result = result[0].transpose(1, 2, 0)
        elif result.shape[-1] == 3:
            result = result[0]
        else:
            raise RuntimeError(f"GHOST output has no 3-channel axis: {result.shape}")
        if result.shape[:2] != (256, 256):
            raise RuntimeError(f"Expected a 256x256 GHOST output crop, got {result.shape[:2]}.")
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
