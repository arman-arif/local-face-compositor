import shutil
import threading
import uuid
from types import SimpleNamespace
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
OUTPUTS = DATA / "outputs"
MODELS = ROOT / "models"
INSIGHTFACE_ROOT = ROOT / ".insightface"
for folder in (UPLOADS, OUTPUTS, MODELS):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Local Face Compositor", version="0.2.0")
app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_SIDE = 10000

# Expensive model initialization is cached for later requests.
_MODEL_LOCK = threading.Lock()
_FACE_APP = None
_SWAPPERS = {}
_PROVIDER_NAMES: list[str] = []
_GFP_GANER = None
_RESTORER_LOCK = threading.Lock()


async def save_image(upload: UploadFile, folder: Path) -> Path:
    extension = Path(upload.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Use JPG, PNG, or WebP images.")
    content = await upload.read()
    if not content:
        raise HTTPException(400, "An uploaded image is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Each image must be 25 MB or smaller.")

    path = folder / f"{uuid.uuid4().hex}{extension}"
    path.write_bytes(content)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        path.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not decode {upload.filename}.")
    if max(image.shape[:2]) > MAX_IMAGE_SIDE:
        path.unlink(missing_ok=True)
        raise HTTPException(413, f"Image dimensions must not exceed {MAX_IMAGE_SIDE} pixels per side.")
    return path


def get_models(model_name: str = "inswapper_128"):
    """Initialize InsightFace once and cache model-specific swap adapters."""
    global _FACE_APP, _SWAPPERS, _PROVIDER_NAMES
    if _FACE_APP is not None and model_name in _SWAPPERS:
        return _FACE_APP, _SWAPPERS[model_name], _PROVIDER_NAMES
    with _MODEL_LOCK:
        if _FACE_APP is not None and model_name in _SWAPPERS:
            return _FACE_APP, _SWAPPERS[model_name], _PROVIDER_NAMES
        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:
            raise RuntimeError(
                "InsightFace is not installed. Activate your virtual environment and run pip install -r requirements.txt."
            ) from exc

        available = ort.get_available_providers()
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"])
        ctx_id = 0 if "CUDAExecutionProvider" in available else -1
        try:
            face_app = _FACE_APP
            if face_app is None:
                face_app = FaceAnalysis(
                    name="buffalo_l",
                    providers=providers,
                    root=str(INSIGHTFACE_ROOT),
                )
                face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            from face_swap_models import create_swapper
            swapper = create_swapper(model_name, MODELS, providers)
        except Exception as exc:
            raise RuntimeError(
                f"Could not initialize InsightFace models. Check the model files and ONNX Runtime installation. Details: {exc}"
            ) from exc
        _FACE_APP, _PROVIDER_NAMES = face_app, providers
        _SWAPPERS[model_name] = swapper
        return _FACE_APP, swapper, _PROVIDER_NAMES


def face_area(face) -> float:
    width = max(0.0, float(face.bbox[2] - face.bbox[0]))
    height = max(0.0, float(face.bbox[3] - face.bbox[1]))
    confidence = float(getattr(face, "det_score", 1.0))
    return width * height * confidence


def source_quality_score(face, image: np.ndarray) -> float:
    """Estimate reference usefulness from detector confidence, face size and sharpness."""
    height, width = image.shape[:2]
    box = np.asarray(face.bbox, dtype=np.float32)
    x1, y1 = np.maximum(box[:2].astype(int), 0)
    x2, y2 = np.minimum(box[2:].astype(int), [width, height])
    if x2 <= x1 or y2 <= y1:
        return 1e-3
    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_score = float(np.clip(np.log1p(sharpness) / np.log1p(500.0), 0.1, 1.0))
    face_fraction = float((x2 - x1) * (y2 - y1) / max(width * height, 1))
    size_score = float(np.clip(np.sqrt(face_fraction / 0.08), 0.1, 1.0))
    detector_score = float(np.clip(getattr(face, "det_score", 0.5), 0.1, 1.0))
    return max(1e-3, sharpness_score * size_score * detector_score)


def normalized_embedding(face) -> np.ndarray:
    """Return a finite, unit-length ArcFace embedding for an InsightFace face."""
    raw = getattr(face, "normed_embedding", None)
    if raw is None:
        raw = getattr(face, "embedding", None)
    if raw is None:
        raise RuntimeError("InsightFace did not return a face identity embedding.")
    vector = np.asarray(raw, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.size == 0 or not np.isfinite(vector).all() or norm < 1e-8:
        raise RuntimeError("A source face produced an invalid identity embedding.")
    return vector / norm


def prepare_source_identity(source_candidates, merge_references: bool):
    """Select one source face or fuse embeddings from confirmed same-person references."""
    if not source_candidates:
        raise RuntimeError("No usable source faces were found.")
    best_face, _best_image = max(source_candidates, key=lambda item: face_area(item[0]))
    if not merge_references or len(source_candidates) == 1:
        embedding = normalized_embedding(best_face)
        return SimpleNamespace(normed_embedding=embedding), embedding, "best detected source face"

    embeddings = np.stack([normalized_embedding(face) for face, _image in source_candidates])
    if len({vector.shape for vector in embeddings}) != 1:
        raise RuntimeError("Source photos produced incompatible identity embeddings.")
    similarities = embeddings @ embeddings.T
    pairwise = similarities[np.triu_indices(len(embeddings), k=1)]
    min_similarity = float(np.min(pairwise)) if pairwise.size else 1.0
    if min_similarity < 0.20:
        raise RuntimeError(
            "The source photos do not appear to show one consistent identity "
            f"(lowest ArcFace cosine similarity: {min_similarity:.3f}). "
            "Use photos of the same person, or turn off reference merging."
        )
    weights = np.asarray(
        [source_quality_score(face, image) for face, image in source_candidates],
        dtype=np.float32,
    )
    weights /= max(float(weights.sum()), 1e-8)
    fused = np.average(embeddings, axis=0, weights=weights).astype(np.float32)
    fused_norm = float(np.linalg.norm(fused))
    if fused_norm < 1e-8 or not np.isfinite(fused).all():
        raise RuntimeError("Could not combine the source identity embeddings.")
    fused /= fused_norm
    return (
        SimpleNamespace(normed_embedding=fused),
        fused,
        f"merged identity embeddings from {len(source_candidates)} same-person reference photos",
    )


def measure_identity_similarity(face_app, result: np.ndarray, source_embedding: np.ndarray, target_face):
    """Measure output-to-reference ArcFace cosine similarity as a diagnostic, not a guarantee."""
    try:
        result_faces = face_app.get(result)
        if not result_faces:
            return None
        target_box = np.asarray(target_face.bbox, dtype=np.float32)
        target_center = (target_box[:2] + target_box[2:]) / 2.0
        def center_distance(face):
            box = np.asarray(face.bbox, dtype=np.float32)
            center = (box[:2] + box[2:]) / 2.0
            return float(np.linalg.norm(center - target_center))
        closest = min(result_faces, key=center_distance)
        output_embedding = normalized_embedding(closest)
        if output_embedding.shape != source_embedding.shape:
            return None
        return float(np.clip(np.dot(output_embedding, source_embedding), -1.0, 1.0))
    except Exception:
        return None


def make_face_mask(face, affine: np.ndarray, size: int = 128) -> np.ndarray:
    """Build a soft aligned-crop mask from the convex hull of dense facial landmarks."""
    mask = np.zeros((size, size), dtype=np.uint8)
    landmarks = getattr(face, "landmark_2d_106", None)
    if landmarks is not None and len(landmarks) >= 20:
        points = np.asarray(landmarks, dtype=np.float32).reshape(1, -1, 2)
        points_crop = cv2.transform(points, affine).reshape(-1, 2)
        points_crop = np.round(points_crop).astype(np.int32)
        hull = cv2.convexHull(points_crop)
        if hull is not None and len(hull) >= 3:
            cv2.fillConvexPoly(mask, hull, 255)
            mask = cv2.dilate(
                mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1
            )
        else:
            landmarks = None
    if landmarks is None or len(landmarks) < 20:
        cv2.ellipse(
            mask, (size // 2, int(size * 0.56)),
            (int(size * 0.39), int(size * 0.45)), 0, 0, 360, 255, -1,
        )
    return cv2.GaussianBlur(mask, (15, 15), sigmaX=0).astype(np.float32) / 255.0

def match_lab_color(swapped: np.ndarray, reference: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    """Conservative LAB mean/std matching against target face lighting and color."""
    if strength <= 0:
        return swapped
    src_lab = cv2.cvtColor(swapped, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    valid = mask > 0.35
    if np.count_nonzero(valid) < 100:
        return swapped

    corrected = src_lab.copy()
    # Match luminance more strongly than chroma to preserve the source person's skin tone.
    channel_strengths = (min(strength, 0.8), strength * 0.55, strength * 0.45)
    for channel, amount in enumerate(channel_strengths):
        src_values = src_lab[:, :, channel][valid]
        ref_values = ref_lab[:, :, channel][valid]
        src_std = float(np.std(src_values))
        ref_std = float(np.std(ref_values))
        if src_std < 1e-3:
            continue
        adjusted = (src_lab[:, :, channel] - float(np.mean(src_values))) * (ref_std / src_std)
        adjusted += float(np.mean(ref_values))
        corrected[:, :, channel] = (
            src_lab[:, :, channel] * (1.0 - amount) + adjusted * amount
        )
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)
    # Apply the correction mostly inside the face mask so crop background remains stable.
    local_alpha = np.clip(mask, 0, 1)[..., None]
    return np.clip(result.astype(np.float32) * local_alpha + swapped.astype(np.float32) * (1-local_alpha), 0, 255).astype(np.uint8)


def restore_face_optional(face_crop: np.ndarray, strength: float = 0.25) -> np.ndarray:
    """Run cached, low-strength GFPGAN restoration to reduce identity-altering changes."""
    global _GFP_GANER
    model_path = MODELS / "GFPGANv1.4.pth"
    if not model_path.exists():
        raise RuntimeError(
            "Face restoration was selected, but models/GFPGANv1.4.pth is missing. "
            "Add licensed weights or uncheck restoration."
        )
    with _RESTORER_LOCK:
        if _GFP_GANER is None:
            try:
                import torch
                from gfpgan import GFPGANer
            except ModuleNotFoundError as exc:
                missing = exc.name or "a dependency"
                if missing == "gfpgan" or missing.startswith("gfpgan."):
                    raise RuntimeError(
                        "GFPGAN is not installed. Install the optional restoration stack, "
                        "or uncheck face restoration."
                    ) from exc
                raise RuntimeError(
                    f"GFPGAN failed to import: missing module '{missing}'. Run "
                    "scripts/patch-basicsr.sh or see README.md for restoration setup."
                ) from exc
            except Exception as exc:
                raise RuntimeError(f"GFPGAN failed to import: {exc}") from exc
            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                _GFP_GANER = GFPGANer(
                    model_path=str(model_path), upscale=1, arch="clean",
                    channel_multiplier=2, bg_upsampler=None, device=device,
                )
            except Exception as exc:
                raise RuntimeError(f"Could not initialize GFPGAN: {exc}") from exc
    try:
        _cropped_faces, restored_faces, _restored_image = _GFP_GANER.enhance(
            face_crop, has_aligned=True, only_center_face=True,
            paste_back=False, weight=float(np.clip(strength, 0.0, 1.0)),
        )
        if not restored_faces:
            raise RuntimeError("GFPGAN did not return a restored face.")
        return cv2.resize(
            restored_faces[0], (face_crop.shape[1], face_crop.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"GFPGAN restoration failed: {exc}") from exc

def blend_face(
    target: np.ndarray,
    swapped_crop: np.ndarray,
    affine: np.ndarray,
    mask_crop: np.ndarray,
    blend_strength: float,
    blend_mode: Literal["feather", "seamless"],
) -> np.ndarray:
    height, width = target.shape[:2]
    inverse = cv2.invertAffineTransform(affine)
    warped = cv2.warpAffine(
        swapped_crop, inverse, (width, height), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpAffine(
        mask_crop, inverse, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    alpha = np.clip(warped_mask * float(blend_strength), 0, 1)
    if not np.any(alpha > 0.02):
        raise RuntimeError("The face mask is empty. Try a clearer target image.")

    if blend_mode == "seamless":
        # Clone only the local face ROI; a feathered alpha blend remains the fallback.
        binary = (warped_mask > 0.18).astype(np.uint8) * 255
        ys, xs = np.where(binary > 0)
        if len(xs):
            pad = 6
            x1, x2 = max(0, int(xs.min())-pad), min(width, int(xs.max())+pad+1)
            y1, y2 = max(0, int(ys.min())-pad), min(height, int(ys.max())+pad+1)
            roi_mask = binary[y1:y2, x1:x2]
            if roi_mask.shape[0] >= 10 and roi_mask.shape[1] >= 10:
                try:
                    target_roi = target[y1:y2, x1:x2]
                    swapped_roi = warped[y1:y2, x1:x2]
                    clone = cv2.seamlessClone(
                        swapped_roi, target_roi, roi_mask,
                        (target_roi.shape[1] // 2, target_roi.shape[0] // 2),
                        cv2.NORMAL_CLONE,
                    )
                    # Limit clone changes to the face mask and respect blend strength.
                    result = target.astype(np.float32).copy()
                    local_alpha = alpha[y1:y2, x1:x2, None]
                    result[y1:y2, x1:x2] = (
                        clone.astype(np.float32) * local_alpha
                        + target_roi.astype(np.float32) * (1 - local_alpha)
                    )
                    return np.clip(result, 0, 255).astype(np.uint8)
                except cv2.error:
                    pass

    alpha3 = alpha[..., None]
    result = warped.astype(np.float32) * alpha3 + target.astype(np.float32) * (1 - alpha3)
    return np.clip(result, 0, 255).astype(np.uint8)


def run_swap(
    sources: list[Path],
    target_path: Path,
    output_path: Path,
    color_match_strength: float = 0.35,
    blend_strength: float = 0.95,
    blend_mode: Literal["feather", "seamless"] = "feather",
    restore_face: bool = False,
    merge_source_references: bool = False,
    swap_model: Literal["inswapper_128", "ghost_1_256"] = "inswapper_128",
) -> tuple[str, str, float | None]:
    face_app, swapper, providers = get_models(swap_model)
    target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    if target is None:
        raise RuntimeError("Could not decode the target image.")
    target_faces = face_app.get(target)
    if not target_faces:
        raise RuntimeError("No face detected in target image. Use a clear image where the face is visible.")
    target_face = max(target_faces, key=face_area)

    source_candidates = []
    for path in sources:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        faces = face_app.get(image)
        if faces:
            source_candidates.append((max(faces, key=face_area), image))
    if not source_candidates:
        raise RuntimeError("No face detected in source images. Try clearer, front-facing reference photos.")
    source_face, source_embedding, source_note = prepare_source_identity(
        source_candidates, merge_source_references
    )

    try:
        # Each adapter returns its own aligned crop size and image-to-crop affine matrix.
        swapped_crop, affine = swapper.get(target, target_face, source_face, paste_back=False)
        crop_size = int(swapped_crop.shape[0])
        if swapped_crop.shape[:2] != (crop_size, crop_size):
            raise RuntimeError(f"Swap model returned a non-square crop: {swapped_crop.shape}")
        aligned_target = cv2.warpAffine(target, affine, (crop_size, crop_size), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = make_face_mask(target_face, affine, crop_size)
        if restore_face:
            # GFPGAN can invent identity details; keep its restoration weight conservative.
            swapped_crop = restore_face_optional(swapped_crop, strength=0.25)
        swapped_crop = match_lab_color(swapped_crop, aligned_target, mask, float(color_match_strength))
        result = blend_face(
            target, swapped_crop, affine, mask,
            float(blend_strength), blend_mode,
        )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Face compositing failed: {exc}") from exc

    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError("Could not save output image.")
    identity_similarity = measure_identity_similarity(
        face_app, result, source_embedding, target_face
    )
    device = "CUDA GPU" if "CUDAExecutionProvider" in providers else "CPU"
    return device, source_note, identity_similarity


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


@app.post("/api/composite")
async def composite(
    target: UploadFile = File(...),
    sources: list[UploadFile] = File(...),
    color_match_strength: float = Form(0.35),
    blend_strength: float = Form(0.95),
    blend_mode: Literal["feather", "seamless"] = Form("feather"),
    restore_face: bool = Form(False),
    merge_source_references: bool = Form(False),
    swap_model: Literal["inswapper_128", "ghost_1_256"] = Form("inswapper_128"),
):
    if not sources:
        raise HTTPException(400, "Upload at least one source image.")
    if not 0 <= color_match_strength <= 1:
        raise HTTPException(422, "Color-match strength must be between 0 and 1.")
    if not 0.1 <= blend_strength <= 1:
        raise HTTPException(422, "Blend strength must be between 0.1 and 1.0.")
    job = UPLOADS / uuid.uuid4().hex
    job.mkdir(parents=True, exist_ok=True)
    try:
        saved_sources = [await save_image(file, job) for file in sources[:12]]
        saved_target = await save_image(target, job)
        output = OUTPUTS / f"{uuid.uuid4().hex}.png"
        try:
            device, source_note, identity_similarity = run_swap(
                saved_sources, saved_target, output,
                color_match_strength=color_match_strength,
                blend_strength=blend_strength,
                blend_mode=blend_mode,
                restore_face=restore_face,
                merge_source_references=merge_source_references,
                swap_model=swap_model,
            )
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {
            "ok": True,
            "output_url": f"/outputs/{output.name}",
            "message": (
                f"Completed locally using {device}. Source: {source_note}. "
                + (
                    f"ArcFace cosine diagnostic: {identity_similarity:.3f} "
                    "(comparison signal, not a probability or guarantee)."
                    if identity_similarity is not None
                    else "Output identity diagnostic was unavailable."
                )
            ),
            "device": device,
            "identity_similarity": identity_similarity,
            "swap_model": swap_model,
        }
    finally:
        shutil.rmtree(job, ignore_errors=True)


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "swap_model_present": (MODELS / "inswapper_128.onnx").exists(),
        "available_swap_models": {
            "inswapper_128": (MODELS / "inswapper_128.onnx").is_file(),
            "ghost_1_256": (MODELS / "ghost_1_256.onnx").is_file() and (MODELS / "crossface_ghost.onnx").is_file(),
        },
        "restoration_model_present": (MODELS / "GFPGANv1.4.pth").exists(),
        "onnx_providers": ort.get_available_providers(),
    }
