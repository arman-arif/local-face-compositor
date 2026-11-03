import shutil, uuid
from pathlib import Path
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'; UPLOADS = DATA / 'uploads'; OUTPUTS = DATA / 'outputs'; MODELS = ROOT / 'models'
for p in (UPLOADS, OUTPUTS, MODELS): p.mkdir(parents=True, exist_ok=True)
app = FastAPI(title='Local Face Compositor')
app.mount('/outputs', StaticFiles(directory=OUTPUTS), name='outputs')
ALLOWED = {'.jpg', '.jpeg', '.png', '.webp'}

async def save_image(upload: UploadFile, folder: Path) -> Path:
    ext = Path(upload.filename or '').suffix.lower()
    if ext not in ALLOWED: raise HTTPException(400, 'Use JPG, PNG, or WebP images.')
    content = await upload.read()
    if not content: raise HTTPException(400, 'An uploaded image is empty.')
    if len(content) > 25 * 1024 * 1024: raise HTTPException(413, 'Each image must be 25 MB or smaller.')
    path = folder / f'{uuid.uuid4().hex}{ext}'; path.write_bytes(content)
    if cv2.imread(str(path)) is None:
        path.unlink(missing_ok=True); raise HTTPException(400, f'Could not decode {upload.filename}.')
    return path

def run_swap(sources: list[Path], target_path: Path, output_path: Path):
    try:
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model
    except Exception as e:
        raise RuntimeError('InsightFace is not installed. Run pip install -r requirements.txt.') from e
    model_path = MODELS / 'inswapper_128.onnx'
    if not model_path.exists():
        raise RuntimeError('Missing models/inswapper_128.onnx. Add a model you are licensed to use; weights are not bundled.')
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        face_app = FaceAnalysis(name='buffalo_l', providers=providers, root=str(ROOT / '.insightface'))
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        swapper = get_model(str(model_path), providers=providers)
    except Exception as e:
        raise RuntimeError(f'Could not initialize models / ONNX Runtime: {e}') from e
    target = cv2.imread(str(target_path)); target_faces = face_app.get(target)
    if not target_faces: raise RuntimeError('No face detected in target image.')
    target_face = max(target_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    candidates = []
    for p in sources:
        im = cv2.imread(str(p)); faces = face_app.get(im)
        if faces:
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            area = (face.bbox[2]-face.bbox[0])*(face.bbox[3]-face.bbox[1])
            candidates.append((area, face))
    if not candidates: raise RuntimeError('No face detected in source images. Try clearer, front-facing references.')
    result = swapper.get(target, target_face, max(candidates, key=lambda x: x[0])[1], paste_back=True)
    if not cv2.imwrite(str(output_path), result): raise RuntimeError('Could not save output image.')

@app.get('/', response_class=HTMLResponse)
def index(): return (ROOT / 'static' / 'index.html').read_text(encoding='utf-8')

@app.post('/api/composite')
async def composite(target: UploadFile = File(...), sources: list[UploadFile] = File(...)):
    if not sources: raise HTTPException(400, 'Upload at least one source image.')
    job = UPLOADS / uuid.uuid4().hex; job.mkdir(parents=True, exist_ok=True)
    try:
        saved_sources = [await save_image(f, job) for f in sources[:12]]
        saved_target = await save_image(target, job)
        output = OUTPUTS / f'{uuid.uuid4().hex}.png'
        try: run_swap(saved_sources, saved_target, output)
        except RuntimeError as e: raise HTTPException(503, str(e))
        return {'ok': True, 'output_url': f'/outputs/{output.name}', 'message': 'Processing completed locally.'}
    finally: shutil.rmtree(job, ignore_errors=True)

@app.get('/api/health')
def health(): return {'ok': True, 'model_present': (MODELS / 'inswapper_128.onnx').exists()}
