# Local Face Compositor

A local browser-based face compositing starter using FastAPI, OpenCV, InsightFace, and ONNX Runtime. It makes no external AI API calls. Pretrained model weights are not bundled.

## Requirements
- Python 3.10 or 3.11 recommended.
- NVIDIA GPU with a compatible CUDA/cuDNN and ONNX Runtime setup for acceleration; CPU fallback can be slow.
- A face-swap model you are licensed to use.

## Install and run on Linux
```bash
cd local-face-compositor
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Put a compatible model at `models/inswapper_128.onnx`, then run:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000. The InsightFace `buffalo_l` analysis pack may be downloaded by the library on first run. For a fully offline environment, acquire and install that pack separately from a source whose license permits your use.

## Behavior and limitations
- Upload 1–12 source images and one target image.
- The app selects the largest detected face in the target and the largest detected face across source images. It does not blend multiple references into a composite identity.
- The swap model modifies the target image; results are not guaranteed to be pixel-perfect. Lighting, hairline, head proportions and seams may need manual review.
- The model weights can have separate license terms from the Python package. Verify the terms, especially for commercial use.
- Temporary uploads are removed after processing. Outputs remain in `data/outputs/`; delete them when no longer needed.
- Do not expose this starter app publicly without authentication, rate limits, and other safeguards.

## Troubleshooting
- **Missing model:** add a compatible, licensed `inswapper_128.onnx` under `models/`.
- **No CUDA provider:** install a matching GPU-enabled ONNX Runtime/CUDA stack, or switch to CPU `onnxruntime` (do not install both runtime packages at once).
- **No face detected:** use a clear, well-lit reference with a sufficiently large face.
