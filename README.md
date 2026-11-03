# Local Face Compositor — upgraded prototype

A browser-based local face-compositing app using FastAPI, OpenCV, InsightFace and ONNX Runtime. The app does not call an external AI API.

## What changed in this version

- More efficient model initialization: InsightFace models remain loaded between requests.
- Uses the swapper's aligned 128×128 crop and affine transform, rather than accepting the model's default full-image paste-back.
- Creates a soft mask using facial contour landmarks when available, with an oval fallback.
- Adds a conservative LAB color/lighting match slider.
- Adds soft feather blending and OpenCV seamless-clone blending modes.
- Adds an opt-in GFPGAN restoration hook. It is off by default because restoration can modify identity details.
- Adds before/after previews, processing feedback, input-size checks and an ONNX provider status endpoint.
- Redesigned interface: drag-and-drop uploads, model status checks, a before/after slider with zoom, and a history of results from the current session.

## Requirements and local setup

Python 3.10 or 3.11 is recommended. For GPU acceleration, install an ONNX Runtime version compatible with your local CUDA/cuDNN setup and NVIDIA driver.

```bash
unzip local-face-compositor.zip
cd local-face-compositor
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Add a compatible face-swap model you are licensed to use here:

```text
models/inswapper_128.onnx
```

InsightFace's `buffalo_l` face-analysis pack may be downloaded into `.insightface/` at startup if absent. To keep execution entirely offline, prepare the pack locally before starting the app. The app itself does not call an external AI API.

Start the web app:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. `/api/health` reports required model-file presence and ONNX Runtime execution providers.

## Optional GFPGAN restoration

GFPGAN is not required for normal use and is intentionally not part of the default dependency set. First install a compatible PyTorch build for your machine using the official PyTorch installation selector. Then install the optional dependency:

```bash
pip install -r requirements-restoration.txt
```

Place the licensed GFPGAN v1.4 weights at:

```text
models/GFPGANv1.4.pth
```

Select **Optional face restoration** in the UI to run it. Restoration may sharpen details, but can change facial features and identity likeness. Keep it off unless you prefer the restored result.

## Processing pipeline

1. Detect faces in the target and every source image.
2. Select the largest/high-confidence face in the target and the highest-area/high-confidence source face among uploaded references.
3. Run the local InsightFace swapper to create an aligned crop.
4. Optionally restore the aligned face crop with GFPGAN.
5. Match LAB statistics conservatively against the target's aligned face crop.
6. Warp the crop and soft facial mask into the target image, then feather-blend or try OpenCV seamless clone.
7. Save the result as a PNG and show a before/after preview.

## Limitations

- This is a prototype. The 128×128 InSwapper model has detail limitations and may not reproduce a person's face perfectly. Higher-resolution alternatives require a separate, compatible inference backend; this project does not claim to support HyperSwap directly.
- Multiple reference images are accepted, but the first version selects one detected source face; it does not fuse identity information across references.
- The color match is intentionally conservative and may need adjustment for unusual lighting, makeup, or skin tones.
- Seamless clone sometimes looks worse than feather blending. Compare both modes.
- Inference has not been verified until you install the dependencies and model weights on your machine.
- No pretrained weights are bundled. Model code and model weights can have different license terms. Verify the specific weight license, especially for commercial use.
- Use only images you own or have permission to edit. Do not use the application to impersonate, defraud, or create non-consensual intimate imagery.
- Outputs remain in `data/outputs/`; remove them when no longer needed. The app is intended to bind to localhost. Do not expose it to the public internet without authentication, access controls and additional safeguards.

## Troubleshooting

- **Missing model file:** provide `models/inswapper_128.onnx` from a source whose terms permit your use.
- **No CUDA provider:** the app falls back to CPU when ONNX Runtime does not report `CUDAExecutionProvider`. CPU inference will usually be slower. For GPU execution, resolve driver/CUDA/cuDNN/runtime compatibility first.
- **No face detected:** use clear, well-lit images where the face is large enough and not heavily occluded.
- **Restoration fails:** uncheck the restoration toggle or install a compatible GFPGAN/PyTorch environment and provide the model file.
- **Unsatisfactory blend:** compare feather blending with seamless clone and adjust the color-match and blend-strength sliders.
