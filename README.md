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

GFPGAN is not required for normal use and is intentionally not part of the default dependency set. The bundled `scripts/install-cpu.sh` and `scripts/install-gpu.sh` already include it. To add it to an existing environment, install `requirements/restoration.txt`, then run the BasicSR patch:

```bash
pip install -r requirements/restoration.txt
bash scripts/patch-basicsr.sh
```

`patch-basicsr.sh` is required because BasicSR 1.4.2 imports `torchvision.transforms.functional_tensor`, a module removed in torchvision 0.17. Without it, `import gfpgan` fails even though GFPGAN is installed.

Place the licensed GFPGAN v1.4 weights at:

```text
models/GFPGANv1.4.pth
```

Select **Optional face restoration** in the UI to run low-strength restoration. This is cached between requests, but it may still change facial features; keep it off when identity likeness is the priority.

## Processing pipeline

1. Detect faces in the target and every source image.
2. Select the largest/high-confidence face in the target and one detected face per source photo.
3. By default, use the best detected source face. Optionally enable **Combine same-person references** to average normalized ArcFace embeddings from multiple source photos. Use this only when every reference depicts the same person; likely mixed identities are rejected.
4. Run the local InSwapper model to create an aligned 128×128 crop.
5. Optionally apply cached, low-strength GFPGAN restoration. It remains opt-in because restoration can invent or alter identity details.
6. Match LAB statistics conservatively against the target's aligned face crop.
7. Warp the crop and convex-hull facial mask into the target image, then feather-blend or try OpenCV seamless clone.
8. Run an ArcFace cosine-similarity diagnostic between the final face and the source identity embedding. This is a comparison signal, not a probability, identity guarantee, or substitute for visual review.
9. Save the result as a PNG and show a before/after preview.

## Limitations

- This is a prototype. The 128×128 InSwapper model is the main quality ceiling: it can lose fine identity details and cannot guarantee a perfectly identical result. A stronger swap model/backend is required for a substantial quality jump.
- Multi-reference fusion combines identity embeddings, not image pixels. Its benefit depends on clear, varied photos of the same person; mixed identities are rejected using a conservative ArcFace similarity check.
- GFPGAN is a generative restoration model, not an identity-preservation model. The app now caches it and applies it at a conservative weight, but leaving restoration off will often preserve more of the original swapper output.
- The ArcFace diagnostic can vary with pose, expression, occlusion, lighting and detector quality. Compare scores only across runs made with the same source/target inputs; a higher score is not proof that the face is identical.
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
- **Restoration fails:** uncheck the restoration toggle, or install a compatible GFPGAN/PyTorch environment and provide the model file. If the error mentions `functional_tensor`, BasicSR needs patching — run `bash scripts/patch-basicsr.sh`.
- **Different source identities:** keep reference merging off, or upload only clear photos of the same person. The app rejects reference sets with low embedding agreement.
- **Low identity similarity:** use one or more sharp, well-lit source photos with different angles; keep GFPGAN off; compare the ArcFace diagnostic across runs. The InSwapper-128 model itself limits attainable detail and identity fidelity.
- **Unsatisfactory blend:** compare feather blending with seamless clone and adjust the color-match and blend-strength sliders.
