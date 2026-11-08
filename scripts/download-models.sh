#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="$(pwd)"
WITH_GHOST=0
for arg in "$@"; do
  case "$arg" in
    --with-ghost) WITH_GHOST=1 ;;
    --help|-h)
      echo "Usage: $0 [PROJECT_DIR] [--with-ghost]"
      echo "Downloads the baseline models. Pass --with-ghost to also download the experimental GHOST-1 256 assets (~515 MB for the swapper model alone)."
      exit 0
      ;;
    -*)
      echo "Unknown option: $arg" >&2
      echo "Usage: $0 [PROJECT_DIR] [--with-ghost]" >&2
      exit 2
      ;;
    *) PROJECT_DIR="$arg" ;;
  esac
done
MODEL_DIR="$PROJECT_DIR/models"
BUFFALO_DIR="$PROJECT_DIR/.insightface/models/buffalo_l"
mkdir -p "$MODEL_DIR" "$BUFFALO_DIR"
command -v curl >/dev/null 2>&1 || { echo "Install curl first." >&2; exit 1; }
command -v unzip >/dev/null 2>&1 || { echo "Install unzip first." >&2; exit 1; }
download() {
  local url="$1" dest="$2" label="$3"
  if [[ -s "$dest" ]]; then echo "[skip] $label exists"; return; fi
  local tmp="${dest}.part"
  rm -f "$tmp"
  echo "[download] $label"
  curl --fail --location --retry 3 --retry-delay 2 --connect-timeout 30 --output "$tmp" "$url"
  [[ -s "$tmp" ]] || { echo "Empty download: $label" >&2; rm -f "$tmp"; exit 1; }
  mv "$tmp" "$dest"
}
download "https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx" "$MODEL_DIR/inswapper_128.onnx" "InSwapper-128"
BUFFALO_ZIP="$MODEL_DIR/buffalo_l.zip"
download "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip" "$BUFFALO_ZIP" "InsightFace buffalo_l"
if [[ ! -s "$BUFFALO_DIR/det_10g.onnx" || ! -s "$BUFFALO_DIR/w600k_r50.onnx" ]]; then
  echo "[extract] buffalo_l"
  unzip -oq "$BUFFALO_ZIP" -d "$PROJECT_DIR/.insightface/models"
fi
[[ -s "$BUFFALO_DIR/det_10g.onnx" && -s "$BUFFALO_DIR/w600k_r50.onnx" ]] || {
  echo "buffalo_l extraction incomplete: $BUFFALO_DIR" >&2; exit 1;
}
# GHOST is intentionally opt-in because its swapper file is very large.
if [[ "$WITH_GHOST" -eq 1 ]]; then
  download "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/ghost_1_256.onnx" "$MODEL_DIR/ghost_1_256.onnx" "GHOST-1 256"
  download "https://github.com/facefusion/facefusion-assets/releases/download/models-3.4.0/crossface_ghost.onnx" "$MODEL_DIR/crossface_ghost.onnx" "GHOST embedding converter"
else
  echo "[skip] GHOST assets not requested. Use --with-ghost to download the experimental model."
fi
download "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" "$MODEL_DIR/GFPGANv1.4.pth" "GFPGAN 1.4 (optional)"
echo
echo "Models downloaded. Verify the model and converter licenses before commercial use."
echo "GHOST is experimental; run scripts/benchmark-models.py before selecting it."
