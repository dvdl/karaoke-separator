# Code and libraries only - no model weights (deux is CC-BY-NC-4.0; fetched at cold start).
# The 2.9.1 tag does not exist; this one ran deux on CUDA in the 2026-10-14 A/B.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
RUN apt-get update -qq && apt-get install -y -qq ffmpeg && rm -rf /var/lib/apt/lists/*
# Keep the image's CUDA torch. The [cpu] extra only brings onnxruntime (CPU), which
# audio-separator imports; deux runs on torch, so no onnxruntime-gpu / CUDA-13 mismatch.
RUN pip install --no-cache-dir uv && UV_HTTP_TIMEOUT=600 uv pip install --system --no-cache \
      "audio-separator[cpu]==0.47.0" audioread "runpod>=1.7" \
 && python -c "import torch; print('torch', torch.__version__)"
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
