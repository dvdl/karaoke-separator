#!/bin/bash
# Runs ON the pod. First thing: arm the pod's own kill timer, independent of the Mac.
set -u
MAX=${MAX_MINUTES:-45}
# ssh sessions do not see the container env: take RUNPOD_POD_ID / the pod-scoped key from PID 1
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_(API_KEY|POD_ID)|MAX_MINUTES)=' | sed 's/^/export /')"
export MAX=${MAX_MINUTES:-$MAX}
nohup bash -c '
sleep $((MAX * 60))
status=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID")
echo "DELETE status: $status"
if [ "$status" != "200" ] && [ "$status" != "204" ]; then
    status=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -d "{\"action\":\"stop\"}" "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID/action")
    echo "POST stop status: $status"
    if [ "$status" != "200" ] && [ "$status" != "204" ]; then
        if command -v runpodctl >/dev/null 2>&1; then
            runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"
        fi
    fi
fi
kill 1
' >/root/selfkill.log 2>&1 &
echo "self-kill armed: ${MAX} min (pod ${RUNPOD_POD_ID:-?}); runpodctl: $(command -v runpodctl || echo MISSING)"
set -e
export UV_HTTP_TIMEOUT=600 PIP_DEFAULT_TIMEOUT=600
T0=$(date +%s)
apt-get -qq update >/dev/null && apt-get -qq install -y ffmpeg >/dev/null
pip install -q uv
# keep the image's CUDA torch; audio-separator's other deps only
# [cpu]: onnxruntime-gpu wanted CUDA 13 (libcublasLt.so.13) on the 12.8 image; HQ4 runs on the server CPU
uv pip install --system -q "audio-separator[cpu]==0.47.0" audioread 2>&1 | tail -2
python -c "import torch;assert torch.cuda.is_available();print('torch',torch.__version__,torch.cuda.get_device_name(0))"
mkdir -p /models && cd /models
curl -sfLO https://huggingface.co/becruily/mel-band-roformer-deux/resolve/main/becruily_deux.ckpt
curl -sfL -o config_deux_becruily_mel_band_roformer.yaml https://huggingface.co/becruily/mel-band-roformer-deux/resolve/main/config_deux_becruily.yaml
U=https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models
curl -sfLO $U/model_bs_roformer_ep_317_sdr_12.9755.ckpt
curl -sfLO $U/UVR-MDX-NET-Inst_HQ_4.onnx
ls -la /models
echo "SETUP_DONE in $(( $(date +%s)-T0 ))s"
