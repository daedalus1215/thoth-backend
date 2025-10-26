# hermes-backend
* Python service for ASR


## How to run:
docker compose up --build


# Setup:

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel

python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch torchvision torchaudio

sudo apt-get update && sudo apt-get install -y ffmpeg
# 1) First: ONLY the torch stack from the PyTorch index
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch torchvision torchaudio

# 2) Then: everything else from PyPI (no --index-url)
python -m pip install -r requirements.txt



# Sanity Check:

# A) Confirm interpreter + sites
python - <<'PY'
import sys, PIL, site
print("Executable:", sys.executable)
print("PIL from:", PIL.__file__)
print("ENABLE_USER_SITE:", site.ENABLE_USER_SITE)
PY
# Expect .venv/bin/python, .../.venv/.../site-packages/PIL/..., and ENABLE_USER_SITE=False

# B) Confirm CUDA Torch
python - <<'PY'
import torch
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY


### Run the App
uvicorn main:app --host 0.0.0.0 --port 8000 --reload


## Transcription Engine Configuration

The service uses **two separate transcription engines** optimized for different use cases:

### 🎤 Real-Time Streaming (`/stream-audio`)
- **Engine**: Chunked Whisper Transcription Engine
- **Chunk Duration**: 3 seconds
- **Optimization**: **LATENCY** - Fastest response rate for real-time transcription
- **GPU Usage**: Both engines use CUDA (use cases never run simultaneously)

### 📁 File Uploads (`/transcribe/`, `/upload`)
- **Engine**: Sequential Whisper Transcription Engine  
- **Chunk Duration**: 30 seconds with sliding window
- **Optimization**: **ACCURACY** - Maximum transcription quality using sliding window overlap
- **GPU Usage**: Both engines use CUDA (use cases never run simultaneously)

### Key Points:
- ✅ Both engines use **CUDA/GPU** by default (CPU was only used due to concurrent model concerns)
- ✅ Sequential engine provides **best accuracy** for uploaded files via sliding window processing
- ✅ Chunked engine provides **best latency** for real-time streaming via smaller 3s chunks
- ✅ Engines are **never run simultaneously**, allowing both to leverage GPU resources