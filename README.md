```
████████╗██╗  ██╗ ██████╗ ████████╗██╗  ██╗
╚══██╔══╝██║  ██║██╔═══██╗╚══██╔══╝██║  ██║
   ██║   ███████║██║   ██║   ██║   ███████║
   ██║   ██╔══██║██║   ██║   ██║   ██╔══██║
   ██║   ██║  ██║╚██████╔╝   ██║   ██║  ██║
   ╚═╝   ╚═╝  ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝

              🎤 Python ASR service with Whisper
```

# Thoth Backend

> Python service for automatic speech recognition (ASR) via Whisper, with real-time streaming and file upload transcription.

---

## Overview

Thoth Backend is a FastAPI service that provides speech-to-text transcription using OpenAI's Whisper model. It supports two usage patterns: **real-time streaming** over WebSockets for low-latency transcription, and **file upload** (single or batch) for higher-accuracy, sliding-window transcription.

The service is built with **Hexagonal Architecture** and **Domain-Driven Design**: domain logic lives in `domain/`, application use cases in `app/`, and external adapters (Whisper, audio processing) in `infra/`. This keeps business rules independent of frameworks and makes it easy to swap engines or add new entrypoints.

### Key Features

- **Real-time streaming**: WebSocket endpoint for chunked audio with low latency (3s chunks).
- **File upload transcription**: Single-file and batch endpoints with sliding-window processing for accuracy.
- **Dual engines**: Chunked Whisper for streaming (latency-optimized), Sequential Whisper for uploads (accuracy-optimized); both use CUDA when available.
- **Configurable**: Whisper model, chunk duration, sample rate, and CUDA/mixed-precision via environment variables.

### Role in Atlas Ecosystem

- Consumed by frontends or other services that need ASR (streaming or batch).
- Provides HTTP and WebSocket APIs; no database or message queues in the current design.

---

## Prerequisites

- **Python**: 3.12
- **pip**: Current pip, setuptools, wheel
- **PyTorch (CUDA)**: Install from [PyTorch wheel index](https://download.pytorch.org/whl/cu124) for CUDA 12.4 (or match your GPU).
- **ffmpeg**: System dependency for audio handling (`apt-get install ffmpeg` or equivalent).
- **GPU**: Optional but recommended; both engines use CUDA when available and are not run simultaneously.

---

## Getting Started

### 1. Installation

```bash
# Clone the repository (if not already done)
git clone [repository-url]
cd thoth-backend

# Create and activate virtual environment
python3.12 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
python -m pip install -U pip setuptools wheel
```

### 2. PyTorch and system dependencies

```bash
# Install PyTorch with CUDA from PyTorch index first
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch torchvision torchaudio

# Install ffmpeg (Debian/Ubuntu)
sudo apt-get update && sudo apt-get install -y ffmpeg

# Install project dependencies from PyPI (no --index-url)
python -m pip install -r requirements.txt
```

### 3. Environment setup

```bash
# Copy the sample environment file
cp env.example .env

# Edit with your local values (host, port, CORS, Whisper model, CUDA, etc.)
```

See `env.example` for all supported environment variables (server, CORS, Whisper model, audio config, transcription engine, CUDA).

### 4. Sanity check (optional)

**Interpreter and PIL:**

```bash
python - <<'PY'
import sys, PIL, site
print("Executable:", sys.executable)
print("PIL from:", PIL.__file__)
print("ENABLE_USER_SITE:", site.ENABLE_USER_SITE)
PY
```

Expect executable under `.venv/bin/python`, PIL under `.venv/.../site-packages/`, and `ENABLE_USER_SITE=False`.

**CUDA and Torch:**

```bash
python - <<'PY'
import torch
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

### 5. Start the application

**Development (reload):**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**With SSL:**

```bash
uvicorn main:app --host 0.0.0.0 --port 8443 --reload \
  --ssl-keyfile=/path/to/key --ssl-certfile=/path/to/crt
```

**Docker:**

```bash
docker compose up --build
```

The application runs on port **8000** by default (or as set in `.env`).

---

## Common Commands

### Development

| Command | Description |
| ------- | ----------- |
| `uvicorn main:app --host 0.0.0.0 --port 8000 --reload` | Start with hot reload |
| `python main.py` | Start using `main.py` entry (uses config from `.env`) |
| `docker compose up --build` | Run with Docker Compose |

### Environment

| Task | Description |
| ---- | ----------- |
| `cp env.example .env` | Create local env from sample |
| Edit `.env` | Configure host, port, CORS, Whisper, CUDA |

---

## API Endpoints

### Transcription

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| `POST` | `/transcribe/` | Transcribe one uploaded audio file (with progress/timeout) |
| `POST` | `/upload` | Upload and transcribe one audio file |
| `POST` | `/transcribe/batch` | Transcribe multiple uploaded audio files |
| `WebSocket` | `/stream-audio` | Stream audio chunks; receive transcriptions in real time |

### Health and performance

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| `GET` | `/health` | Health check |
| `GET` | `/performance` | Performance/device info and audio config |

---

## Transcription engine configuration

The service uses **two separate transcription engines**:

### Real-time streaming (`/stream-audio`)

- **Engine**: Chunked Whisper
- **Chunk duration**: 3 seconds
- **Optimization**: **Latency** for real-time transcription
- **GPU**: CUDA when available

### File uploads (`/transcribe/`, `/upload`, `/transcribe/batch`)

- **Engine**: Sequential Whisper with sliding window
- **Chunk duration**: 30 seconds (configurable)
- **Optimization**: **Accuracy** via sliding-window overlap
- **GPU**: CUDA when available

### Notes

- Both engines use **CUDA/GPU** by default when available.
- Engines are **not run simultaneously**, so both can use the same GPU.
- Sequential engine: best accuracy for uploaded files.
- Chunked engine: best latency for streaming.

---

## Architecture

The service follows **Domain-Driven Design** and **Hexagonal Architecture**:

- **Domain layer** (`domain/`): Entities, value objects, ports (interfaces), and domain services.
- **Application layer** (`app/`): Use cases, API controllers, and dependency injection.
- **Infrastructure layer** (`infra/`): Adapters for Whisper, audio processing, and repositories.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for details and dependency flow.

### Project structure

```
thoth-backend/
├── app/
│   ├── api/controllers/    # HTTP/WebSocket controllers
│   ├── config/             # Settings (from env)
│   ├── di/                 # Dependency injection container
│   └── use_cases/          # Transcribe and stream use cases
├── domain/
│   ├── entities/           # AudioFile, Transcription
│   ├── value_objects/      # AudioConfig, engine config
│   ├── ports/              # Interfaces (audio, transcription, repository)
│   └── services/           # Domain services
├── infra/
│   └── adapters/           # Whisper engines, audio (e.g. librosa), repositories
├── main.py                 # FastAPI app entry
├── env.example             # Example environment variables
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── ARCHITECTURE.md
└── README.md
```

---

## Docker & deployment

### Run with Docker Compose

```bash
docker compose up --build
```

### Build image only

```bash
docker build -t thoth-backend .
```

---

## Troubleshooting

### CUDA / PyTorch

- Install PyTorch from the PyTorch index first, then `pip install -r requirements.txt` without `--index-url` so other packages come from PyPI.
- Confirm with the sanity-check script above that `torch.cuda.is_available()` is `True` and the GPU name is printed.

### Audio / ffmpeg

- Ensure `ffmpeg` is on `PATH`; the stack uses it for decoding.
- If uploads fail, check file type (audio/*) and size (e.g. 100MB limit on `/transcribe/`).

### Timeouts

- File transcription uses a timeout based on file size (capped at 30 minutes). For very long files, consider splitting or increasing limits in code.

### Environment

- Ensure `.env` exists (from `env.example`) and that `HOST`, `PORT`, and `CORS_ORIGINS` match your setup.
- For SSL, set `ssl_keyfile` and `ssl_certfile` in config or pass them to `uvicorn`.

---

## Additional resources

- [ARCHITECTURE.md](./ARCHITECTURE.md) – Hexagonal layout and layers
- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenAI Whisper](https://github.com/openai/whisper)

---

## Contributing

1. Follow the architecture in `domain/` → `app/` → `infra/`.
2. Run tests and fix any failures.
3. Use the existing env and CLI conventions (see Getting Started and Common Commands).
