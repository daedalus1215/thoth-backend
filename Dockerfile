FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

WORKDIR /app

# Install required system dependencies
RUN apt update && apt install -y \
    python3 python3-pip ffmpeg git \
    portaudio19-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
