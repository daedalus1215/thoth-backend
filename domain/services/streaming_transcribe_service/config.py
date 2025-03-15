from dataclasses import dataclass

@dataclass
class StreamingConfig:
    sample_rate: int = 16000
    buffer_duration_seconds: float = 1.0  # Buffer 1 second of audio
    model_name: str = "openai/whisper-base"
    chunk_overlap: float = 0.2  # 200ms overlap between chunks 