from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AudioConfig:
    """Value object representing audio processing configuration"""
    sample_rate: int = 16000
    buffer_duration_seconds: float = 3.0
    chunk_overlap: float = 0.1
    silence_threshold: float = 0.01
    min_audio_length: float = 0.5
    confidence_threshold: float = 0.3
    
    def get_buffer_size(self) -> int:
        """Calculate buffer size in samples"""
        return int(self.buffer_duration_seconds * self.sample_rate)
    
    def get_chunk_overlap_samples(self) -> int:
        """Calculate chunk overlap in samples"""
        return int(self.chunk_overlap * self.sample_rate)


@dataclass(frozen=True)
class ModelConfig:
    """Value object representing AI model configuration"""
    model_name: str = "openai/whisper-medium"
    max_length: int = 448
    num_beams: int = 1
    do_sample: bool = False
    early_stopping: bool = True
