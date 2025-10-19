"""
Configuration management for the Thoth backend
"""
import os
from typing import List
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True


@dataclass
class CORSConfig:
    """CORS configuration"""
    origins: List[str] = None
    
    def __post_init__(self):
        if self.origins is None:
            self.origins = [
                "http://localhost:3000",
                "http://localhost:8080", 
                "http://localhost:9000",
                "http://localhost:9001",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8080",
                "http://127.0.0.1:9000",
                "http://127.0.0.1:9001",
            ]


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "openai/whisper-large-v3"
    max_length: int = 448
    num_beams: int = 1
    do_sample: bool = False
    early_stopping: bool = True


@dataclass
class AudioConfig:
    """Audio configuration"""
    sample_rate: int = 16000
    buffer_duration_seconds: float = 3.0
    chunk_overlap: float = 0.1
    silence_threshold: float = 0.01
    min_audio_length: float = 0.5
    confidence_threshold: float = 0.3


@dataclass
class TranscriptionEngineConfig:
    """Transcription engine configuration"""
    engine_type: str = "chunked"
    batch_size: int = 4
    enable_mixed_precision: bool = True
    use_cache: bool = True
    chunk_duration_seconds: float = 30.0


@dataclass
class CUDAConfig:
    """CUDA configuration"""
    enabled: bool = True
    mixed_precision: str = "fp16"


class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.server = self._load_server_config()
        self.cors = self._load_cors_config()
        self.model = self._load_model_config()
        self.audio = self._load_audio_config()
        self.transcription_engine = self._load_transcription_engine_config()
        self.cuda = self._load_cuda_config()
    
    def _load_server_config(self) -> ServerConfig:
        """Load server configuration from environment variables"""
        return ServerConfig(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            debug=os.getenv("DEBUG", "true").lower() == "true"
        )
    
    def _load_cors_config(self) -> CORSConfig:
        """Load CORS configuration from environment variables"""
        origins_str = os.getenv("CORS_ORIGINS")
        if origins_str:
            origins = [origin.strip() for origin in origins_str.split(",")]
        else:
            origins = None
        return CORSConfig(origins=origins)
    
    def _load_model_config(self) -> ModelConfig:
        """Load model configuration from environment variables"""
        return ModelConfig(
            model_name=os.getenv("WHISPER_MODEL_NAME", "openai/whisper-large-v3"),
            max_length=int(os.getenv("WHISPER_MAX_LENGTH", "448")),
            num_beams=int(os.getenv("WHISPER_NUM_BEAMS", "1")),
            do_sample=os.getenv("WHISPER_DO_SAMPLE", "false").lower() == "true",
            early_stopping=os.getenv("WHISPER_EARLY_STOPPING", "true").lower() == "true"
        )
    
    def _load_audio_config(self) -> AudioConfig:
        """Load audio configuration from environment variables"""
        return AudioConfig(
            sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", "16000")),
            buffer_duration_seconds=float(os.getenv("AUDIO_BUFFER_DURATION_SECONDS", "3.0")),
            chunk_overlap=float(os.getenv("AUDIO_CHUNK_OVERLAP", "0.1")),
            silence_threshold=float(os.getenv("AUDIO_SILENCE_THRESHOLD", "0.01")),
            min_audio_length=float(os.getenv("AUDIO_MIN_LENGTH", "0.5")),
            confidence_threshold=float(os.getenv("AUDIO_CONFIDENCE_THRESHOLD", "0.3"))
        )
    
    def _load_transcription_engine_config(self) -> TranscriptionEngineConfig:
        """Load transcription engine configuration from environment variables"""
        return TranscriptionEngineConfig(
            engine_type=os.getenv("TRANSCRIPTION_ENGINE_TYPE", "chunked"),
            batch_size=int(os.getenv("TRANSCRIPTION_BATCH_SIZE", "4")),
            enable_mixed_precision=os.getenv("TRANSCRIPTION_ENABLE_MIXED_PRECISION", "true").lower() == "true",
            use_cache=os.getenv("TRANSCRIPTION_USE_CACHE", "true").lower() == "true",
            chunk_duration_seconds=float(os.getenv("TRANSCRIPTION_CHUNK_DURATION_SECONDS", "30.0"))
        )
    
    def _load_cuda_config(self) -> CUDAConfig:
        """Load CUDA configuration from environment variables"""
        return CUDAConfig(
            enabled=os.getenv("CUDA_ENABLED", "true").lower() == "true",
            mixed_precision=os.getenv("CUDA_MIXED_PRECISION", "fp16")
        )


# Global configuration instance
config = Config()
