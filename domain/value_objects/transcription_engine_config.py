from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TranscriptionEngineConfig:
    """Configuration for transcription engine selection"""
    engine_type: Literal["standard", "accelerated", "batch", "chunked"] = "chunked"
    batch_size: int = 4
    chunk_duration_seconds: float = 30.0
    enable_mixed_precision: bool = True
    use_cache: bool = True
    
    def is_accelerated(self) -> bool:
        """Check if using accelerated engine"""
        return self.engine_type == "accelerated"
    
    def is_batch(self) -> bool:
        """Check if using batch engine"""
        return self.engine_type == "batch"
    
    def is_chunked(self) -> bool:
        """Check if using chunked engine"""
        return self.engine_type == "chunked"
    
    def is_standard(self) -> bool:
        """Check if using standard engine"""
        return self.engine_type == "standard"
