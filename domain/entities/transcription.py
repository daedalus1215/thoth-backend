from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass(frozen=True)
class Transcription:
    """Domain entity representing a transcription result"""
    text: str
    confidence: Optional[float] = None
    language: Optional[str] = None
    duration_seconds: Optional[float] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            object.__setattr__(self, 'created_at', datetime.utcnow())
    
    def is_empty(self) -> bool:
        """Check if transcription is empty or contains only whitespace"""
        return not self.text or not self.text.strip()
    
    def is_valid(self) -> bool:
        """Check if transcription is valid (not empty and has reasonable length)"""
        if self.is_empty():
            return False
        
        # Filter out very short transcriptions that might be noise
        return len(self.text.strip()) >= 3
    
    def get_word_count(self) -> int:
        """Get the number of words in the transcription"""
        if self.is_empty():
            return 0
        return len(self.text.strip().split())
