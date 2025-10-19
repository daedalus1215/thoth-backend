from abc import ABC, abstractmethod
from typing import Optional
from domain.entities.audio_file import AudioFile
from domain.entities.transcription import Transcription


class AudioProcessor(ABC):
    """Port for audio processing operations"""
    
    @abstractmethod
    async def adjust_sample_rate(self, audio_file: AudioFile, target_sample_rate: int) -> AudioFile:
        """Adjust the sample rate of an audio file"""
        pass
    
    @abstractmethod
    def validate_audio_type(self, audio_file: AudioFile) -> bool:
        """Validate if the audio file type is supported"""
        pass


class TranscriptionEngine(ABC):
    """Port for transcription operations"""
    
    @abstractmethod
    async def transcribe_audio(self, audio_file: AudioFile) -> Transcription:
        """Transcribe audio file to text"""
        pass
    
    @abstractmethod
    async def transcribe_stream_chunk(self, audio_chunk: bytes) -> Optional[Transcription]:
        """Transcribe a chunk of streaming audio"""
        pass
    
    @abstractmethod
    def reset_stream_state(self) -> None:
        """Reset the state for streaming transcription"""
        pass


class AudioBuffer(ABC):
    """Port for audio buffering operations"""
    
    @abstractmethod
    def add_chunk(self, audio_chunk: bytes) -> None:
        """Add audio chunk to buffer"""
        pass
    
    @abstractmethod
    def has_sufficient_audio(self) -> bool:
        """Check if buffer has sufficient audio for processing"""
        pass
    
    @abstractmethod
    def get_buffered_audio(self) -> bytes:
        """Get buffered audio data"""
        pass
    
    @abstractmethod
    def clear_buffer(self) -> None:
        """Clear the audio buffer"""
        pass
    
    @abstractmethod
    def is_silence(self) -> bool:
        """Check if buffered audio is silence"""
        pass
