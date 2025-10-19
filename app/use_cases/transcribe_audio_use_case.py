from abc import ABC, abstractmethod
from typing import Optional
from domain.entities.audio_file import AudioFile
from domain.entities.transcription import Transcription
from domain.value_objects.audio_config import AudioConfig


class TranscribeAudioUseCase(ABC):
    """Use case for transcribing audio files"""
    
    @abstractmethod
    async def execute(self, audio_file: AudioFile, audio_config: AudioConfig) -> Transcription:
        """Execute the transcription use case"""
        pass


#TODO: Probably belongs to it's own class file.
class StreamAudioUseCase(ABC):
    """Use case for streaming audio transcription"""
    
    @abstractmethod
    async def execute(self, audio_chunk: bytes, audio_config: AudioConfig) -> Optional[Transcription]:
        """Execute the streaming transcription use case"""
        pass
    
    @abstractmethod
    def reset_stream(self) -> None:
        """Reset the streaming state"""
        pass
