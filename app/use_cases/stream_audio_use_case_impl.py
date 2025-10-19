from domain.entities.transcription import Transcription
from domain.services.streaming_transcription_domain_service import StreamingTranscriptionDomainService
from domain.value_objects.audio_config import AudioConfig
from app.use_cases.transcribe_audio_use_case import StreamAudioUseCase
from typing import Optional


class StreamAudioUseCaseImpl(StreamAudioUseCase):
    """Implementation of the stream audio use case"""
    
    def __init__(self, streaming_transcription_domain_service: StreamingTranscriptionDomainService):
        self.streaming_transcription_domain_service = streaming_transcription_domain_service
    
    async def execute(self, audio_chunk: bytes, audio_config: AudioConfig) -> Optional[Transcription]:
        """Execute the streaming transcription use case"""
        return await self.streaming_transcription_domain_service.process_audio_chunk(
            audio_chunk, 
            audio_config
        )
    
    def reset_stream(self) -> None:
        """Reset the streaming state"""
        self.streaming_transcription_domain_service.reset_streaming_state()
