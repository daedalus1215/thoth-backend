from domain.entities.transcription import Transcription
from domain.ports.audio_processor import AudioBuffer, TranscriptionEngine
from domain.value_objects.audio_config import AudioConfig
from typing import Optional


class StreamingTranscriptionDomainService:
    """Domain service for streaming transcription business logic"""
    
    def __init__(self, audio_buffer: AudioBuffer, transcription_engine: TranscriptionEngine):
        self.audio_buffer = audio_buffer
        self.transcription_engine = transcription_engine
    
    async def process_audio_chunk(self, audio_chunk: bytes, audio_config: AudioConfig) -> Optional[Transcription]:
        """
        Process an audio chunk for streaming transcription following domain business rules
        """
        # Add chunk to buffer
        self.audio_buffer.add_chunk(audio_chunk)
        
        # Check if we have sufficient audio for processing
        if not self.audio_buffer.has_sufficient_audio():
            return None
        
        # Check if audio is silence
        if self.audio_buffer.is_silence():
            self.audio_buffer.clear_buffer()
            return None
        
        # Get buffered audio for transcription
        buffered_audio = self.audio_buffer.get_buffered_audio()
        
        # Transcribe the buffered audio
        transcription = await self.transcription_engine.transcribe_stream_chunk(buffered_audio)
        
        # Clear buffer after processing
        self.audio_buffer.clear_buffer()
        
        # Apply domain business rules
        if transcription and not transcription.is_valid():
            return None
        
        return transcription
    
    def reset_streaming_state(self) -> None:
        """Reset the streaming transcription state"""
        self.audio_buffer.clear_buffer()
        self.transcription_engine.reset_stream_state()
