from domain.entities.audio_file import AudioFile
from domain.entities.transcription import Transcription
from domain.ports.audio_processor import AudioProcessor, TranscriptionEngine
from domain.value_objects.audio_config import AudioConfig, ModelConfig


class TranscriptionDomainService:
    """Domain service for transcription business logic"""
    
    def __init__(self, audio_processor: AudioProcessor, transcription_engine: TranscriptionEngine):
        self.audio_processor = audio_processor
        self.transcription_engine = transcription_engine
    
    async def transcribe_audio_file(self, audio_file: AudioFile, audio_config: AudioConfig) -> Transcription:
        """
        Transcribe an audio file following domain business rules
        """
        # Validate audio type
        if not self.audio_processor.validate_audio_type(audio_file):
            raise ValueError(f"Unsupported audio type: {audio_file.content_type}")
        
        # Adjust sample rate if needed
        processed_audio = await self.audio_processor.adjust_sample_rate(
            audio_file, 
            audio_config.sample_rate
        )
        
        # Transcribe the audio
        transcription = await self.transcription_engine.transcribe_audio(processed_audio)
        
        # Apply domain business rules
        if not transcription.is_valid():
            raise ValueError("Transcription result is not valid")
        
        return transcription
    
    def filter_transcription(self, transcription: Transcription) -> Transcription | None:
        """
        Apply domain business rules to filter transcription results
        """
        if not transcription.is_valid():
            return None
        
        # Remove common filler words that appear during silence
        filler_words = ["thank you", "thanks", "um", "uh", "hmm", "mm", "yeah", "yes", "no"]
        transcription_lower = transcription.text.lower().strip()
        
        # If it's just a filler word, ignore it
        if transcription_lower in filler_words:
            return None
        
        return transcription
