from domain.entities.audio_file import AudioFile
from domain.entities.transcription import Transcription
from domain.services.transcription_domain_service import TranscriptionDomainService
from domain.ports.transcription_repository import TranscriptionRepository
from domain.ports.transcription_repository import NotificationService
from domain.value_objects.audio_config import AudioConfig
from app.use_cases.transcribe_audio_use_case import TranscribeAudioUseCase
from typing import Optional


class TranscribeAudioUseCaseImpl(TranscribeAudioUseCase):
    """Implementation of the transcribe audio use case"""
    
    def __init__(
        self,
        transcription_domain_service: TranscriptionDomainService,
        transcription_repository: Optional[TranscriptionRepository] = None,
        notification_service: Optional[NotificationService] = None
    ):
        self.transcription_domain_service = transcription_domain_service
        self.transcription_repository = transcription_repository
        self.notification_service = notification_service
    
    async def execute(self, audio_file: AudioFile, audio_config: AudioConfig) -> Transcription:
        """Execute the transcription use case"""
        try:
            # Execute domain logic
            transcription = await self.transcription_domain_service.transcribe_audio_file(
                audio_file, 
                audio_config
            )
            
            # Persist transcription if repository is available
            if self.transcription_repository:
                await self.transcription_repository.save_transcription(transcription)
            
            # Send notification if service is available
            if self.notification_service:
                await self.notification_service.send_transcription_complete(transcription)
            
            return transcription
            
        except Exception as e:
            # Send error notification if service is available
            if self.notification_service:
                await self.notification_service.send_error_notification(str(e))
            raise
