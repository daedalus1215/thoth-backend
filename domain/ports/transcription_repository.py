from abc import ABC, abstractmethod
from typing import Optional
from domain.entities.transcription import Transcription


class TranscriptionRepository(ABC):
    """Port for transcription persistence operations"""
    
    @abstractmethod
    async def save_transcription(self, transcription: Transcription) -> str:
        """Save transcription and return its ID"""
        pass
    
    @abstractmethod
    async def get_transcription(self, transcription_id: str) -> Optional[Transcription]:
        """Get transcription by ID"""
        pass
    
    @abstractmethod
    async def get_transcriptions_by_date_range(self, start_date, end_date) -> list[Transcription]:
        """Get transcriptions within a date range"""
        pass


class NotificationService(ABC):
    """Port for notification operations"""
    
    @abstractmethod
    async def send_transcription_complete(self, transcription: Transcription) -> None:
        """Send notification when transcription is complete"""
        pass
    
    @abstractmethod
    async def send_error_notification(self, error_message: str) -> None:
        """Send error notification"""
        pass
