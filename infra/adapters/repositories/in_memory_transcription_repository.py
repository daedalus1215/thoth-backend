from domain.entities.transcription import Transcription
from domain.ports.transcription_repository import TranscriptionRepository
from domain.ports.transcription_repository import NotificationService
from typing import Optional


class InMemoryTranscriptionRepository(TranscriptionRepository):
    """In-memory implementation of transcription repository for testing/demo purposes"""
    
    def __init__(self):
        self.transcriptions: dict[str, Transcription] = {}
        self.next_id = 1
    
    async def save_transcription(self, transcription: Transcription) -> str:
        """Save transcription and return its ID"""
        transcription_id = str(self.next_id)
        self.next_id += 1
        self.transcriptions[transcription_id] = transcription
        return transcription_id
    
    async def get_transcription(self, transcription_id: str) -> Optional[Transcription]:
        """Get transcription by ID"""
        return self.transcriptions.get(transcription_id)
    
    async def get_transcriptions_by_date_range(self, start_date, end_date) -> list[Transcription]:
        """Get transcriptions within a date range"""
        # Simple implementation - in production, you'd filter by date
        return list(self.transcriptions.values())


class ConsoleNotificationService(NotificationService):
    """Console-based notification service for testing/demo purposes"""
    
    async def send_transcription_complete(self, transcription: Transcription) -> None:
        """Send notification when transcription is complete"""
        print(f"Transcription completed: {transcription.text}")
    
    async def send_error_notification(self, error_message: str) -> None:
        """Send error notification"""
        print(f"Error occurred: {error_message}")
