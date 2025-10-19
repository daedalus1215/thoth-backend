from dataclasses import dataclass
from typing import Optional
import io


@dataclass(frozen=True)
class AudioFile:
    """Domain entity representing an audio file"""
    content: bytes
    filename: str
    content_type: str
    size: int
    
    @classmethod
    def from_upload_file(cls, upload_file) -> 'AudioFile':
        """Factory method to create AudioFile from FastAPI UploadFile"""
        content = upload_file.file.read()
        return cls(
            content=content,
            filename=upload_file.filename,
            content_type=upload_file.content_type,
            size=len(content)
        )
    
    def to_bytes(self) -> bytes:
        """Convert audio file to bytes"""
        return self.content
    
    def get_file_extension(self) -> str:
        """Get file extension from filename"""
        if not self.filename:
            return ""
        return self.filename.split('.')[-1].lower() if '.' in self.filename else ""
