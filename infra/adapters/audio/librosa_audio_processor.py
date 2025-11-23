from domain.entities.audio_file import AudioFile
from domain.ports.audio_processor import AudioProcessor
import librosa
import io
import numpy as np


class LibrosaAudioProcessor(AudioProcessor):
    """Infrastructure adapter for audio processing using Librosa"""
    
    async def adjust_sample_rate(self, audio_file: AudioFile, target_sample_rate: int) -> AudioFile:
        """Adjust the sample rate of an audio file using Librosa"""
        try:
            # Load audio with librosa
            audio_data, original_sample_rate = librosa.load(
                io.BytesIO(audio_file.content),
                sr=target_sample_rate
            )
            
            # Convert back to bytes
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            
            return AudioFile(
                content=audio_bytes,
                filename=audio_file.filename,
                content_type=audio_file.content_type,
                size=len(audio_bytes)
            )
        except Exception as e:
            raise ValueError(f"Failed to adjust sample rate: {str(e)}")
    
    def validate_audio_type(self, audio_file: AudioFile) -> bool:
        """Validate if the audio file type is supported"""
        supported_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        file_extension = audio_file.get_file_extension()
        
        if not file_extension:
            return False
        
        return f'.{file_extension}' in supported_extensions
