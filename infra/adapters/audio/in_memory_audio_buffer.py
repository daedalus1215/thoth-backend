from domain.ports.audio_processor import AudioBuffer
from domain.value_objects.audio_config import AudioConfig
import numpy as np
from typing import List


class InMemoryAudioBuffer(AudioBuffer):
    """Infrastructure adapter for in-memory audio buffering"""
    
    def __init__(self, audio_config: AudioConfig):
        self.audio_config = audio_config
        self.buffer: List[float] = []
        self.buffer_size = audio_config.get_buffer_size()
    
    def add_chunk(self, audio_chunk: bytes) -> None:
        """Add audio chunk to buffer"""
        chunk_np = np.frombuffer(audio_chunk, dtype=np.float32)
        self.buffer.extend(chunk_np)
    
    def has_sufficient_audio(self) -> bool:
        """Check if buffer has sufficient audio for processing"""
        return len(self.buffer) >= self.buffer_size
    
    def get_buffered_audio(self) -> bytes:
        """Get buffered audio data"""
        if not self.buffer:
            return b''
        
        audio_data = np.array(self.buffer[:self.buffer_size], dtype=np.float32)
        return audio_data.tobytes()
    
    def clear_buffer(self) -> None:
        """Clear the audio buffer"""
        # Keep any remaining audio that exceeds buffer size
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[self.buffer_size:]
        else:
            self.buffer = []
    
    def is_silence(self) -> bool:
        """Check if buffered audio is silence"""
        if not self.buffer:
            return True
        
        audio_data = np.array(self.buffer[:self.buffer_size], dtype=np.float32)
        rms = np.sqrt(np.mean(audio_data**2))
        return rms < self.audio_config.silence_threshold
