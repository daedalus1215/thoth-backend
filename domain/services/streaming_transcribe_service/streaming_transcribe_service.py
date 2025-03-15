from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import torch
from .config import StreamingConfig

class StreamingTranscribeService:
    def __init__(self, config: StreamingConfig = None):
        self.config = config or StreamingConfig()
        
        self.processor = WhisperProcessor.from_pretrained(self.config.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.config.model_name)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        
        self.sample_rate = self.config.sample_rate
        self.chunk_overlap = int(self.config.chunk_overlap * self.config.sample_rate)
        self.buffer_size = int(self.config.buffer_duration_seconds * self.config.sample_rate)
        self.audio_buffer = []
        
    async def process_chunk(self, audio_chunk: bytes) -> str | None:
        """
        Process an incoming audio chunk. Returns transcription only when 
        enough audio has been buffered.
        """
        try:
            # Convert bytes to numpy array (assuming float32 audio data)
            chunk_np = np.frombuffer(audio_chunk, dtype=np.float32)
            self.audio_buffer.extend(chunk_np)
            
            # If we have enough audio data, process it
            if len(self.audio_buffer) >= self.buffer_size:
                # Convert buffer to numpy array
                audio_data = np.array(self.audio_buffer, dtype=np.float32)
                
                # Process audio with Whisper
                input_features = self.processor(
                    audio_data,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt"
                ).input_features
                
                if torch.cuda.is_available():
                    input_features = input_features.to("cuda")

                # Generate token ids
                predicted_ids = self.model.generate(input_features)
                
                # Decode token ids to text
                transcription = self.processor.batch_decode(
                    predicted_ids, 
                    skip_special_tokens=True
                )[0]
                
                # Clear the buffer but keep any remaining audio
                self.audio_buffer = self.audio_buffer[self.buffer_size:]
                
                return transcription
                
            return None  # Not enough audio data yet
            
        except Exception as e:
            raise Exception(f"Error processing audio stream: {str(e)}")
            
    def reset_buffer(self):
        """Reset the audio buffer"""
        self.audio_buffer = [] 