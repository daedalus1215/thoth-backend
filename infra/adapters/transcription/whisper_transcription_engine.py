from domain.entities.audio_file import AudioFile
from domain.entities.transcription import Transcription
from domain.ports.audio_processor import TranscriptionEngine
from domain.value_objects.audio_config import ModelConfig
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import numpy as np
from typing import Optional


class WhisperTranscriptionEngine(TranscriptionEngine):
    """Infrastructure adapter for transcription using Whisper"""
    
    def __init__(self, model_config: ModelConfig, device: str = None):
        self.model_config = model_config
        self.processor = WhisperProcessor.from_pretrained(model_config.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_config.model_name)
        
        # Use specified device or auto-detect
        if device:
            self.model = self.model.to(device)
            print(f"Using specified device: {device}")
        elif torch.cuda.is_available():
            self.model = self.model.to("cuda")
            print("Using CUDA device")
        else:
            print("Using CPU device")
    
    async def transcribe_audio(self, audio_file: AudioFile) -> Transcription:
        """Transcribe audio file to text using Whisper"""
        try:
            # Convert audio bytes to numpy array
            audio_data = np.frombuffer(audio_file.content, dtype=np.int16).astype(np.float32) / 32767.0
            
            # Process with Whisper
            input_features = self.processor(
                audio_data,
                sampling_rate=16000,  # Assuming 16kHz sample rate
                return_tensors="pt"
            ).input_features
            
            # Move input features to the same device as the model
            input_features = input_features.to(self.model.device)
            
            # Convert input to match model dtype (float16 if model is in half precision)
            model_dtype = next(self.model.parameters()).dtype
            if input_features.dtype != model_dtype:
                input_features = input_features.to(dtype=model_dtype)
            
            # Generate transcription with memory optimization
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    max_length=self.model_config.max_length,
                    num_beams=self.model_config.num_beams,
                    do_sample=self.model_config.do_sample,
                    early_stopping=self.model_config.early_stopping,
                    use_cache=False  # Disable cache to save memory
                )
            
            # Decode to text
            transcription_text = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                del input_features, predicted_ids
                torch.cuda.empty_cache()
            
            return Transcription(text=transcription_text)
            
        except Exception as e:
            raise ValueError(f"Failed to transcribe audio: {str(e)}")
    
    async def transcribe_stream_chunk(self, audio_chunk: bytes) -> Optional[Transcription]:
        """Transcribe a chunk of streaming audio"""
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
            
            # Process with Whisper
            input_features = self.processor(
                audio_data,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features
            
            # Move input features to the same device as the model
            input_features = input_features.to(self.model.device)
            
            # Convert input to match model dtype (float16 if model is in half precision)
            model_dtype = next(self.model.parameters()).dtype
            if input_features.dtype != model_dtype:
                input_features = input_features.to(dtype=model_dtype)
            
            # Generate transcription with memory optimization
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    max_length=self.model_config.max_length,
                    num_beams=self.model_config.num_beams,
                    do_sample=self.model_config.do_sample,
                    early_stopping=self.model_config.early_stopping,
                    use_cache=False  # Disable cache to save memory
                )
            
            # Decode to text
            transcription_text = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                del input_features, predicted_ids
                torch.cuda.empty_cache()
            
            return Transcription(text=transcription_text)
            
        except Exception as e:
            raise ValueError(f"Failed to transcribe stream chunk: {str(e)}")
    
    def reset_stream_state(self) -> None:
        """Reset the state for streaming transcription"""
        # Whisper doesn't maintain state between calls, so nothing to reset
        pass
