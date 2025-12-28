from domain.entities.audio_file import AudioFile
from domain.entities.transcription import Transcription
from domain.ports.audio_processor import TranscriptionEngine
from domain.value_objects.audio_config import ModelConfig
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from accelerate import Accelerator
import torch
import numpy as np
from typing import Optional


class AcceleratedWhisperTranscriptionEngine(TranscriptionEngine):
    """Optimized Whisper transcription engine using Hugging Face Accelerate"""
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.accelerator = Accelerator()
        
        # Load model and processor
        self.processor = WhisperProcessor.from_pretrained(model_config.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_config.model_name)
        
        # Move model to accelerator device
        self.model = self.model.to(self.accelerator.device)
        
        # Enable mixed precision if supported and on GPU
        if torch.cuda.is_available() and self.accelerator.device.type == 'cuda':
            if self.accelerator.mixed_precision == "fp16":
                print("Using FP16 mixed precision for faster inference")
            elif self.accelerator.mixed_precision == "bf16":
                print("Using BF16 mixed precision for faster inference")
        else:
            print(f"Using CPU inference on device: {self.accelerator.device}")
    
    async def transcribe_audio(self, audio_file: AudioFile) -> Transcription:
        """Transcribe audio file to text using accelerated Whisper"""
        try:
            # Convert audio bytes to numpy array
            audio_data = np.frombuffer(audio_file.content, dtype=np.int16).astype(np.float32) / 32767.0
            
            # Process with Whisper using accelerator
            with self.accelerator.autocast():
                input_features = self.processor(
                    audio_data,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features
                
                # Move to accelerator device
                input_features = input_features.to(self.accelerator.device)
                
                # Ensure model and input are on the same device
                if self.model.device != input_features.device:
                    print(f"Warning: Model on {self.model.device}, input on {input_features.device}")
                    input_features = input_features.to(self.model.device)
                
                # Convert input to match model dtype (float16 if model is in half precision)
                model_dtype = next(self.model.parameters()).dtype
                if input_features.dtype != model_dtype:
                    input_features = input_features.to(dtype=model_dtype)
                
                # Generate transcription with optimized settings
                # Use torch.no_grad() to reduce memory usage during inference
                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        input_features,
                        max_length=self.model_config.max_length,
                        num_beams=self.model_config.num_beams,
                        do_sample=self.model_config.do_sample,
                        early_stopping=self.model_config.early_stopping,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
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
        """Transcribe a chunk of streaming audio using acceleration"""
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
            
            # Process with Whisper using accelerator
            with self.accelerator.autocast():
                input_features = self.processor(
                    audio_data,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features
                
                # Move to accelerator device
                input_features = input_features.to(self.accelerator.device)
                
                # Ensure model and input are on the same device
                if self.model.device != input_features.device:
                    print(f"Warning: Model on {self.model.device}, input on {input_features.device}")
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
                        pad_token_id=self.processor.tokenizer.eos_token_id,
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
        # Clear any cached states in the accelerator
        if hasattr(self.accelerator, 'clear_cache'):
            self.accelerator.clear_cache()
    
    def get_device_info(self) -> dict:
        """Get information about the accelerator device"""
        return {
            "device": str(self.accelerator.device),
            "mixed_precision": self.accelerator.mixed_precision,
            "num_processes": self.accelerator.num_processes,
            "is_main_process": self.accelerator.is_main_process
        }
