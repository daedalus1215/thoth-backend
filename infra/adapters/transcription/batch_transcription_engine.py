from domain.entities.audio_file import AudioFile
from domain.entities.transcription import Transcription
from domain.ports.audio_processor import TranscriptionEngine
from domain.value_objects.audio_config import ModelConfig
from accelerate import Accelerator
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import numpy as np
from typing import List, Optional
import asyncio


class BatchTranscriptionEngine(TranscriptionEngine):
    """Batch processing transcription engine for handling multiple audio files efficiently"""
    
    def __init__(self, model_config: ModelConfig, batch_size: int = 4):
        self.model_config = model_config
        self.batch_size = batch_size
        self.accelerator = Accelerator()
        
        # Load model and processor
        self.processor = WhisperProcessor.from_pretrained(model_config.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_config.model_name)
        
        # Move model to accelerator device
        self.model = self.model.to(self.accelerator.device)
        
        # Batch processing queue
        self.pending_audio_files: List[AudioFile] = []
        self.pending_stream_chunks: List[bytes] = []
    
    async def transcribe_audio(self, audio_file: AudioFile) -> Transcription:
        """Transcribe audio file - adds to batch for processing"""
        # For single file, process immediately
        return await self._process_single_audio(audio_file)
    
    async def transcribe_batch(self, audio_files: List[AudioFile]) -> List[Transcription]:
        """Transcribe multiple audio files in batches for efficiency"""
        transcriptions = []
        
        # Process in batches
        for i in range(0, len(audio_files), self.batch_size):
            batch = audio_files[i:i + self.batch_size]
            batch_transcriptions = await self._process_batch_audio(batch)
            transcriptions.extend(batch_transcriptions)
        
        return transcriptions
    
    async def _process_single_audio(self, audio_file: AudioFile) -> Transcription:
        """Process a single audio file"""
        try:
            # Convert audio bytes to numpy array
            audio_data = np.frombuffer(audio_file.content, dtype=np.int16).astype(np.float32) / 32767.0
            
            # Process with Whisper
            with self.accelerator.autocast():
                input_features = self.processor(
                    audio_data,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features
                
                input_features = input_features.to(self.accelerator.device)
                
                # Convert input to match model dtype (float16 if model is in half precision)
                model_dtype = next(self.model.parameters()).dtype
                if input_features.dtype != model_dtype:
                    input_features = input_features.to(dtype=model_dtype)
                
                # Generate with memory optimization
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
    
    async def _process_batch_audio(self, audio_files: List[AudioFile]) -> List[Transcription]:
        """Process a batch of audio files efficiently"""
        try:
            # Prepare batch input features
            batch_features = []
            
            for audio_file in audio_files:
                audio_data = np.frombuffer(audio_file.content, dtype=np.int16).astype(np.float32) / 32767.0
                input_features = self.processor(
                    audio_data,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features
                batch_features.append(input_features)
            
            # Stack features for batch processing
            batch_input_features = torch.cat(batch_features, dim=0)
            batch_input_features = batch_input_features.to(self.accelerator.device)
            
            # Convert input to match model dtype (float16 if model is in half precision)
            model_dtype = next(self.model.parameters()).dtype
            if batch_input_features.dtype != model_dtype:
                batch_input_features = batch_input_features.to(dtype=model_dtype)
            
            # Generate transcriptions for the entire batch
            with self.accelerator.autocast():
                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        batch_input_features,
                        max_length=self.model_config.max_length,
                        num_beams=self.model_config.num_beams,
                        do_sample=self.model_config.do_sample,
                        early_stopping=self.model_config.early_stopping,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=False  # Disable cache to save memory
                    )
                
                # Decode all transcriptions
                transcriptions_text = self.processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )
                
                # Convert to Transcription objects
                transcriptions = [Transcription(text=text) for text in transcriptions_text]
                
                # Clean up GPU memory
                if torch.cuda.is_available():
                    del batch_input_features, predicted_ids
                    torch.cuda.empty_cache()
                
                return transcriptions
                
        except Exception as e:
            raise ValueError(f"Failed to transcribe batch: {str(e)}")
    
    async def transcribe_stream_chunk(self, audio_chunk: bytes) -> Optional[Transcription]:
        """Transcribe a chunk of streaming audio"""
        # For streaming, process immediately
        try:
            audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
            
            with self.accelerator.autocast():
                input_features = self.processor(
                    audio_data,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features
                
                input_features = input_features.to(self.accelerator.device)
                
                # Convert input to match model dtype (float16 if model is in half precision)
                model_dtype = next(self.model.parameters()).dtype
                if input_features.dtype != model_dtype:
                    input_features = input_features.to(dtype=model_dtype)
                
                # Generate with memory optimization
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
        if hasattr(self.accelerator, 'clear_cache'):
            self.accelerator.clear_cache()
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        return {
            "batch_size": self.batch_size,
            "device": str(self.accelerator.device),
            "mixed_precision": self.accelerator.mixed_precision,
            "num_processes": self.accelerator.num_processes
        }
