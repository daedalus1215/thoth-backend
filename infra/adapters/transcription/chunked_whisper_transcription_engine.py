from domain.entities.audio_file import AudioFile
from domain.entities.transcription import Transcription
from domain.ports.audio_processor import TranscriptionEngine
from domain.value_objects.audio_config import ModelConfig
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from accelerate import Accelerator
import torch
import numpy as np
import librosa
import io
from typing import Optional, List
import asyncio


class ChunkedWhisperTranscriptionEngine(TranscriptionEngine):
    """Chunked Whisper transcription engine for handling long audio files"""
    
    def __init__(self, model_config: ModelConfig, chunk_duration_seconds: float = 30.0):
        self.model_config = model_config
        self.chunk_duration_seconds = chunk_duration_seconds
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
        
        print(f"Chunked transcription engine initialized with {chunk_duration_seconds}s chunks")
    
    async def transcribe_audio(self, audio_file: AudioFile) -> Transcription:
        """Transcribe audio file to text using chunked Whisper processing"""
        try:
            # Try to load audio with librosa, with fallback to different methods
            audio_data, sample_rate = await self._load_audio_data(audio_file)
            
            duration = len(audio_data) / sample_rate
            print(f"Audio duration: {duration:.2f} seconds")
            
            # If audio is short enough, process normally
            if duration <= self.chunk_duration_seconds:
                return await self._transcribe_single_chunk(audio_data, sample_rate)
            
            # For longer audio, split into chunks
            chunks = self._split_audio_into_chunks(audio_data, sample_rate)
            print(f"Split audio into {len(chunks)} chunks")
            
            # Transcribe each chunk
            transcriptions = []
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}")
                chunk_transcription = await self._transcribe_single_chunk(chunk, sample_rate)
                if chunk_transcription and chunk_transcription.text.strip():
                    transcriptions.append(chunk_transcription.text.strip())
            
            # Combine all transcriptions
            if transcriptions:
                combined_text = " ".join(transcriptions)
                return Transcription(text=combined_text)
            else:
                return Transcription(text="")
                
        except Exception as e:
            raise ValueError(f"Failed to transcribe audio: {str(e)}")
    
    async def _load_audio_data(self, audio_file: AudioFile) -> tuple[np.ndarray, int]:
        """Load audio data with multiple fallback methods"""
        print(f"Loading audio file: {audio_file.filename}, size: {len(audio_file.content)} bytes, type: {audio_file.content_type}")
        
        try:
            # Method 1: Try librosa with BytesIO
            print("Method 1: Trying librosa with BytesIO...")
            audio_data, sample_rate = librosa.load(
                io.BytesIO(audio_file.content),
                sr=16000,
                mono=True
            )
            print(f"✅ Method 1 succeeded: {len(audio_data)} samples at {sample_rate}Hz")
            return audio_data, sample_rate
        except Exception as e1:
            print(f"❌ Method 1 failed: {e1}")
            
            try:
                # Method 2: Try librosa with file extension hint
                print("Method 2: Trying librosa with temporary file...")
                import tempfile
                import os
                
                # Create temporary file with proper extension
                file_extension = audio_file.get_file_extension()
                if not file_extension:
                    file_extension = 'wav'  # Default fallback
                
                print(f"Using file extension: {file_extension}")
                
                with tempfile.NamedTemporaryFile(suffix=f'.{file_extension}', delete=False) as temp_file:
                    temp_file.write(audio_file.content)
                    temp_file_path = temp_file.name
                
                try:
                    audio_data, sample_rate = librosa.load(
                        temp_file_path,
                        sr=16000,
                        mono=True
                    )
                    print(f"✅ Method 2 succeeded: {len(audio_data)} samples at {sample_rate}Hz")
                    return audio_data, sample_rate
                finally:
                    # Clean up temporary file
                    os.unlink(temp_file_path)
                    
            except Exception as e2:
                print(f"❌ Method 2 failed: {e2}")
                
                try:
                    # Method 3: Try soundfile directly
                    print("Method 3: Trying soundfile...")
                    import soundfile as sf
                    
                    with io.BytesIO(audio_file.content) as audio_buffer:
                        audio_data, sample_rate = sf.read(audio_buffer)
                        
                        # Convert to mono if stereo
                        if len(audio_data.shape) > 1:
                            audio_data = np.mean(audio_data, axis=1)
                        
                        # Resample to 16kHz if needed
                        if sample_rate != 16000:
                            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                            sample_rate = 16000
                        
                        print(f"✅ Method 3 succeeded: {len(audio_data)} samples at {sample_rate}Hz")
                        return audio_data, sample_rate
                        
                except Exception as e3:
                    print(f"❌ Method 3 failed: {e3}")
                    
                    # Method 4: Fallback to raw bytes interpretation
                    print("Method 4: Using fallback raw bytes method...")
                    try:
                        audio_data = np.frombuffer(audio_file.content, dtype=np.int16).astype(np.float32) / 32767.0
                        sample_rate = 16000  # Assume 16kHz
                        print(f"✅ Method 4 succeeded: {len(audio_data)} samples at {sample_rate}Hz")
                        return audio_data, sample_rate
                    except Exception as e4:
                        print(f"❌ Method 4 failed: {e4}")
                        raise ValueError(f"All audio loading methods failed. File: {audio_file.filename}, Size: {len(audio_file.content)} bytes, Type: {audio_file.content_type}")
    
    def _split_audio_into_chunks(self, audio_data: np.ndarray, sample_rate: int) -> List[np.ndarray]:
        """Split audio into overlapping chunks"""
        chunk_samples = int(self.chunk_duration_seconds * sample_rate)
        overlap_samples = int(2.0 * sample_rate)  # 2 second overlap
        
        chunks = []
        start = 0
        
        while start < len(audio_data):
            end = min(start + chunk_samples, len(audio_data))
            chunk = audio_data[start:end]
            chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap_samples
            if start >= len(audio_data):
                break
        
        return chunks
    
    async def _transcribe_single_chunk(self, audio_data: np.ndarray, sample_rate: int) -> Optional[Transcription]:
        """Transcribe a single audio chunk"""
        try:
            # Process with Whisper using accelerator
            with self.accelerator.autocast():
                input_features = self.processor(
                    audio_data,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                ).input_features
                
                # Move to accelerator device
                input_features = input_features.to(self.accelerator.device)
                
                # Ensure model and input are on the same device
                if self.model.device != input_features.device:
                    input_features = input_features.to(self.model.device)
                
                # Generate transcription with optimized settings
                predicted_ids = self.model.generate(
                    input_features,
                    max_length=self.model_config.max_length,
                    num_beams=self.model_config.num_beams,
                    do_sample=self.model_config.do_sample,
                    early_stopping=self.model_config.early_stopping,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True
                )
                
                # Decode to text
                transcription_text = self.processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0]
                
                return Transcription(text=transcription_text)
                
        except Exception as e:
            print(f"Error transcribing chunk: {str(e)}")
            return None
    
    async def transcribe_stream_chunk(self, audio_chunk: bytes) -> Optional[Transcription]:
        """Transcribe a chunk of streaming audio"""
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
                    input_features = input_features.to(self.model.device)
                
                # Generate transcription
                predicted_ids = self.model.generate(
                    input_features,
                    max_length=self.model_config.max_length,
                    num_beams=self.model_config.num_beams,
                    do_sample=self.model_config.do_sample,
                    early_stopping=self.model_config.early_stopping,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True
                )
                
                # Decode to text
                transcription_text = self.processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0]
                
                return Transcription(text=transcription_text)
                
        except Exception as e:
            raise ValueError(f"Failed to transcribe stream chunk: {str(e)}")
    
    def reset_stream_state(self) -> None:
        """Reset the state for streaming transcription"""
        if hasattr(self.accelerator, 'clear_cache'):
            self.accelerator.clear_cache()
    
    def get_device_info(self) -> dict:
        """Get information about the accelerator device"""
        return {
            "device": str(self.accelerator.device),
            "mixed_precision": self.accelerator.mixed_precision,
            "num_processes": self.accelerator.num_processes,
            "is_main_process": self.accelerator.is_main_process,
            "chunk_duration": self.chunk_duration_seconds,
            "engine_type": "chunked"
        }
