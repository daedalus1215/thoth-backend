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
        """Transcribe audio file to text using industry-standard processing"""
        import tempfile
        import os
        import asyncio
        from pathlib import Path
        
        temp_file_path = None
        try:
            print(f"🎵 Starting transcription of file: {audio_file.filename}")
            
            # Industry standard: Save to temporary file first
            print("🔄 Saving audio to temporary file...")
            temp_file_path = await self._save_to_temp_file(audio_file)
            
            # Get audio duration without loading entire file into memory
            print("🔄 Getting audio duration...")
            duration = await self._get_audio_duration(temp_file_path)
            print(f"✅ Audio duration: {duration:.2f} seconds")
            
            # For short audio (≤30s), process directly without chunking
            if duration <= 30.0:
                print(f"📝 Processing as single file (≤30s) - no chunking needed")
                result = await self._transcribe_file_direct(temp_file_path)
                print(f"✅ Direct transcription completed!")
                return result
            
            # For longer audio, use proper chunking with temporary files
            print(f"🔄 Processing long audio ({duration:.1f}s) with chunking...")
            result = await self._transcribe_long_audio(temp_file_path, duration)
            print(f"✅ Long audio transcription completed!")
            return result
                
        except Exception as e:
            print(f"❌ Failed to transcribe audio: {str(e)}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Failed to transcribe audio: {str(e)}")
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    print(f"🧹 Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    print(f"⚠️  Failed to clean up temp file: {e}")
    
    async def _save_to_temp_file(self, audio_file: AudioFile) -> str:
        """Save audio file to temporary file with proper extension"""
        import tempfile
        import os
        
        # Get proper file extension
        file_extension = audio_file.get_file_extension() or 'wav'
        
        # Create temporary file with proper extension
        temp_fd, temp_path = tempfile.mkstemp(suffix=f'.{file_extension}')
        
        try:
            with os.fdopen(temp_fd, 'wb') as temp_file:
                temp_file.write(audio_file.content)
            return temp_path
        except Exception as e:
            os.close(temp_fd)
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
    
    async def _get_audio_duration(self, file_path: str) -> float:
        """Get audio duration without loading entire file into memory"""
        try:
            # Use librosa to get duration efficiently
            import librosa
            duration = librosa.get_duration(path=file_path)
            return duration
        except Exception as e:
            print(f"⚠️  Failed to get duration with librosa: {e}")
            # Fallback: estimate from file size (rough approximation)
            file_size = os.path.getsize(file_path)
            # Rough estimate: 16kHz, 16-bit mono = 32KB per second
            estimated_duration = file_size / 32000
            print(f"📊 Estimated duration from file size: {estimated_duration:.2f}s")
            return estimated_duration
    
    async def _transcribe_file_direct(self, file_path: str) -> Transcription:
        """Transcribe file directly without chunking - industry standard for short audio"""
        try:
            print("🔄 Loading audio for direct transcription...")
            audio_data, sample_rate = librosa.load(file_path, sr=16000, mono=True)
            print(f"✅ Loaded {len(audio_data)} samples at {sample_rate}Hz")
            
            # Process with timeout
            print("🔄 Starting transcription with 60s timeout...")
            result = await asyncio.wait_for(
                self._transcribe_single_chunk(audio_data, sample_rate),
                timeout=60.0  # 60 second timeout
            )
            
            if result:
                print(f"✅ Direct transcription completed: '{result.text[:50]}...'")
                return result
            else:
                print("⚠️  No transcription produced")
                return Transcription(text="")
                
        except asyncio.TimeoutError:
            print("❌ Transcription timed out after 60 seconds")
            raise ValueError("Transcription timed out - audio may be too long or complex")
        except Exception as e:
            print(f"❌ Direct transcription failed: {e}")
            raise e
    
    async def _transcribe_long_audio(self, file_path: str, duration: float) -> Transcription:
        """Transcribe long audio using proper chunking with temporary files"""
        try:
            # Load audio in chunks to avoid memory issues
            chunk_duration = min(30.0, duration / 2)  # Adaptive chunk size
            print(f"🔄 Processing {duration:.1f}s audio in {chunk_duration:.1f}s chunks...")
            
            # Use librosa to process in chunks
            audio_data, sample_rate = librosa.load(file_path, sr=16000, mono=True)
            chunk_samples = int(chunk_duration * sample_rate)
            overlap_samples = int(2.0 * sample_rate)  # 2 second overlap
            
            transcriptions = []
            start = 0
            chunk_num = 0
            
            while start < len(audio_data):
                end = min(start + chunk_samples, len(audio_data))
                chunk = audio_data[start:end]
                chunk_num += 1
                
                print(f"🔄 Processing chunk {chunk_num} ({start/sample_rate:.1f}s - {end/sample_rate:.1f}s)")
                
                # Process chunk with timeout
                try:
                    chunk_result = await asyncio.wait_for(
                        self._transcribe_single_chunk(chunk, sample_rate),
                        timeout=30.0  # 30 second timeout per chunk
                    )
                    
                    if chunk_result and chunk_result.text.strip():
                        transcriptions.append(chunk_result.text.strip())
                        print(f"✅ Chunk {chunk_num} completed: '{chunk_result.text[:30]}...'")
                    else:
                        print(f"⚠️  Chunk {chunk_num} produced no transcription")
                        
                except asyncio.TimeoutError:
                    print(f"⚠️  Chunk {chunk_num} timed out - skipping")
                    continue
                
                # Move to next chunk with overlap
                start = end - overlap_samples
                if start >= len(audio_data):
                    break
            
            # Combine results
            if transcriptions:
                combined_text = " ".join(transcriptions)
                print(f"✅ Combined {len(transcriptions)} chunks: '{combined_text[:100]}...'")
                return Transcription(text=combined_text)
            else:
                print("⚠️  No transcriptions produced from any chunks")
                return Transcription(text="")
                
        except Exception as e:
            print(f"❌ Long audio transcription failed: {e}")
            raise e
    
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
            print(f"🔄 Starting transcription of chunk: {len(audio_data)} samples at {sample_rate}Hz")
            
            # Process with Whisper using accelerator
            print("🔄 Processing audio with Whisper processor...")
            with self.accelerator.autocast():
                input_features = self.processor(
                    audio_data,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                ).input_features
                print(f"✅ Processor completed. Input shape: {input_features.shape}")
                
                # Move to accelerator device
                print(f"🔄 Moving to device: {self.accelerator.device}")
                input_features = input_features.to(self.accelerator.device)
                
                # Ensure model and input are on the same device
                if self.model.device != input_features.device:
                    print(f"🔄 Moving input to model device: {self.model.device}")
                    input_features = input_features.to(self.model.device)
                
                # Generate transcription with optimized settings
                print("🔄 Starting Whisper model generation (this is where it might hang)...")
                print(f"   Model: {self.model_config.model_name}")
                print(f"   Max length: {self.model_config.max_length}")
                print(f"   Device: {self.model.device}")
                
                predicted_ids = self.model.generate(
                    input_features,
                    max_length=self.model_config.max_length,
                    num_beams=self.model_config.num_beams,
                    do_sample=self.model_config.do_sample,
                    early_stopping=self.model_config.early_stopping,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True
                )
                print("✅ Model generation completed!")
                
                # Decode to text
                print("🔄 Decoding transcription...")
                transcription_text = self.processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0]
                print(f"✅ Transcription completed: '{transcription_text[:50]}...'")
                
                return Transcription(text=transcription_text)
                
        except Exception as e:
            print(f"❌ Error transcribing chunk: {str(e)}")
            import traceback
            traceback.print_exc()
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
