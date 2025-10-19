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
        import asyncio
        
        try:
            print(f"🎵 Starting transcription of file: {audio_file.filename}")
            
            # Get duration first using the original method (more reliable)
            print("🔄 Getting audio duration...")
            duration = await self._get_audio_duration_from_bytes(audio_file)
            print(f"✅ Audio duration: {duration:.2f} seconds")
            
            # For short audio (≤60s), process directly without temporary files
            if duration <= 60.0:
                print(f"📝 Processing as single chunk (≤60s) - no chunking needed")
                result = await self._transcribe_short_audio_direct(audio_file)
                print(f"✅ Direct transcription completed!")
                return result
            
            # For longer audio, use chunking but still process from bytes
            print(f"🔄 Processing long audio ({duration:.1f}s) with chunking from bytes...")
            result = await self._transcribe_long_audio_from_bytes(audio_file, duration)
            print(f"✅ Long audio transcription completed!")
            return result
                
        except Exception as e:
            print(f"❌ Failed to transcribe audio: {str(e)}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Failed to transcribe audio: {str(e)}")
    
    async def _get_audio_duration_from_bytes(self, audio_file: AudioFile) -> float:
        """Get audio duration from bytes without saving to file"""
        try:
            # Try to load audio directly from bytes
            import librosa
            import io
            
            print("🔄 Loading audio from bytes to get duration...")
            audio_data, sample_rate = librosa.load(
                io.BytesIO(audio_file.content),
                sr=16000,
                mono=True
            )
            duration = len(audio_data) / sample_rate
            print(f"✅ Duration calculated: {duration:.2f}s from {len(audio_data)} samples")
            return duration
            
        except Exception as e:
            print(f"⚠️  Failed to get duration from bytes: {e}")
            # Fallback: estimate from file size
            file_size = len(audio_file.content)
            # Rough estimate: 16kHz, 16-bit mono = 32KB per second
            estimated_duration = file_size / 32000
            print(f"📊 Estimated duration from file size: {estimated_duration:.2f}s")
            return estimated_duration
    
    async def _transcribe_short_audio_direct(self, audio_file: AudioFile) -> Transcription:
        """Transcribe short audio directly from bytes without temporary files"""
        try:
            print("🔄 Loading audio for direct transcription...")
            import librosa
            import io
            
            audio_data, sample_rate = librosa.load(
                io.BytesIO(audio_file.content),
                sr=16000,
                mono=True
            )
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
    
    
    async def _transcribe_long_audio_from_bytes(self, audio_file: AudioFile, duration: float) -> Transcription:
        """Transcribe long audio using chunking directly from bytes"""
        try:
            # Load audio data from bytes
            print("🔄 Loading long audio from bytes...")
            audio_data, sample_rate = await self._load_audio_from_bytes(audio_file)
            
            # Calculate chunk parameters - optimize for long audio
            if duration > 300:  # > 5 minutes
                chunk_duration = min(60.0, duration / 4)  # Larger chunks for long audio
            else:
                chunk_duration = min(30.0, duration / 2)  # Smaller chunks for short audio
            print(f"🔄 Processing {duration:.1f}s audio in {chunk_duration:.1f}s chunks...")
            print(f"📊 Audio data: {len(audio_data)} samples at {sample_rate}Hz")
            print(f"📊 Expected duration: {len(audio_data) / sample_rate:.1f}s")
            
            chunk_samples = int(chunk_duration * sample_rate)
            overlap_samples = int(2.0 * sample_rate)  # 2 second overlap
            
            print(f"📊 Chunk samples: {chunk_samples}")
            print(f"📊 Overlap samples: {overlap_samples}")
            print(f"📊 Step size: {chunk_samples - overlap_samples}")
            
            transcriptions = []
            start = 0
            chunk_num = 0
            
            # Safety check: prevent infinite loops
            max_expected_chunks = int(duration / chunk_duration) + 10  # Add buffer
            print(f"🛡️  Safety check: Max expected chunks = {max_expected_chunks}")
            
            while start < len(audio_data):
                # Safety check: prevent infinite loops
                if chunk_num > max_expected_chunks:
                    print(f"❌ SAFETY BREAK: Too many chunks ({chunk_num} > {max_expected_chunks})")
                    print(f"   Audio duration: {duration:.1f}s")
                    print(f"   Chunk duration: {chunk_duration:.1f}s")
                    print(f"   Audio data length: {len(audio_data)} samples")
                    print(f"   Sample rate: {sample_rate}Hz")
                    raise ValueError(f"Too many chunks generated ({chunk_num}). Audio file may be corrupted.")
                
                end = min(start + chunk_samples, len(audio_data))
                chunk = audio_data[start:end]
                chunk_num += 1
                
                print(f"🔄 Processing chunk {chunk_num}/{max_expected_chunks} ({start/sample_rate:.1f}s - {end/sample_rate:.1f}s)")
                
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
                old_start = start
                start = end - overlap_samples
                
                print(f"📊 Chunk {chunk_num} progress: {old_start/sample_rate:.1f}s → {start/sample_rate:.1f}s (step: {(start-old_start)/sample_rate:.1f}s)")
                
                # Safety check: ensure we're actually advancing
                if start <= old_start:
                    print(f"❌ CHUNKING ERROR: Not advancing! start={start}, old_start={old_start}")
                    print(f"   end={end}, overlap_samples={overlap_samples}")
                    print(f"   chunk_samples={chunk_samples}")
                    raise ValueError("Chunking logic error: not advancing through audio")
                
                if start >= len(audio_data):
                    print(f"✅ Reached end of audio at sample {len(audio_data)}")
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
    
    async def _load_audio_from_bytes(self, audio_file: AudioFile) -> tuple[np.ndarray, int]:
        """Load audio from bytes with multiple fallback methods"""
        print(f"🔍 Debugging audio file: {audio_file.filename}")
        print(f"   Content type: {audio_file.content_type}")
        print(f"   File size: {len(audio_file.content)} bytes")
        print(f"   First 20 bytes: {audio_file.content[:20].hex()}")
        
        try:
            # Method 1: Try librosa with BytesIO
            print("🔄 Method 1: Loading with librosa from bytes...")
            import librosa
            import io
            
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
                # Method 2: Try with soundfile
                print("🔄 Method 2: Trying with soundfile...")
                import soundfile as sf
                import io
                
                # Read with soundfile from bytes
                audio_data, sample_rate = sf.read(io.BytesIO(audio_file.content))
                
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    import librosa
                    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                    sample_rate = 16000
                
                print(f"✅ Method 2 succeeded: {len(audio_data)} samples at {sample_rate}Hz")
                return audio_data, sample_rate
                
            except Exception as e2:
                print(f"❌ Method 2 failed: {e2}")
                
                try:
                    # Method 3: Try with pydub (optional dependency)
                    print("🔄 Method 3: Trying with pydub...")
                    try:
                        from pydub import AudioSegment
                    except ImportError:
                        print("❌ Pydub not available, skipping method 3")
                        raise ImportError("Pydub not installed")
                    
                    import io
                    
                    # Load with pydub from bytes
                    audio = AudioSegment.from_file(io.BytesIO(audio_file.content))
                    
                    # Convert to mono and 16kHz
                    audio = audio.set_channels(1).set_frame_rate(16000)
                    
                    # Convert to numpy array
                    audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
                    audio_data = audio_data / 32768.0  # Normalize to [-1, 1]
                    sample_rate = 16000
                    
                    print(f"✅ Method 3 succeeded: {len(audio_data)} samples at {sample_rate}Hz")
                    return audio_data, sample_rate
                    
                except Exception as e3:
                    print(f"❌ Method 3 failed: {e3}")
                    
                    try:
                        # Method 4: Try raw audio interpretation
                        print("🔄 Method 4: Trying raw audio interpretation...")
                        
                        # Try different raw audio formats
                        for dtype, sample_rate in [(np.int16, 16000), (np.int32, 16000), (np.float32, 16000)]:
                            try:
                                print(f"   Trying {dtype} at {sample_rate}Hz...")
                                audio_data = np.frombuffer(audio_file.content, dtype=dtype).astype(np.float32)
                                
                                # Normalize based on dtype
                                if dtype == np.int16:
                                    audio_data = audio_data / 32767.0
                                elif dtype == np.int32:
                                    audio_data = audio_data / 2147483647.0
                                # float32 is already normalized
                                
                                # Ensure reasonable length (not too short or too long)
                                if len(audio_data) > 1000 and len(audio_data) < 10000000:  # 0.06s to 10 minutes at 16kHz
                                    print(f"✅ Method 4 succeeded with {dtype}: {len(audio_data)} samples at {sample_rate}Hz")
                                    return audio_data, sample_rate
                                    
                            except Exception as e4:
                                print(f"   Failed with {dtype}: {e4}")
                                continue
                        
                        raise ValueError("Raw audio interpretation failed")
                        
                    except Exception as e4:
                        print(f"❌ Method 4 failed: {e4}")
                        
                        try:
                            # Method 5: Try with different sample rates
                            print("🔄 Method 5: Trying different sample rates...")
                            import librosa
                            import io
                            
                            for sr in [8000, 22050, 44100, 48000]:
                                try:
                                    print(f"   Trying sample rate {sr}Hz...")
                                    audio_data, sample_rate = librosa.load(
                                        io.BytesIO(audio_file.content),
                                        sr=sr,
                                        mono=True
                                    )
                                    
                                    # Resample to 16kHz
                                    if sample_rate != 16000:
                                        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                                        sample_rate = 16000
                                    
                                    print(f"✅ Method 5 succeeded at {sr}Hz: {len(audio_data)} samples at {sample_rate}Hz")
                                    return audio_data, sample_rate
                                    
                                except Exception as e5:
                                    print(f"   Failed at {sr}Hz: {e5}")
                                    continue
                            
                            raise ValueError("All sample rates failed")
                            
                        except Exception as e5:
                            print(f"❌ Method 5 failed: {e5}")
                            raise ValueError(f"All audio loading methods failed for file: {audio_file.filename}. File appears to be corrupted or in an unsupported format.")
    
    
    
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
