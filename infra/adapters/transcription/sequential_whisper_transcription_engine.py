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
from infra.adapters.transcription.shared_model_manager import SharedModelManager


class SequentialWhisperTranscriptionEngine(TranscriptionEngine):
    """Sequential Whisper transcription engine using sliding window approach for better accuracy"""
    
    def __init__(self, model_config: ModelConfig, chunk_duration_seconds: float = 30.0):
        self.model_config = model_config
        self.chunk_duration_seconds = chunk_duration_seconds
        
        # Use shared model manager to avoid loading duplicate models
        self._model_manager = SharedModelManager()
        self.model, self.processor, self.accelerator = self._model_manager.get_model(model_config)
        
        print(f"Sequential transcription engine initialized with {chunk_duration_seconds}s sliding window")
    
    async def transcribe_audio(self, audio_file: AudioFile) -> Transcription:
        """Transcribe audio file to text using sequential Whisper processing with sliding window"""
        try:
            # Load audio data
            audio_data, sample_rate = await self._load_audio_data(audio_file)
            
            duration = len(audio_data) / sample_rate
            print(f"Audio duration: {duration:.2f} seconds")
            
            # If audio is short enough, process normally
            if duration <= self.chunk_duration_seconds:
                return await self._transcribe_single_chunk(audio_data, sample_rate)
            
            # For longer audio, use sequential sliding window approach
            return await self._transcribe_sequential(audio_data, sample_rate)
                
        except Exception as e:
            raise ValueError(f"Failed to transcribe audio: {str(e)}")
    
    async def _transcribe_sequential(self, audio_data: np.ndarray, sample_rate: int) -> Transcription:
        """Transcribe using sequential sliding window approach for better accuracy"""
        chunk_samples = int(self.chunk_duration_seconds * sample_rate)
        overlap_samples = int(2.0 * sample_rate)  # 2 second overlap for context
        
        print(f"Using sequential sliding window approach with {self.chunk_duration_seconds}s windows")
        print(f"📊 Audio data: {len(audio_data)} samples at {sample_rate}Hz")
        print(f"📊 Expected duration: {len(audio_data)/sample_rate:.1f}s")
        print(f"📊 Chunk samples: {chunk_samples}")
        print(f"📊 Overlap samples: {overlap_samples}")
        print(f"📊 Step size: {chunk_samples - overlap_samples}")
        
        # Initialize sliding window buffer
        sliding_window = []
        transcriptions = []
        chunk_details = []
        
        start = 0
        chunk_index = 0
        total_audio_duration = len(audio_data) / sample_rate
        
        while start < len(audio_data):
            end = min(start + chunk_samples, len(audio_data))
            chunk = audio_data[start:end]
            chunk_duration = len(chunk) / sample_rate
            
            # Add chunk to sliding window
            sliding_window.append(chunk)
            
            # Keep only the last 2 chunks for context (current + previous)
            if len(sliding_window) > 2:
                sliding_window.pop(0)
            
            print(f"🔄 Processing sequential chunk {chunk_index + 1} (start: {start/sample_rate:.1f}s, end: {end/sample_rate:.1f}s, duration: {chunk_duration:.1f}s)")
            
            # Transcribe with context from sliding window
            chunk_transcription = await self._transcribe_with_context(sliding_window, sample_rate)
            
            if chunk_transcription and chunk_transcription.text.strip():
                word_count = len(chunk_transcription.text.strip().split())
                transcriptions.append(chunk_transcription.text.strip())
                chunk_details.append({
                    'index': chunk_index + 1,
                    'start': start/sample_rate,
                    'end': end/sample_rate,
                    'duration': chunk_duration,
                    'words': word_count,
                    'text': chunk_transcription.text.strip()[:100] + "..." if len(chunk_transcription.text.strip()) > 100 else chunk_transcription.text.strip()
                })
                print(f"✅ Chunk {chunk_index + 1} completed: '{chunk_transcription.text.strip()[:50]}...' ({word_count} words, {chunk_duration:.1f}s)")
            else:
                print(f"⚠️  Chunk {chunk_index + 1} produced no transcription")
                chunk_details.append({
                    'index': chunk_index + 1,
                    'start': start/sample_rate,
                    'end': end/sample_rate,
                    'duration': chunk_duration,
                    'words': 0,
                    'text': '[NO TRANSCRIPTION]'
                })
            
            # Move to next chunk with overlap
            old_start = start
            start = end - overlap_samples
            chunk_index += 1
            
            # Safety check to prevent infinite loops
            if start <= old_start:
                print(f"⚠️  Safety check: start not advancing! start={start}, old_start={old_start}")
                start = old_start + (chunk_samples - overlap_samples)
            
            # Check if we've covered the entire audio
            if start >= len(audio_data):
                print(f"📊 Reached end of audio at chunk {chunk_index}")
                break
        
        # Calculate coverage analysis
        total_chunks = chunk_index
        successful_chunks = len([c for c in chunk_details if c['words'] > 0])
        failed_chunks = total_chunks - successful_chunks
        total_words = sum(c['words'] for c in chunk_details)
        
        print(f"\n🔍 COVERAGE ANALYSIS:")
        print(f"   📊 Total chunks processed: {total_chunks}")
        print(f"   📊 Successful chunks: {successful_chunks}")
        print(f"   📊 Failed chunks: {failed_chunks}")
        print(f"   📊 Audio duration: {total_audio_duration:.1f}s")
        print(f"   📊 Expected coverage: {total_chunks * self.chunk_duration_seconds:.1f}s")
        print(f"   📊 Coverage percentage: {(total_chunks * self.chunk_duration_seconds / total_audio_duration * 100):.1f}%")
        
        if failed_chunks > 0:
            print(f"   ⚠️  Failed chunks: {[c['index'] for c in chunk_details if c['words'] == 0]}")
        
        # Combine all transcriptions
        if transcriptions:
            combined_text = " ".join(transcriptions)
            words_per_minute = (total_words / total_audio_duration) * 60 if total_audio_duration > 0 else 0
            
            print(f"\n✅ FINAL RESULTS:")
            print(f"   📊 Total words: {total_words}")
            print(f"   📊 Total characters: {len(combined_text)}")
            print(f"   📊 Audio duration: {total_audio_duration/60:.1f} minutes")
            print(f"   📊 Words per minute: {words_per_minute:.1f}")
            print(f"   📊 Preview: '{combined_text[:200]}...'")
            
            if words_per_minute < 100:
                print(f"   ⚠️  WARNING: Very low WPM ({words_per_minute:.1f}). Expected 150-200 for normal speech.")
            
            return Transcription(text=combined_text)
        else:
            print(f"❌ No transcriptions produced!")
            return Transcription(text="")
    
    async def _transcribe_with_context(self, sliding_window: List[np.ndarray], sample_rate: int) -> Optional[Transcription]:
        """Transcribe a chunk with context from the sliding window"""
        try:
            # Use the most recent chunk for transcription
            current_chunk = sliding_window[-1]
            
            # Process with Whisper processor (outside autocast to control dtype)
            input_features = self.processor(
                current_chunk,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_features
            
            # Clear CUDA cache before moving tensors to avoid OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Move to accelerator device
            try:
                input_features = input_features.to(self.accelerator.device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                input_features = input_features.to(self.accelerator.device)
            
            # Ensure model and input are on the same device
            if self.model.device != input_features.device:
                try:
                    input_features = input_features.to(self.model.device)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    input_features = input_features.to(self.model.device)
            
            # Convert input to match model dtype (float16 if model is in half precision)
            # This MUST happen before generation to ensure correct dtype
            model_dtype = next(self.model.parameters()).dtype
            if input_features.dtype != model_dtype:
                input_features = input_features.to(dtype=model_dtype)
            
            # Generate transcription with optimized settings for sequential processing
            # Use torch.no_grad() to reduce memory usage during inference
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    max_length=self.model_config.max_length,
                    num_beams=self.model_config.num_beams,
                    do_sample=self.model_config.do_sample,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=False,  # Disable cache to save memory
                    language="en",  # Force English to avoid language detection issues
                    task="transcribe"  # Explicitly set task
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
            print(f"Error transcribing sequential chunk: {str(e)}")
            return None
    
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
    
    async def _transcribe_single_chunk(self, audio_data: np.ndarray, sample_rate: int) -> Optional[Transcription]:
        """Transcribe a single audio chunk (for short audio files)"""
        try:
            # Process with Whisper using accelerator
            with self.accelerator.autocast():
                input_features = self.processor(
                    audio_data,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                ).input_features
                
                # Clear CUDA cache before moving tensors to avoid OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Move to accelerator device
                try:
                    input_features = input_features.to(self.accelerator.device)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    input_features = input_features.to(self.accelerator.device)
                
                # Ensure model and input are on the same device
                if self.model.device != input_features.device:
                    try:
                        input_features = input_features.to(self.model.device)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        input_features = input_features.to(self.model.device)
                
                # Generate transcription with optimized settings
                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        input_features,
                        max_length=self.model_config.max_length,
                        num_beams=self.model_config.num_beams,
                        do_sample=self.model_config.do_sample,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=False,  # Disable cache to save memory
                        language="en",  # Force English to avoid language detection issues
                        task="transcribe"  # Explicitly set task
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
            print(f"Error transcribing single chunk: {str(e)}")
            return None
    
    async def transcribe_stream_chunk(self, audio_chunk: bytes) -> Optional[Transcription]:
        """Transcribe a chunk of streaming audio (not recommended for sequential)"""
        # Sequential is not ideal for streaming, but provide fallback
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
                
                # Clear CUDA cache before moving tensors to avoid OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Move to accelerator device
                try:
                    input_features = input_features.to(self.accelerator.device)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    input_features = input_features.to(self.accelerator.device)
                
                # Ensure model and input are on the same device
                if self.model.device != input_features.device:
                    try:
                        input_features = input_features.to(self.model.device)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        input_features = input_features.to(self.model.device)
                
                # Generate transcription with memory optimization
                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        input_features,
                        max_length=self.model_config.max_length,
                        num_beams=self.model_config.num_beams,
                        do_sample=self.model_config.do_sample,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=False,  # Disable cache to save memory
                        language="en",  # Force English to avoid language detection issues
                        task="transcribe"  # Explicitly set task
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
            "engine_type": "sequential"
        }


