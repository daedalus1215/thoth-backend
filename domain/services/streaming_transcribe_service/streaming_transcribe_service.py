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
        
        # Voice activity detection parameters
        self.silence_threshold = 0.01  # Adjust based on your microphone sensitivity
        self.min_audio_length = 0.5  # Minimum audio length in seconds to process
        self.confidence_threshold = 0.3  # Minimum confidence for transcription
        
    def _is_silence(self, audio_data: np.ndarray) -> bool:
        """Check if audio data is mostly silence"""
        rms = np.sqrt(np.mean(audio_data**2))
        return rms < self.silence_threshold
    
    def _has_sufficient_audio(self, audio_data: np.ndarray) -> bool:
        """Check if audio data is long enough to process"""
        duration = len(audio_data) / self.sample_rate
        return duration >= self.min_audio_length
    
    def _filter_transcription(self, transcription: str) -> str | None:
        """Filter out low-quality transcriptions"""
        if not transcription or len(transcription.strip()) == 0:
            return None
            
        # Remove common filler words that appear during silence
        filler_words = ["thank you", "thanks", "um", "uh", "hmm", "mm", "yeah", "yes", "no"]
        transcription_lower = transcription.lower().strip()
        
        # If it's just a filler word, ignore it
        if transcription_lower in filler_words:
            return None
            
        # If it's very short and common, might be noise
        if len(transcription.strip()) < 3:
            return None
            
        return transcription
        
    async def process_chunk(self, audio_chunk: bytes) -> str | None:
        """
        Process an incoming audio chunk. Returns transcription only when 
        enough audio has been buffered and it's not silence.
        """
        try:
            # Convert bytes to numpy array (assuming float32 audio data)
            chunk_np = np.frombuffer(audio_chunk, dtype=np.float32)
            self.audio_buffer.extend(chunk_np)
            
            # If we have enough audio data, process it
            if len(self.audio_buffer) >= self.buffer_size:
                # Convert buffer to numpy array
                audio_data = np.array(self.audio_buffer, dtype=np.float32)
                
                # Check if audio is silence or too short
                if self._is_silence(audio_data) or not self._has_sufficient_audio(audio_data):
                    # Clear buffer and return None for silence
                    self.audio_buffer = self.audio_buffer[self.buffer_size:]
                    return None
                
                # Process audio with Whisper
                input_features = self.processor(
                    audio_data,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt"
                ).input_features
                
                if torch.cuda.is_available():
                    input_features = input_features.to("cuda")
                    # Convert input to match model dtype (float16 if model is in half precision)
                    model_dtype = next(self.model.parameters()).dtype
                    if input_features.dtype != model_dtype:
                        input_features = input_features.to(dtype=model_dtype)

                # Generate token ids with additional parameters for better quality
                # Use torch.no_grad() to reduce memory usage
                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        input_features,
                        max_length=448,  # Limit output length
                        num_beams=1,     # Use greedy decoding for speed
                        do_sample=False, # Deterministic output
                        early_stopping=True,
                        use_cache=False  # Disable cache to save memory
                    )
                
                # Decode token ids to text
                transcription = self.processor.batch_decode(
                    predicted_ids, 
                    skip_special_tokens=True
                )[0]
                
                # Clean up GPU memory
                if torch.cuda.is_available():
                    del input_features, predicted_ids
                    torch.cuda.empty_cache()
                
                # Filter the transcription
                filtered_transcription = self._filter_transcription(transcription)
                
                # Clear the buffer but keep any remaining audio
                self.audio_buffer = self.audio_buffer[self.buffer_size:]
                
                return filtered_transcription
                
            return None  # Not enough audio data yet
            
        except Exception as e:
            raise Exception(f"Error processing audio stream: {str(e)}")
            
    def reset_buffer(self):
        """Reset the audio buffer"""
        self.audio_buffer = [] 