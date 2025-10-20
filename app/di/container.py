from domain.value_objects.audio_config import AudioConfig, ModelConfig
from domain.value_objects.transcription_engine_config import TranscriptionEngineConfig
from app.config.settings import config
from domain.ports.audio_processor import AudioProcessor, TranscriptionEngine, AudioBuffer
from domain.ports.transcription_repository import TranscriptionRepository
from domain.ports.transcription_repository import NotificationService
from domain.services.transcription_domain_service import TranscriptionDomainService
from domain.services.streaming_transcription_domain_service import StreamingTranscriptionDomainService
from app.use_cases.transcribe_audio_use_case import TranscribeAudioUseCase, StreamAudioUseCase
from app.use_cases.transcribe_audio_use_case_impl import TranscribeAudioUseCaseImpl
from app.use_cases.stream_audio_use_case_impl import StreamAudioUseCaseImpl
from app.api.controllers.transcription_controller import TranscriptionController
from infra.adapters.audio.librosa_audio_processor import LibrosaAudioProcessor
from infra.adapters.transcription.whisper_transcription_engine import WhisperTranscriptionEngine
from infra.adapters.transcription.accelerated_whisper_transcription_engine import AcceleratedWhisperTranscriptionEngine
from infra.adapters.transcription.batch_transcription_engine import BatchTranscriptionEngine
from infra.adapters.transcription.chunked_whisper_transcription_engine import ChunkedWhisperTranscriptionEngine
from infra.adapters.transcription.sequential_whisper_transcription_engine import SequentialWhisperTranscriptionEngine
from infra.adapters.audio.in_memory_audio_buffer import InMemoryAudioBuffer
from infra.adapters.repositories.in_memory_transcription_repository import InMemoryTranscriptionRepository
from infra.adapters.repositories.in_memory_transcription_repository import ConsoleNotificationService


class DependencyContainer:
    """Dependency injection container following hexagonal architecture"""
    
    def __init__(self):
        self._audio_config: AudioConfig = None
        self._model_config: ModelConfig = None
        self._audio_processor: AudioProcessor = None
        self._transcription_engine: TranscriptionEngine = None
        self._audio_buffer: AudioBuffer = None
        self._transcription_repository: TranscriptionRepository = None
        self._notification_service: NotificationService = None
        self._transcription_domain_service: TranscriptionDomainService = None
        self._streaming_transcription_domain_service: StreamingTranscriptionDomainService = None
        self._transcribe_audio_use_case: TranscribeAudioUseCase = None
        self._stream_audio_use_case: StreamAudioUseCase = None
        self._transcription_controller: TranscriptionController = None
    
    def configure(self):
        """Configure all dependencies"""
        # Configuration from environment variables
        self._audio_config = AudioConfig(
            sample_rate=config.audio.sample_rate,
            buffer_duration_seconds=config.audio.buffer_duration_seconds,
            chunk_overlap=config.audio.chunk_overlap,
            silence_threshold=config.audio.silence_threshold,
            min_audio_length=config.audio.min_audio_length,
            confidence_threshold=config.audio.confidence_threshold
        )
        
        self._model_config = ModelConfig(
            model_name=config.model.model_name,
            max_length=config.model.max_length,
            num_beams=config.model.num_beams,
            do_sample=config.model.do_sample,
            early_stopping=config.model.early_stopping
        )
        
        # Infrastructure adapters
        self._audio_processor = LibrosaAudioProcessor()
        
        # Transcription engine configuration
        self._transcription_engine_config = TranscriptionEngineConfig(
            engine_type=config.transcription_engine.engine_type,
            batch_size=config.transcription_engine.batch_size,
            enable_mixed_precision=config.transcription_engine.enable_mixed_precision,
            use_cache=config.transcription_engine.use_cache,
            chunk_duration_seconds=config.transcription_engine.chunk_duration_seconds
        )
        
        # Choose transcription engine based on configuration
        # Use CPU to avoid CUDA memory issues with multiple models
        print(f"🔍 DEBUG: Engine type from config: {self._transcription_engine_config.engine_type}")
        if self._transcription_engine_config.engine_type == "sequential":
            try:
                self._transcription_engine = SequentialWhisperTranscriptionEngine(
                    self._model_config, 
                    chunk_duration_seconds=30.0
                )
                print("Using Sequential Whisper Transcription Engine (30s sliding window)")
            except Exception as e:
                print(f"Sequential engine failed to initialize: {e}")
                print("Falling back to CPU Whisper Transcription Engine")
                self._transcription_engine = WhisperTranscriptionEngine(self._model_config, device="cpu")
        elif self._transcription_engine_config.engine_type == "chunked":
            try:
                self._transcription_engine = ChunkedWhisperTranscriptionEngine(
                    self._model_config, 
                    chunk_duration_seconds=30.0
                )
                print("Using Chunked Whisper Transcription Engine (30s chunks)")
            except Exception as e:
                print(f"Chunked engine failed to initialize: {e}")
                print("Falling back to CPU Whisper Transcription Engine")
                self._transcription_engine = WhisperTranscriptionEngine(self._model_config, device="cpu")
        elif self._transcription_engine_config.is_accelerated():
            self._transcription_engine = AcceleratedWhisperTranscriptionEngine(self._model_config)
            print("Using Accelerated Whisper Transcription Engine")
        elif self._transcription_engine_config.is_batch():
            self._transcription_engine = BatchTranscriptionEngine(
                self._model_config, 
                batch_size=self._transcription_engine_config.batch_size
            )
            print(f"Using Batch Transcription Engine (batch_size={self._transcription_engine_config.batch_size})")
        else:
            self._transcription_engine = WhisperTranscriptionEngine(self._model_config, device="cpu")
            print("Using CPU Whisper Transcription Engine")
        
        # Create a separate streaming engine optimized for real-time audio
        # Use CPU for streaming to avoid CUDA memory issues with dual models
        self._streaming_transcription_engine = WhisperTranscriptionEngine(self._model_config, device="cpu")
        print("Using CPU Whisper Transcription Engine for streaming (to avoid CUDA memory issues)")
        
        self._audio_buffer = InMemoryAudioBuffer(self._audio_config)
        self._transcription_repository = InMemoryTranscriptionRepository()
        self._notification_service = ConsoleNotificationService()
        
        # Domain services
        self._transcription_domain_service = TranscriptionDomainService(
            self._audio_processor,
            self._transcription_engine
        )
        
        self._streaming_transcription_domain_service = StreamingTranscriptionDomainService(
            self._audio_buffer,
            self._streaming_transcription_engine
        )
        
        # Use cases
        self._transcribe_audio_use_case = TranscribeAudioUseCaseImpl(
            self._transcription_domain_service,
            self._transcription_repository,
            self._notification_service
        )
        
        self._stream_audio_use_case = StreamAudioUseCaseImpl(
            self._streaming_transcription_domain_service
        )
        
        # Controllers
        self._transcription_controller = TranscriptionController(
            self._transcribe_audio_use_case,
            self._stream_audio_use_case,
            self._audio_config
        )
    
    @property
    def transcription_controller(self) -> TranscriptionController:
        return self._transcription_controller
    
    @property
    def audio_config(self) -> AudioConfig:
        return self._audio_config
    
    @property
    def model_config(self) -> ModelConfig:
        return self._model_config
