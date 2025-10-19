from domain.value_objects.audio_config import AudioConfig, ModelConfig
from domain.value_objects.transcription_engine_config import TranscriptionEngineConfig
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
        # Configuration
        self._audio_config = AudioConfig(
            sample_rate=16000,
            buffer_duration_seconds=3.0,
            chunk_overlap=0.1,
            silence_threshold=0.01,
            min_audio_length=0.5,
            confidence_threshold=0.3
        )
        
        self._model_config = ModelConfig(
            model_name="openai/whisper-large-v3",
            max_length=448,
            num_beams=1,
            do_sample=False,
            early_stopping=True
        )
        
        # Infrastructure adapters
        self._audio_processor = LibrosaAudioProcessor()
        
        # Transcription engine configuration
        self._transcription_engine_config = TranscriptionEngineConfig(
            engine_type="accelerated",
            batch_size=4,
            enable_mixed_precision=True,
            use_cache=True
        )
        
        # Choose transcription engine based on configuration
        if self._transcription_engine_config.is_accelerated():
            self._transcription_engine = AcceleratedWhisperTranscriptionEngine(self._model_config)
            print("Using Accelerated Whisper Transcription Engine")
        elif self._transcription_engine_config.is_batch():
            self._transcription_engine = BatchTranscriptionEngine(
                self._model_config, 
                batch_size=self._transcription_engine_config.batch_size
            )
            print(f"Using Batch Transcription Engine (batch_size={self._transcription_engine_config.batch_size})")
        else:
            self._transcription_engine = WhisperTranscriptionEngine(self._model_config)
            print("Using Standard Whisper Transcription Engine")
        
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
            self._transcription_engine
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
