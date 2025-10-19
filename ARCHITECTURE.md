# Thoth Backend - Hexagonal Architecture

This project has been migrated to follow Domain-Driven Design (DDD) and Hexagonal Architecture principles.

## Architecture Overview

The codebase is organized into three main layers:

### 1. Domain Layer (`domain/`)
Contains the core business logic and is independent of external frameworks.

- **Entities** (`domain/entities/`): Core business objects
  - `AudioFile`: Represents an audio file with metadata
  - `Transcription`: Represents a transcription result

- **Value Objects** (`domain/value_objects/`): Immutable objects that represent concepts
  - `AudioConfig`: Audio processing configuration
  - `ModelConfig`: AI model configuration

- **Ports** (`domain/ports/`): Interfaces that define contracts for external dependencies
  - `AudioProcessor`: Audio processing operations
  - `TranscriptionEngine`: Transcription operations
  - `AudioBuffer`: Audio buffering operations
  - `TranscriptionRepository`: Persistence operations
  - `NotificationService`: Notification operations

- **Services** (`domain/services/`): Domain services that contain business logic
  - `TranscriptionDomainService`: Core transcription business logic
  - `StreamingTranscriptionDomainService`: Streaming transcription business logic

### 2. Application Layer (`app/`)
Contains use cases and application services that orchestrate domain operations.

- **Use Cases** (`app/use_cases/`): Application use cases
  - `TranscribeAudioUseCase`: Transcribe uploaded audio files
  - `StreamAudioUseCase`: Handle streaming audio transcription

- **Controllers** (`app/api/controllers/`): API controllers
  - `TranscriptionController`: HTTP/WebSocket endpoints

- **Dependency Injection** (`app/di/`): Dependency injection container
  - `DependencyContainer`: Configures and wires all dependencies

### 3. Infrastructure Layer (`infra/`)
Contains adapters that implement the ports defined in the domain layer.

- **Audio Adapters** (`infra/adapters/audio/`):
  - `LibrosaAudioProcessor`: Audio processing using Librosa
  - `InMemoryAudioBuffer`: In-memory audio buffering

- **Transcription Adapters** (`infra/adapters/transcription/`):
  - `WhisperTranscriptionEngine`: Transcription using Whisper

- **Repository Adapters** (`infra/adapters/repositories/`):
  - `InMemoryTranscriptionRepository`: In-memory persistence
  - `ConsoleNotificationService`: Console-based notifications

## Key Benefits

1. **Separation of Concerns**: Each layer has a clear responsibility
2. **Testability**: Domain logic can be tested independently
3. **Flexibility**: Easy to swap implementations (e.g., different AI models)
4. **Maintainability**: Changes in one layer don't affect others
5. **Business Logic Protection**: Core business rules are isolated from external concerns

## Dependency Flow

```
API Layer (Controllers) 
    ↓
Application Layer (Use Cases)
    ↓
Domain Layer (Services, Entities, Ports)
    ↑
Infrastructure Layer (Adapters implementing Ports)
```

## Running the Application

The application entry point is `main.py`, which:
1. Creates a FastAPI application
2. Configures the dependency injection container
3. Includes the transcription controller routes

```bash
python main.py
```

## Adding New Features

To add new features following this architecture:

1. **Define Domain Entities/Value Objects** in `domain/`
2. **Create Ports** for external dependencies in `domain/ports/`
3. **Implement Domain Services** with business logic in `domain/services/`
4. **Create Use Cases** in `app/use_cases/`
5. **Implement Infrastructure Adapters** in `infra/adapters/`
6. **Add Controllers** in `app/api/controllers/`
7. **Update Dependency Container** in `app/di/container.py`
