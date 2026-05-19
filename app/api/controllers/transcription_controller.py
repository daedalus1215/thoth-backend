import asyncio
from fastapi import APIRouter, UploadFile, File, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import JSONResponse, PlainTextResponse
from domain.entities.audio_file import AudioFile
from domain.entities.transcription import Transcription
from domain.value_objects.audio_config import AudioConfig
from app.use_cases.transcribe_audio_use_case import TranscribeAudioUseCase
from app.use_cases.transcribe_audio_use_case import StreamAudioUseCase
from typing import Optional, List

_MAX_UPLOAD_BYTES = 100 * 1024 * 1024
_AUDIO_EXTENSIONS = frozenset({
    "mp3", "wav", "m4a", "flac", "ogg", "opus", "webm", "mpeg", "mpga", "oga"
})


class TranscriptionController:
    """API controller for transcription endpoints"""
    
    def __init__(
        self,
        transcribe_audio_use_case: TranscribeAudioUseCase,
        stream_audio_use_case: StreamAudioUseCase,
        audio_config: AudioConfig
    ):
        self.transcribe_audio_use_case = transcribe_audio_use_case
        self.stream_audio_use_case = stream_audio_use_case
        self.audio_config = audio_config
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        self.router.post("/transcribe/")(self.transcribe_file)
        self.router.post("/upload")(self.upload_audio)
        self.router.post("/transcribe/batch")(self.transcribe_batch)
        self.router.post("/v1/audio/transcriptions")(self.openai_transcribe)
        self.router.websocket("/stream-audio")(self.stream_audio)
        self.router.get("/health")(self.health_check)
        self.router.get("/performance")(self.get_performance_info)
    
    @staticmethod
    def _openai_error_response(
        message: str,
        status_code: int,
        error_type: str = "server_error",
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "message": message,
                    "type": error_type,
                    "code": None,
                }
            },
        )
    
    @staticmethod
    def _calculate_timeout_seconds(file_size: Optional[int]) -> float:
        estimated_duration_minutes = (file_size or 0) / (32000 * 60)
        timeout_minutes = max(5, estimated_duration_minutes * 2)
        return min(timeout_minutes * 60, 1800)
    
    @staticmethod
    def _has_audio_extension(filename: Optional[str]) -> bool:
        if not filename or "." not in filename:
            return False
        return filename.rsplit(".", 1)[-1].lower() in _AUDIO_EXTENSIONS
    
    def _is_strict_audio_upload(self, file: UploadFile) -> bool:
        return bool(file.content_type and file.content_type.startswith("audio/"))
    
    def _is_openai_audio_upload(self, file: UploadFile) -> bool:
        if self._is_strict_audio_upload(file):
            return True
        if file.content_type in ("application/octet-stream", "video/mp4", "video/webm"):
            return self._has_audio_extension(file.filename)
        return self._has_audio_extension(file.filename)
    
    async def _run_file_transcription(self, file: UploadFile) -> Transcription:
        if file.size and file.size > _MAX_UPLOAD_BYTES:
            raise ValueError("File too large. Maximum size is 100MB.")
        
        audio_file = AudioFile.from_upload_file(file)
        timeout_seconds = self._calculate_timeout_seconds(file.size or audio_file.size)
        
        print(f"⏱️  Setting timeout to {timeout_seconds / 60:.1f} minutes for {audio_file.size} byte file")
        
        return await asyncio.wait_for(
            self.transcribe_audio_use_case.execute(audio_file, self.audio_config),
            timeout=timeout_seconds,
        )
    
    async def transcribe_file(self, file: UploadFile = File(...)):
        """Transcribe uploaded audio file with progress tracking"""
        try:
            print(f"📁 Received file upload: {file.filename} ({file.size} bytes)")
            
            if file.size and file.size > _MAX_UPLOAD_BYTES:
                return JSONResponse(
                    content={"error": "File too large. Maximum size is 100MB."},
                    status_code=413,
                )
            
            if not self._is_strict_audio_upload(file):
                return JSONResponse(
                    content={"error": "Invalid file type. Please upload an audio file."},
                    status_code=400,
                )
            
            print("✅ File validation passed. Starting transcription...")
            transcription = await self._run_file_transcription(file)
            
            print("✅ Transcription completed successfully")
            return JSONResponse(content={
                "transcription": transcription.text,
                "status": "success",
                "filename": file.filename,
            })
            
        except asyncio.TimeoutError:
            print("❌ Transcription timed out")
            return JSONResponse(
                content={"error": "Transcription timed out. Please try with a shorter audio file."},
                status_code=408,
            )
        except Exception as e:
            print(f"❌ Transcription failed: {str(e)}")
            return JSONResponse(content={"error": str(e)}, status_code=400)
    
    async def openai_transcribe(
        self,
        file: UploadFile = File(...),
        model: Optional[str] = Form(default=None),
        response_format: Optional[str] = Form(default="json"),
        language: Optional[str] = Form(default=None),
        temperature: Optional[float] = Form(default=None),
    ):
        """
        OpenAI-compatible audio transcription endpoint for clients such as Hermes Agent.
        """
        del model, language, temperature  # accepted for API compatibility; engine config is internal
        
        try:
            print(f"📁 OpenAI transcribe: {file.filename} ({file.size} bytes), format={response_format}")
            
            if file.size and file.size > _MAX_UPLOAD_BYTES:
                return self._openai_error_response(
                    "File too large. Maximum size is 100MB.",
                    status_code=413,
                    error_type="invalid_request_error",
                )
            
            if not self._is_openai_audio_upload(file):
                return self._openai_error_response(
                    "Invalid file type. Please upload an audio file.",
                    status_code=400,
                    error_type="invalid_request_error",
                )
            
            normalized_format = (response_format or "json").strip().lower()
            if normalized_format not in ("json", "text"):
                return self._openai_error_response(
                    f"Unsupported response_format: {response_format}. Supported: json, text.",
                    status_code=400,
                    error_type="invalid_request_error",
                )
            
            transcription = await self._run_file_transcription(file)
            text = transcription.text
            
            if normalized_format == "text":
                return PlainTextResponse(content=text, media_type="text/plain")
            
            return JSONResponse(content={"text": text})
            
        except asyncio.TimeoutError:
            print("❌ OpenAI transcribe timed out")
            return self._openai_error_response(
                "Transcription timed out. Please try with a shorter audio file.",
                status_code=408,
            )
        except Exception as e:
            print(f"❌ OpenAI transcribe failed: {str(e)}")
            return self._openai_error_response(
                str(e),
                status_code=500 if "not valid" in str(e).lower() else 400,
            )
    
    async def upload_audio(self, file: UploadFile = File(...)):
        """Upload and transcribe audio file"""
        try:
            audio_file = AudioFile.from_upload_file(file)
            transcription = await self.transcribe_audio_use_case.execute(audio_file, self.audio_config)
            return {"transcription": transcription.text}
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=400)
    
    async def stream_audio(self, websocket: WebSocket):
        """WebSocket endpoint for streaming audio transcription"""
        await websocket.accept()
        
        try:
            self.stream_audio_use_case.reset_stream()
            
            while True:
                audio_chunk = await websocket.receive_bytes()
                transcription = await self.stream_audio_use_case.execute(audio_chunk, self.audio_config)
                
                if transcription:
                    await websocket.send_json({"transcription": transcription.text})
                    
        except WebSocketDisconnect:
            print("WebSocket disconnected")
        except Exception as e:
            print(f"Error in WebSocket: {str(e)}")
            await websocket.close()
        finally:
            self.stream_audio_use_case.reset_stream()
    
    async def transcribe_batch(self, files: List[UploadFile] = File(...)):
        """Transcribe multiple audio files in batch for efficiency"""
        try:
            audio_files = [AudioFile.from_upload_file(file) for file in files]
            
            transcriptions = []
            for audio_file in audio_files:
                transcription = await self.transcribe_audio_use_case.execute(audio_file, self.audio_config)
                transcriptions.append({
                    "filename": audio_file.filename,
                    "transcription": transcription.text,
                })
            
            return {"transcriptions": transcriptions, "count": len(transcriptions)}
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=400)
    
    async def get_performance_info(self):
        """Get performance information about the transcription engine"""
        try:
            if hasattr(self.transcribe_audio_use_case.transcription_domain_service.transcription_engine, 'get_device_info'):
                device_info = self.transcribe_audio_use_case.transcription_domain_service.transcription_engine.get_device_info()
            elif hasattr(self.transcribe_audio_use_case.transcription_domain_service.transcription_engine, 'get_performance_stats'):
                device_info = self.transcribe_audio_use_case.transcription_domain_service.transcription_engine.get_performance_stats()
            else:
                device_info = {"engine": "standard"}
            
            return {
                "status": "healthy",
                "performance": device_info,
                "audio_config": {
                    "sample_rate": self.audio_config.sample_rate,
                    "buffer_duration": self.audio_config.buffer_duration_seconds,
                },
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def health_check(self):
        """Health check endpoint"""
        return {"status": "healthy"}
