from fastapi import APIRouter, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from domain.entities.audio_file import AudioFile
from domain.value_objects.audio_config import AudioConfig
from app.use_cases.transcribe_audio_use_case import TranscribeAudioUseCase
from app.use_cases.transcribe_audio_use_case import StreamAudioUseCase
from typing import Optional, List


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
        self.router.websocket("/stream-audio")(self.stream_audio)
        self.router.get("/health")(self.health_check)
        self.router.get("/performance")(self.get_performance_info)
    
    async def transcribe_file(self, file: UploadFile = File(...)):
        """Transcribe uploaded audio file with progress tracking"""
        try:
            print(f"📁 Received file upload: {file.filename} ({file.size} bytes)")
            
            # Validate file size (industry standard: max 100MB)
            if file.size and file.size > 100 * 1024 * 1024:
                return JSONResponse(
                    content={"error": "File too large. Maximum size is 100MB."}, 
                    status_code=413
                )
            
            # Validate file type
            if not file.content_type or not file.content_type.startswith('audio/'):
                return JSONResponse(
                    content={"error": "Invalid file type. Please upload an audio file."}, 
                    status_code=400
                )
            
            print(f"✅ File validation passed. Starting transcription...")
            audio_file = AudioFile.from_upload_file(file)
            
            # Add timeout to prevent infinite hanging
            import asyncio
            transcription = await asyncio.wait_for(
                self.transcribe_audio_use_case.execute(audio_file, self.audio_config),
                timeout=300.0  # 5 minute timeout
            )
            
            print(f"✅ Transcription completed successfully")
            return JSONResponse(content={
                "transcription": transcription.text,
                "status": "success",
                "filename": file.filename
            })
            
        except asyncio.TimeoutError:
            print(f"❌ Transcription timed out after 5 minutes")
            return JSONResponse(
                content={"error": "Transcription timed out. Please try with a shorter audio file."}, 
                status_code=408
            )
        except Exception as e:
            print(f"❌ Transcription failed: {str(e)}")
            return JSONResponse(content={"error": str(e)}, status_code=400)
    
    async def upload_audio(self, file: UploadFile = File(...)):
        """Upload and transcribe audio file"""
        try:
            audio_file = AudioFile.from_upload_file(file)
            transcription = await self.transcribe_audio_use_case.execute(audio_file, self.audio_config)
            return {"transcription": transcription.text}
        except Exception as e:
            return {"error": str(e)}, 400
    
    async def stream_audio(self, websocket: WebSocket):
        """WebSocket endpoint for streaming audio transcription"""
        await websocket.accept()
        
        try:
            # Reset the stream state for new connection
            self.stream_audio_use_case.reset_stream()
            
            while True:
                # Receive audio chunks from the WebSocket
                audio_chunk = await websocket.receive_bytes()
                
                # Process the chunk
                transcription = await self.stream_audio_use_case.execute(audio_chunk, self.audio_config)
                
                # If we have a transcription, send it back
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
            
            # Process each file individually (could be optimized for true batch processing)
            transcriptions = []
            for audio_file in audio_files:
                transcription = await self.transcribe_audio_use_case.execute(audio_file, self.audio_config)
                transcriptions.append({
                    "filename": audio_file.filename,
                    "transcription": transcription.text
                })
            
            return {"transcriptions": transcriptions, "count": len(transcriptions)}
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=400)
    
    async def get_performance_info(self):
        """Get performance information about the transcription engine"""
        try:
            # Get performance stats if available
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
                    "buffer_duration": self.audio_config.buffer_duration_seconds
                }
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def health_check(self):
        """Health check endpoint"""
        return {"status": "healthy"}
