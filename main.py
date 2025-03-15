from domain.services.transcribe_service.transaction_scripts.transcribe.validators.validate_audio_type import ValidateAudioType
from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.responses import JSONResponse
from domain.services.transcribe_service.transcribe_service import TranscribeService
from domain.services.transcribe_service.transaction_scripts.adjust_sample_rate_transaction_script import AdjustSampleRateTransactionScript
from domain.services.transcribe_service.transaction_scripts.transcribe.transcribe_transaction_script import TranscribeTransactionScript
from domain.services.streaming_transcribe_service.streaming_transcribe_service import StreamingTranscribeService
from domain.services.streaming_transcribe_service.config import StreamingConfig
import asyncio
import io
import numpy as np
from typing import List

app = FastAPI()

# Instantiate the dependencies
adjust_sample_rate_script = AdjustSampleRateTransactionScript()

validate_audio = ValidateAudioType()
transcribe_script = TranscribeTransactionScript(validate_audio)

# Create an instance of TranscribeService with the dependencies
transcribe_service = TranscribeService(adjust_sample_rate_script, transcribe_script)

config = StreamingConfig(
    sample_rate=16000,
    buffer_duration_seconds=2.0,  # Buffer 2 seconds of audio
    model_name="openai/whisper-base",
    chunk_overlap=0.1
)
streaming_transcribe_service = StreamingTranscribeService(config)

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    transcription = await transcribe_service.transcribe(file)
    return JSONResponse(content={"transcription": transcription})

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    Endpoint to handle audio file uploads and transcribe them.
    Accepts audio files and returns the transcription as text.
    """
    try:
        transcription = await transcribe_service.transcribe(file)
        return {"transcription": transcription}
    except Exception as e:
        return {"error": str(e)}, 400

@app.websocket("/stream-audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # Reset the buffer for new connection
        streaming_transcribe_service.reset_buffer()
        
        while True:
            # Receive audio chunks from the WebSocket
            audio_chunk = await websocket.receive_bytes()
            
            # Process the chunk
            transcription = await streaming_transcribe_service.process_chunk(audio_chunk)
            
            # If we have a transcription, send it back
            if transcription:
                await websocket.send_json({"transcription": transcription})
                
    except Exception as e:
        print(f"Error in WebSocket: {str(e)}")
        await websocket.close()
    finally:
        streaming_transcribe_service.reset_buffer()

# Optional: Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
