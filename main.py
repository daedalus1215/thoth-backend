from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from domain.services.transcribe_service.transcribe_service import TranscribeService

app = FastAPI()

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    transcription = await TranscribeService.transcribe(file)
    return JSONResponse(content={"transcription": transcription})
