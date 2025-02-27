from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pydantic import BaseModel

app = FastAPI()

# Load Whisper Model and Processor
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to("cuda")

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    if file.content_type not in ["audio/wav", "audio/mpeg"]:
        raise HTTPException(status_code=400, detail="Invalid file type.")

    audio = await file.read()
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000).input_features.to("cuda")
    predicted_ids = model.generate(inputs)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return JSONResponse(content={"transcription": transcription})