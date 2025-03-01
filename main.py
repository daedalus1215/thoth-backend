from domain.services.transcribe_service.transaction_scripts.transcribe.validators.validate_audio_type import ValidateAudioType
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from domain.services.transcribe_service.transcribe_service import TranscribeService
from domain.services.transcribe_service.transaction_scripts.adjust_sample_rate_transaction_script import AdjustSampleRateTransactionScript
from domain.services.transcribe_service.transaction_scripts.transcribe.transcribe_transaction_script import TranscribeTransactionScript

app = FastAPI()

# Instantiate the dependencies
adjust_sample_rate_script = AdjustSampleRateTransactionScript()

validate_audio = ValidateAudioType()
transcribe_script = TranscribeTransactionScript(validate_audio)

# Create an instance of TranscribeService with the dependencies
transcribe_service = TranscribeService(adjust_sample_rate_script, transcribe_script)

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    transcription = await transcribe_service.transcribe(file)
    return JSONResponse(content={"transcription": transcription})
