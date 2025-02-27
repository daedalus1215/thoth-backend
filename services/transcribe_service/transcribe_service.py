from fastapi import UploadFile
from services.validators.validate_audio_type import ValidateAudioType
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to("cuda")

class TranscribeService:

    def __init__(self, validateAudioType: ValidateAudioType):
        self.validateAudioType = validateAudioType
    
    async def transcribe(file: UploadFile) -> str:
        ValidateAudioType.apply(file)
        
        audio = await file.read()
        
        inputs = processor(audio, return_tensors="pt", sampling_rate=16000).input_features.to("cuda")
        predicted_ids = model.generate(inputs)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription
