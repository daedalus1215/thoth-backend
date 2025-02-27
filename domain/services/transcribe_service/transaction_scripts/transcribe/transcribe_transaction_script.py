from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
from .validators.validate_audio_type import ValidateAudioType
from fastapi import UploadFile

# @TODO: Needs to be moved to infra in an adapter
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to("cuda")

sixteen_khz = 16000
operating_hardware = 'cuda'

class TranscribeTransactionScript:
    def __init__(self, validateAudioType: ValidateAudioType):
        self.validateAudioType = validateAudioType
    

    async def apply(file: UploadFile) -> str:
        ValidateAudioType.apply(file)
    
        audio = await file.read()
        
        inputs = processor(audio, return_tensors="pt", sampling_rate=sixteen_khz).input_features.to(operating_hardware)
        predicted_ids = model.generate(inputs)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription