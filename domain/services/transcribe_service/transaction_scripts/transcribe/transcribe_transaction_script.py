from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import io
import torchaudio
import torch
from domain.services.transcribe_service.transaction_scripts.transcribe.validators.validate_audio_type import ValidateAudioType
from fastapi import UploadFile


sixteen_khz = 16000

# @TODO: Needs to be moved to infra in an adapter
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(device)

class TranscribeTransactionScript:
    def __init__(self, validateAudioType: ValidateAudioType):
        self.validateAudioType = validateAudioType
    
    async def apply(self, file: UploadFile) -> str:
        self.validateAudioType.apply(file)

        # Read audio file as bytes
        audio_bytes = await file.read()

        # Decode bytes using torchaudio
        audio_tensor, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        
        # Resample if needed
        if sample_rate != sixteen_khz:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sixteen_khz)
            audio_tensor = resampler(audio_tensor)
        
        # Convert to mono if necessary
        if audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0)
        
        # Convert to numpy and process with Wav2Vec2Processor
        input_values = processor(audio_tensor.numpy(), sampling_rate=sixteen_khz, return_tensors="pt").input_values.to(device)
        
        # Perform inference
        with torch.no_grad():
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)

        # Decode the predicted ids to text
        transcription = processor.batch_decode(predicted_ids)[0]

        return transcription

