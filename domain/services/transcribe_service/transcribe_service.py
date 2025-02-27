from transaction_scripts.transcribe.transcribe_transaction_script import TranscribeTransactionScript
from transaction_scripts.adjust_sample_rate_transaction_script import AdjustSampleRateTransactionScript
from fastapi import UploadFile
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to("cuda")

class TranscribeService:

    def __init__(self, adjustSampleRateTransactionScript:AdjustSampleRateTransactionScript, transcribeTransactionScript: TranscribeTransactionScript):
        self.transcribeTransactionScript = transcribeTransactionScript
        self.adjustSampleRateTransactionScript = adjustSampleRateTransactionScript
    
    async def transcribe(self, file: UploadFile) -> str:

        return self.transcribeTransactionScript.apply(file)
