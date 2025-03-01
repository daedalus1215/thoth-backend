import torch
import torchaudio
from transformers import WhisperProcessor
from fastapi import UploadFile

class AdjustSampleRateTransactionScript: 
    _target_sampling_rate = 16000

    async def apply(self, file: UploadFile) -> bytes:
        # Load audio asynchronously
        audio, original_sampling_rate = await torchaudio.load(file)

        # Resample audio if needed
        audio = self._resample_to_target_rate(audio, original_sampling_rate)

        # Ensure mono channel if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Process audio with Whisper processor
        processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        inputs = processor(audio.squeeze().numpy(), return_tensors="pt", sampling_rate=self._target_sampling_rate).input_features.to("cuda")

        return inputs

    def _resample_to_target_rate(self, audio: torch.Tensor, original_sampling_rate: int) -> torch.Tensor:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sampling_rate, new_freq=self._target_sampling_rate)
        return resampler(audio)
