from transformers import WhisperProcessor, WhisperForConditionalGeneration
from accelerate import Accelerator
import torch
from typing import Optional
from domain.value_objects.audio_config import ModelConfig


class SharedModelManager:
    """Singleton manager for sharing Whisper models across multiple engines to save GPU memory"""
    
    _instance: Optional['SharedModelManager'] = None
    _model: Optional[WhisperForConditionalGeneration] = None
    _processor: Optional[WhisperProcessor] = None
    _accelerator: Optional[Accelerator] = None
    _model_config: Optional[ModelConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SharedModelManager, cls).__new__(cls)
        return cls._instance
    
    def get_model(self, model_config: ModelConfig):
        """Get or create shared model, processor, and accelerator"""
        # Clear CUDA cache before loading to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # If model config changed or model not loaded, reload
        # Compare model_name since that's what determines which model to load
        if self._model is None or (self._model_config is None or self._model_config.model_name != model_config.model_name):
            print("🔄 Loading shared Whisper model (this may take a moment)...")
            
            # Clear any existing model from GPU
            if self._model is not None:
                del self._model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self._model_config = model_config
            self._accelerator = Accelerator()
            
            # Load processor
            self._processor = WhisperProcessor.from_pretrained(model_config.model_name)
            
            # Load model with half precision to save memory
            print(f"📦 Loading model: {model_config.model_name}")
            target_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self._model = WhisperForConditionalGeneration.from_pretrained(
                model_config.model_name,
                torch_dtype=target_dtype
            )
            
            # Move model to accelerator device
            self._model = self._model.to(self._accelerator.device)
            
            # Explicitly ensure model is in the correct dtype (sometimes .to(device) changes it)
            if torch.cuda.is_available() and target_dtype == torch.float16:
                # Convert all parameters to float16 explicitly
                self._model = self._model.half()
                print(f"✅ Model explicitly converted to float16")
            
            # Verify model dtype
            actual_dtype = next(self._model.parameters()).dtype
            print(f"✅ Shared model loaded on device: {self._accelerator.device}, dtype: {actual_dtype}")
            
            # Enable evaluation mode
            self._model.eval()
            
            # Enable mixed precision if supported and on GPU
            if torch.cuda.is_available() and self._accelerator.device.type == 'cuda':
                if self._accelerator.mixed_precision == "fp16":
                    print("Using FP16 mixed precision for faster inference")
                elif self._accelerator.mixed_precision == "bf16":
                    print("Using BF16 mixed precision for faster inference")
        else:
            print("♻️  Reusing existing shared model instance")
        
        return self._model, self._processor, self._accelerator
    
    def clear_cache(self):
        """Clear CUDA cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model_dtype(self):
        """Get the dtype of the loaded model"""
        if self._model is None:
            return torch.float32
        return next(self._model.parameters()).dtype
    
    def get_memory_info(self) -> dict:
        """Get current GPU memory usage"""
        if not torch.cuda.is_available():
            return {"available": False}
        
        return {
            "available": True,
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "reserved": torch.cuda.memory_reserved() / 1024**3,  # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,  # GB
        }

