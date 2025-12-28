import re
from typing import Optional


class TranscriptionPostProcessor:
    """
    Domain service for post-processing transcription text to fix common formatting issues.
    
    Handles:
    - Multiple spaces
    - Floating periods (periods not attached to words)
    - Duplicate punctuation
    - Inconsistent spacing around punctuation
    """
    
    @staticmethod
    def post_process(text: str) -> str:
        """
        Post-process transcription text to fix common formatting issues.
        
        This method is idempotent - safe to run multiple times.
        
        Args:
            text: Raw transcription text from Whisper model
            
        Returns:
            Cleaned and normalized transcription text
        """
        if not text:
            return text
        
        # Step 1: Remove duplicate punctuation marks
        text = re.sub(r'\.{2,}', '.', text)  # "word.." -> "word."
        text = re.sub(r'[!?]{2,}', lambda m: m.group()[0], text)  # Keep first of multiple ! or ?
        text = re.sub(r',{2,}', ',', text)  # Remove duplicate commas
        
        # Step 2: Fix floating periods - periods surrounded by spaces that shouldn't be there
        # These are artifacts where a period appears between words incorrectly
        # Pattern: word . word (period with spaces on both sides, not sentence end)
        # We need to be careful not to remove legitimate sentence endings
        # Remove floating periods before lowercase letters
        text = re.sub(r'\s+\.\s+([a-z])', r' \1', text)
        # Remove floating periods before uppercase letters (but preserve if it's a sentence start)
        # Only remove if there's a word before it (not at start of text)
        text = re.sub(r'([a-zA-Z0-9])\s+\.\s+([A-Z])', r'\1 \2', text)
        # Remove floating periods before numbers
        text = re.sub(r'\s+\.\s+(\d)', r' \1', text)
        
        # Step 3: Normalize spacing around punctuation
        # Remove spaces before punctuation (except opening quotes/parentheses)
        text = re.sub(r'\s+([.!?,])', r'\1', text)
        # Ensure single space after periods, exclamation marks, question marks
        text = re.sub(r'([.!?])\s+', r'\1 ', text)
        # Ensure single space after commas
        text = re.sub(r',\s+', ', ', text)
        
        # Step 4: Collapse multiple spaces to single space
        text = re.sub(r'\s+', ' ', text)
        
        # Step 5: Fix spacing at sentence boundaries (add space if missing)
        # Ensure proper spacing after sentence-ending punctuation before capital letters
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        # Step 6: Clean up leading/trailing whitespace
        text = text.strip()
        
        return text

