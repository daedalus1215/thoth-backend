from fastapi import UploadFile, HTTPException

class ValidateAudioType:
    def apply(self, file: UploadFile) -> None:
        if file.content_type not in ["audio/wav", "audio/mpeg"]:
            raise HTTPException(status_code=400, detail="Invalid file type.")