from fastapi import FastAPI
from app.di.container import DependencyContainer

# Create FastAPI application
app = FastAPI(title="Thoth Transcription Service", version="1.0.0")

# Initialize dependency container
container = DependencyContainer()
container.configure()

# Include the transcription controller routes
app.include_router(container.transcription_controller.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
