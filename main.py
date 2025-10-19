from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.di.container import DependencyContainer

# Create FastAPI application
app = FastAPI(title="Thoth Transcription Service", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React default port
        "http://localhost:8080",  # Vue default port
        "http://localhost:9000",  # Quasar default port
        "http://localhost:9001",  # Quasar alternative port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080", 
        "http://127.0.0.1:9000",
        "http://127.0.0.1:9001",
        # For development, you can also use:
        # "*"  # Allow all origins (NOT recommended for production)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize dependency container
container = DependencyContainer()
container.configure()

# Include the transcription controller routes
app.include_router(container.transcription_controller.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
