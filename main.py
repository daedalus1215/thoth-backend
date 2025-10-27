from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

from app.di.container import DependencyContainer
from app.config.settings import config

# Create FastAPI application
app = FastAPI(
    title="Thoth Transcription Service", 
    version="1.0.0",
    debug=config.server.debug
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors.origins,
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
    import uvicorn.config
    
    # Prepare SSL parameters if certificates are provided
    ssl_kwargs = {}
    if config.server.ssl_keyfile and config.server.ssl_certfile:
        ssl_kwargs['ssl_keyfile'] = config.server.ssl_keyfile
        ssl_kwargs['ssl_certfile'] = config.server.ssl_certfile
        print(f"🔒 SSL enabled: key={config.server.ssl_keyfile}, cert={config.server.ssl_certfile}")
    
    uvicorn.run(
        app, 
        host=config.server.host, 
        port=config.server.port,
        reload=config.server.debug,
        **ssl_kwargs
    )
