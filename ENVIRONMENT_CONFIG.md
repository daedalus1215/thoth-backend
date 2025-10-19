# Environment Configuration Guide

This guide explains how to configure the Thoth application using environment variables for different deployment environments.

## 🚀 **Quick Setup**

### **Frontend Setup**
```bash
# Copy the example environment file
cp env.example .env.local

# Edit the configuration
nano .env.local
```

### **Backend Setup**
```bash
# Copy the example environment file
cp env.example .env

# Edit the configuration
nano .env
```

## 📁 **Environment Files**

### **Frontend Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_BASE_URL` | `http://localhost:8000` | Backend API base URL |
| `VITE_WS_BASE_URL` | `ws://localhost:8000` | WebSocket base URL |
| `VITE_AUDIO_SAMPLE_RATE` | `16000` | Audio sample rate in Hz |
| `VITE_AUDIO_BUFFER_SIZE` | `4096` | Audio buffer size |
| `VITE_AUDIO_CHANNELS` | `1` | Number of audio channels |
| `VITE_DEV_MODE` | `true` | Development mode flag |
| `VITE_DEBUG_LOGGING` | `true` | Debug logging flag |

### **Backend Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `DEBUG` | `true` | Debug mode |
| `CORS_ORIGINS` | `localhost:3000,8080,9000,9001` | Allowed CORS origins |
| `WHISPER_MODEL_NAME` | `openai/whisper-large-v3` | Whisper model name |
| `WHISPER_MAX_LENGTH` | `448` | Maximum sequence length |
| `WHISPER_NUM_BEAMS` | `1` | Number of beams for generation |
| `AUDIO_SAMPLE_RATE` | `16000` | Audio sample rate |
| `AUDIO_BUFFER_DURATION_SECONDS` | `3.0` | Audio buffer duration |
| `TRANSCRIPTION_ENGINE_TYPE` | `chunked` | Transcription engine type |
| `CUDA_ENABLED` | `true` | CUDA acceleration |

## 🌍 **Environment Examples**

### **Development Environment**
```bash
# Frontend (.env.local)
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
VITE_DEV_MODE=true
VITE_DEBUG_LOGGING=true

# Backend (.env)
HOST=0.0.0.0
PORT=8000
DEBUG=true
CORS_ORIGINS=http://localhost:3000,http://localhost:8080,http://localhost:9000,http://localhost:9001
```

### **Production Environment**
```bash
# Frontend (.env.production)
VITE_API_BASE_URL=https://api.yourdomain.com
VITE_WS_BASE_URL=wss://api.yourdomain.com
VITE_DEV_MODE=false
VITE_DEBUG_LOGGING=false

# Backend (.env.production)
HOST=0.0.0.0
PORT=8000
DEBUG=false
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
CUDA_ENABLED=true
```

### **Docker Environment**
```bash
# Backend (.env.docker)
HOST=0.0.0.0
PORT=8000
DEBUG=false
CORS_ORIGINS=http://localhost:3000,http://localhost:8080,http://localhost:9000
CUDA_ENABLED=true
```

## 🔧 **Configuration Management**

### **Frontend Configuration**
The frontend uses Vite's environment variable system:
- Variables must be prefixed with `VITE_`
- Access via `import.meta.env.VITE_VARIABLE_NAME`
- Fallback to defaults if not set

### **Backend Configuration**
The backend uses a centralized configuration system:
- Loads from `.env` file automatically
- Falls back to defaults if not set
- Type-safe configuration classes

## 🚀 **Deployment**

### **Local Development**
```bash
# Frontend
npm run dev

# Backend
uvicorn main:app --reload
```

### **Production**
```bash
# Frontend
npm run build

# Backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

### **Docker**
```bash
# Build and run with Docker Compose
docker-compose up --build
```

## 🔒 **Security Notes**

- **Never commit `.env` files** to version control
- **Use different configurations** for different environments
- **Validate environment variables** before deployment
- **Use HTTPS/WSS** in production
- **Restrict CORS origins** in production

## 🐛 **Troubleshooting**

### **Common Issues**

1. **CORS Errors**: Check `CORS_ORIGINS` configuration
2. **WebSocket Connection Failed**: Verify `VITE_WS_BASE_URL`
3. **API Connection Failed**: Check `VITE_API_BASE_URL`
4. **Audio Issues**: Verify audio configuration variables

### **Debug Mode**
Enable debug logging to troubleshoot issues:
```bash
# Frontend
VITE_DEBUG_LOGGING=true

# Backend
DEBUG=true
```

## 📚 **Additional Resources**

- [Vite Environment Variables](https://vitejs.dev/guide/env-and-mode.html)
- [Python-dotenv Documentation](https://python-dotenv.readthedocs.io/)
- [FastAPI Configuration](https://fastapi.tiangolo.com/advanced/settings/)
