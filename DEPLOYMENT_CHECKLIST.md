# Deployment Checklist for SSL/HTTPS

When deploying to a remote host with SSL enabled, follow this checklist:

## ✅ Checklist

### Backend Setup
- [ ] Copy SSL certificate files to backend root:
  - [ ] `server.key`
  - [ ] `server.crt`
- [ ] Create or update `.env` file with SSL paths:
  ```bash
  SSL_KEYFILE=server.key
  SSL_CERTFILE=server.crt
  ```
- [ ] Restart backend service

### Frontend Setup
- [ ] Update frontend `.env.production` file with HTTPS/WSS URLs:
  ```bash
  VITE_API_BASE_URL=https://your-domain:8000
  VITE_WS_BASE_URL=wss://your-domain:8000
  ```
- [ ] Rebuild frontend (if needed):
  ```bash
  npm run build
  ```

### Verification
- [ ] Test HTTPS connection: `https://your-domain:8000/health`
- [ ] Test WebSocket connection from browser console
- [ ] Test microphone access (requires HTTPS)

## Quick Commands

### Backend
```bash
# If using systemd service
sudo systemctl restart thoth-backend

# If using Docker
docker-compose restart

# If running manually
# Kill existing process and restart
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend
```bash
# Rebuild with new env vars
npm run build

# Or for development
npm run dev  # Uses .env.local
```

## Troubleshooting

### SSL Certificate Errors
- Verify certificate files are in correct location
- Check file permissions: `chmod 600 server.key`
- Verify certificate is valid for your domain

### Microphone Still Not Working
- Ensure frontend is also served over HTTPS
- Check browser console for security errors
- Verify WebSocket is using `wss://` not `ws://`

### Port Access Issues
- Ensure port 8000 is open in firewall
- Check if port needs to be exposed in Docker/docker-compose
- Verify CORS settings allow your frontend domain

## Environment Variables Summary

**Backend `.env`:**
```bash
HOST=0.0.0.0
PORT=8000
SSL_KEYFILE=server.key
SSL_CERTFILE=server.crt
```

**Frontend `.env.production`:**
```bash
VITE_API_BASE_URL=https://your-domain:8000
VITE_WS_BASE_URL=wss://your-domain:8000
```

