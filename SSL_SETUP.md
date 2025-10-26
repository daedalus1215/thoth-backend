# SSL Configuration Guide

This guide explains how to configure SSL/HTTPS for the Thoth backend to enable microphone access when hosting off the current box.

## Why SSL is Required

Browsers require HTTPS (SSL) for microphone access when accessing the application from a remote host (not localhost). This is a security requirement enforced by modern browsers.

## Quick Setup

### Step 1: Create SSL Directory (if using Docker)

```bash
mkdir -p ssl
```

### Step 2: Copy Your SSL Files

Place your SSL certificate files in the backend directory:

**For Docker deployment:**
```bash
cp server.key ssl/server.key
cp server.crt ssl/server.crt
```

**For local deployment (without Docker):**
```bash
# Place files directly in the backend directory
cp server.key .
cp server.crt .
```

### Step 3: Configure Environment Variables

#### Option A: Using .env file (Recommended)

1. Copy the example environment file:
```bash
cp env.example .env
```

2. Edit `.env` and uncomment/modify the SSL lines:
```bash
nano .env
```

3. For **local deployment** (without Docker):
```bash
# SSL Configuration
SSL_KEYFILE=server.key
SSL_CERTFILE=server.crt
```

4. For **Docker deployment**:
```bash
# SSL Configuration
SSL_KEYFILE=/app/ssl/server.key
SSL_CERTFILE=/app/ssl/server.crt
```

#### Option B: Command Line Environment Variables

```bash
# For local deployment
export SSL_KEYFILE=server.key
export SSL_CERTFILE=server.crt

# For Docker deployment
export SSL_KEYFILE=/app/ssl/server.key
export SSL_CERTFILE=/app/ssl/server.crt
```

### Step 4: Run the Application

#### With Docker:
```bash
docker-compose up --build
```

#### Without Docker:
```bash
# Set environment variables
export SSL_KEYFILE=server.key
export SSL_CERTFILE=server.crt

# Run the application
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Verification

When the application starts with SSL enabled, you should see:
```
🔒 SSL enabled: key=server.key, cert=server.crt
```

The server will now accept HTTPS connections on port 8000.

## Accessing the API

### HTTP Endpoints (when SSL is enabled)
- API: `https://your-domain:8000/api/...`
- Health Check: `https://your-domain:8000/health`

### WebSocket Endpoints (when SSL is enabled)
- Stream Audio: `wss://your-domain:8000/stream-audio`

## Frontend Configuration

Update your frontend to use HTTPS/WSS when connecting to a remote backend:

```bash
# In thoth-frontend/.env or .env.production
VITE_API_BASE_URL=https://your-domain:8000
VITE_WS_BASE_URL=wss://your-domain:8000
```

## Troubleshooting

### SSL Certificate Errors

If you see SSL errors in the browser:
1. Make sure you're accessing via HTTPS (not HTTP)
2. If using self-signed certificates, you'll need to accept the security warning in your browser
3. For production, use proper CA-signed certificates

### Microphone Still Not Working

1. Ensure the frontend is also using HTTPS
2. Check browser console for SSL/TLS errors
3. Verify the certificate is valid for your domain

### Permission Denied Errors

If you see permission errors:
```bash
# Make sure SSL files are readable
chmod 644 ssl/server.crt
chmod 600 ssl/server.key  # Private key should be more restrictive
```

## Security Notes

⚠️ **Important Security Considerations:**

1. **Never commit SSL keys to Git** - Add to `.gitignore`:
   ```
   *.key
   *.crt
   *.pem
   ssl/
   ```

2. **Use strong passphrase** for private keys in production

3. **Rotate certificates** regularly

4. **Use proper CA-signed certificates** for production deployments

5. **Restrict permissions** on private keys:
   ```bash
   chmod 600 server.key
   ```

## Disabling SSL

To disable SSL, simply remove or comment out the SSL environment variables:

```bash
# Comment out in .env file
# SSL_KEYFILE=server.key
# SSL_CERTFILE=server.crt
```

The server will fall back to HTTP mode.

## Self-Signed Certificate Example

If you don't have SSL certificates yet, you can create self-signed certificates for testing:

```bash
# Create a self-signed certificate (for testing only!)
openssl req -x509 -newkey rsa:4096 -keyout server.key -out server.crt -days 365 -nodes

# Move to ssl directory for Docker
mkdir -p ssl
mv server.key ssl/
mv server.crt ssl/
```

**Note:** Self-signed certificates will show security warnings in browsers. For production, use properly signed certificates from a Certificate Authority (CA).

