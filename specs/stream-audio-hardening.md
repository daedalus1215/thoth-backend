# Spec: Harden `/stream-audio` — make thoth reachable only by the Chronus backend

**Status:** Ready to implement
**Repos:** `thoth-backend` (this spec), `chronus-react-nestjs` (`specs/transcription-gateway.md`)
**Sequencing:** land this **before** the Chronus frontend is repointed, so there is never a window
where both the direct-browser path and the proxied path are open.

---

## 1. Problem

`app/api/controllers/transcription_controller.py::stream_audio` accepts every WebSocket that
reaches it:

```python
async def stream_audio(self, websocket: WebSocket):
    await websocket.accept()          # no auth, no origin check, no connection limit
```

Today a browser connects to it directly from `chronus-react-nestjs`. After the Chronus gateway
ships, thoth's only client should be the Chronus backend — but nothing in thoth enforces that.

**`CORS_ORIGINS` does not help.** Starlette's `CORSMiddleware`, configured in `main.py`, does not
run on WebSocket handshakes. The `CORS_ORIGINS` setting has never protected `/stream-audio` and
never will. Do not treat it as a control.

There is also a correctness bug that has been masked by only ever having one user (§4).

## 2. Scope

| In | Out |
|---|---|
| Reject browser-originated handshakes on `/stream-audio` (§3) | Per-connection streaming state / real concurrency (§5 — specced, deferred) |
| Guard against a second concurrent stream corrupting the first (§4) | Any change to the HTTP upload endpoints |
| Network binding and firewall posture (§7) | Whisper model, engine, or accuracy tuning |
| A disabled-by-default shared-secret hook (§6) | Authentication for `/transcribe/`, `/upload`, `/v1/audio/transcriptions` |

## 3. Reject browser handshakes via the `Origin` header

The cheapest control that precisely kills the vector we care about.

**Why it works:** every browser sends an `Origin` header on a WebSocket handshake — it is mandatory
in the WHATWG spec and not suppressible from JavaScript. Node's `ws` client (what the Chronus
gateway uses) sends `Origin` **only** if the caller passes an `origin` option, which we will not.
So: *reject any `/stream-audio` handshake carrying an `Origin` header* blocks every browser while
letting the Chronus backend through. No key to distribute, store, or rotate.

**Implementation** — in `TranscriptionController.stream_audio`, before `accept()`:

```python
async def stream_audio(self, websocket: WebSocket):
    origin = websocket.headers.get("origin")
    if origin is not None and origin not in config.stream.allowed_origins:
        # Browsers always send Origin; server-side clients (Node `ws`, Python `websockets`)
        # do not. A present Origin here means a browser is connecting directly, which is
        # exactly what this service must not serve. See specs/stream-audio-hardening.md §3.
        await websocket.close(code=4403, reason="Direct client connections are not permitted")
        return
    await websocket.accept()
    ...
```

New setting in `app/config/settings.py`, exposed as `config.stream.allowed_origins`:

```bash
# Comma-separated Origin allowlist for the /stream-audio WebSocket.
# EMPTY (the default) means: reject every handshake that carries an Origin header,
# i.e. every browser. Server-to-server clients send no Origin and are unaffected.
# Populate ONLY for local debugging with a browser tool; never in production.
STREAM_ALLOWED_ORIGINS=
```

Default empty. Add it to `env.example` with that comment intact.

Log every rejection at WARNING with the origin and the peer address — this is your tripwire for a
misconfigured client or a probe.

> This is a defence-in-depth layer, not the primary control. The firewall (§7) is the primary
> control. The value here is that it still holds when the firewall is wrong, which firewalls
> periodically are.

## 4. Fail loudly on a second concurrent stream

### The bug

`app/di/container.py` builds **one** `InMemoryAudioBuffer` and **one**
`ChunkedWhisperTranscriptionEngine` at boot, wires them through a single
`StreamingTranscriptionDomainService` → `StreamAudioUseCaseImpl` → `TranscriptionController`. That
state is shared by every WebSocket connection:

- `stream_audio` calls `reset_stream()` on connect **and** in `finally`. A second client connecting
  wipes the first client's buffer; either disconnecting wipes the other's.
- `StreamingTranscriptionDomainService.process_audio_chunk` appends to one shared
  `InMemoryAudioBuffer.buffer` list. Two concurrent streams interleave float32 samples from
  different speakers into a single Whisper window.

The output is not degraded, it is **wrong** — and it fails silently. Both users get plausible-looking
transcriptions containing each other's words.

### The fix for now

Per the Chronus-side decision, real concurrency is deferred (§5). What is not acceptable is silent
corruption. Reject the second connection instead:

```python
# TranscriptionController.__init__
self._stream_in_use = False

# in stream_audio, after the origin check
if self._stream_in_use:
    await websocket.close(code=1013, reason="Transcription stream busy")
    return
self._stream_in_use = True
try:
    await websocket.accept()
    ...
finally:
    self._stream_in_use = False
    self.stream_audio_use_case.reset_stream()
```

Close code `1013` ("Try Again Later") is the correct RFC 6455 code for a temporarily-unavailable
service. The Chronus gateway maps it onto its own `4409` for the browser.

A single boolean is sufficient and correct here: uvicorn runs one event loop, and there is no
`await` between the check and the set.

**The `finally` must always run.** `reset_stream()` is currently in a `finally` — keep it there and
add the flag release alongside. A leaked flag makes the service permanently refuse connections until
restart, which is a worse failure than the one being fixed.

Also remove the `reset_stream()` call at the *top* of the `try`. With the busy guard in place,
`finally` already guarantees a clean buffer for the next connection, and the connect-time reset is
what makes a second connection destructive.

## 5. Deferred: per-connection streaming state

Not in scope. Recorded so the design is not re-derived later.

The fix is to stop treating the streaming pipeline as a singleton. In `app/di/container.py`, expose
a **factory** rather than an instance:

```python
def create_streaming_session(self) -> StreamAudioUseCase:
    """A fresh buffer per connection. The Whisper engine is stateless across
    generate() calls and can stay shared; only the buffer must be per-session."""
    return StreamAudioUseCaseImpl(
        StreamingTranscriptionDomainService(
            InMemoryAudioBuffer(self._audio_config),      # per-session
            self._streaming_transcription_engine,          # shared, read-only
        )
    )
```

`stream_audio` then calls `self.create_streaming_session()` per connection and drops the global
`reset_stream()` entirely.

Two things to verify before doing this:

1. `ChunkedWhisperTranscriptionEngine.reset_stream_state()` — confirm the engine holds no
   cross-call state (decoder context, prompt carry-over). If it does, it must be per-session too,
   and that is more expensive than a buffer.
2. GPU contention becomes the new limit. Concurrent `model.generate()` calls on one CUDA device
   will serialise or OOM. A semaphore sized to available VRAM is the follow-up, not optional.

Until both are done, §4's busy guard is the correct behaviour.

## 6. Deferred: shared-secret hook

Not enabled. Sketched so turning it on later is a config change, not a redesign.

```python
# in stream_audio, after the origin check
expected = config.stream.token
if expected and websocket.headers.get("x-thoth-token") != expected:
    await websocket.close(code=4401, reason="Unauthorized")
    return
```

```bash
# env.example — leave empty to disable
STREAM_TOKEN=
```

Chronus side: `THOTH_STREAM_TOKEN` in `backend/.env`, sent by `ThothStreamRemoteCaller` as an
`x-thoth-token` handshake header.

**Turn this on if any of these become true:** thoth becomes reachable from a network you do not
control; the Chronus backend and thoth stop sharing a trusted segment; or another service is granted
access to thoth and you need to distinguish callers. Until then it is a key to manage for no gain
over §3 plus §7.

## 7. Network posture

This is the primary control. §3 and §4 are what stands when this is misconfigured.

- **Bind to the private interface.** `HOST=0.0.0.0` in `env.example` binds everywhere. Set
  `HOST=172.16.0.49` (or the container's private address) so thoth is not listening on anything
  publicly routable.
- **Firewall `:8443` to the Chronus backend host only.** Ingress allowlist of exactly one source
  address. This is the step that makes the direct browser path impossible rather than merely
  rejected.
- **Keep the TLS cert.** Chronus trusts it explicitly via `THOTH_CA_CERT`; it will not accept
  `rejectUnauthorized: false`. If the cert is regenerated, redistribute the PEM to the Chronus host
  or the gateway starts failing with `1011`.
- If thoth runs under `docker-compose.yml`, publish the port to the private interface only
  (`172.16.0.49:8443:8443`), not `8443:8443` — the latter bypasses host firewall rules on many
  Docker setups, which is a common and quiet way to undo this whole section.

## 8. Documentation changes

The single-session constraint (§4) is invisible in the code and cost real debugging time. Make it
loud:

- `README.md` → "Real-time streaming" section: state that `/stream-audio` serves **one connection at
  a time** and that concurrent connections are rejected with `1013`. Link this spec.
- `ARCHITECTURE.md` → note that `InMemoryAudioBuffer` and the streaming domain service are
  process-wide singletons and why that constrains the WebSocket endpoint.
- `env.example` → `STREAM_ALLOWED_ORIGINS` and `STREAM_TOKEN` with the comments from §3 and §6.
- A comment at the `_stream_in_use` guard pointing at §4 and §5, so the next person reads the
  reasoning before "simplifying" it away.

## 9. Testing

- **Origin rejection:** handshake with `Origin: https://chronus.cc-an.com` → closed `4403`.
  Handshake with no `Origin` → accepted. Handshake with an allowlisted origin when
  `STREAM_ALLOWED_ORIGINS` is populated → accepted.
- **Busy guard:** open one stream, attempt a second → closed `1013`. Close the first, open a new
  one → accepted (proves the flag is released).
- **Flag release on crash:** kill a stream mid-chunk with an exception → next connection is
  accepted, not stuck at `1013`.
- **Buffer isolation:** stream audio, disconnect, reconnect, stream different audio → the second
  transcription contains nothing from the first.
- **Regression:** `/transcribe/`, `/upload`, `/transcribe/batch`, `/v1/audio/transcriptions`,
  `/health`, `/performance` all unchanged. None of this touches them.

Manual:

```bash
# must be rejected — browsers always send Origin
wscat -c "wss://172.16.0.49:8443/stream-audio" -H "Origin: https://chronus.cc-an.com"

# must be accepted — no Origin, like the Chronus gateway
wscat -c "wss://172.16.0.49:8443/stream-audio"

# from a client machine, after §7 — must fail to connect at all, not merely be rejected
wscat -c "wss://172.16.0.49:8443/stream-audio"
```

The last one is the acceptance test for the whole effort. "Rejected" means the code works.
"Cannot connect" means the architecture works.

## 10. Non-goals

- The HTTP upload endpoints stay unauthenticated. They are behind the same firewall; if that
  changes, they need §6 too — and they are the more attractive target, since `/transcribe/` accepts
  100 MB files and burns GPU time per request.
- No rate limiting inside thoth. The Chronus gateway owns per-user limits and session duration caps;
  duplicating that here would give two places to be wrong.
