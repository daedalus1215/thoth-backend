"""
Access-control and concurrency guards on the /stream-audio WebSocket.

These tests deliberately avoid the DI container, so they run without torch,
CUDA, or a Whisper checkpoint:

    pip install fastapi pytest httpx
    pytest tests/

See specs/stream-audio-hardening.md.
"""
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.controllers.transcription_controller import TranscriptionController
from app.config.settings import config
from domain.entities.transcription import Transcription
from domain.value_objects.audio_config import AudioConfig


class StubStreamUseCase:
    """Records what reached the streaming pipeline, without running Whisper."""

    def __init__(self):
        self.received_chunks = []
        self.reset_calls = 0
        self.next_text = None

    async def execute(self, audio_chunk, audio_config):
        self.received_chunks.append(audio_chunk)
        if self.next_text is None:
            return None
        return Transcription(text=self.next_text)

    def reset_stream(self):
        self.reset_calls += 1


@pytest.fixture(autouse=True)
def reset_stream_config():
    """Each test starts from the shipped defaults: no origins, no token."""
    original_origins = config.stream.allowed_origins
    original_token = config.stream.token
    config.stream.allowed_origins = []
    config.stream.token = ""
    yield
    config.stream.allowed_origins = original_origins
    config.stream.token = original_token


@pytest.fixture
def stream_use_case():
    return StubStreamUseCase()


@pytest.fixture
def client(stream_use_case):
    controller = TranscriptionController(
        transcribe_audio_use_case=None,
        stream_audio_use_case=stream_use_case,
        audio_config=AudioConfig(),
    )
    app = FastAPI()
    app.include_router(controller.router)
    return TestClient(app)


def audio_chunk(samples=4096):
    return np.zeros(samples, dtype=np.float32).tobytes()


class TestOriginRejection:
    """Browsers always send Origin; server-side clients do not."""

    def test_rejects_a_browser_origin(self, client):
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect(
                "/stream-audio", headers={"origin": "https://chronus.cc-an.com"}
            ):
                pass

        assert excinfo.value.code == 4403

    def test_accepts_a_handshake_with_no_origin(self, client):
        with client.websocket_connect("/stream-audio") as ws:
            ws.send_bytes(audio_chunk())

    def test_accepts_an_allowlisted_origin(self, client):
        config.stream.allowed_origins = ["https://debug.local"]

        with client.websocket_connect(
            "/stream-audio", headers={"origin": "https://debug.local"}
        ) as ws:
            ws.send_bytes(audio_chunk())

    def test_still_rejects_an_origin_outside_the_allowlist(self, client):
        config.stream.allowed_origins = ["https://debug.local"]

        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect(
                "/stream-audio", headers={"origin": "https://evil.example"}
            ):
                pass

        assert excinfo.value.code == 4403


class TestSharedSecret:
    """Disabled by default; enabling it must not break the no-token path."""

    def test_rejects_a_missing_token_when_configured(self, client):
        config.stream.token = "s3cret"

        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect("/stream-audio"):
                pass

        assert excinfo.value.code == 4401

    def test_accepts_the_configured_token(self, client):
        config.stream.token = "s3cret"

        with client.websocket_connect(
            "/stream-audio", headers={"x-thoth-token": "s3cret"}
        ) as ws:
            ws.send_bytes(audio_chunk())

    def test_no_token_required_by_default(self, client):
        with client.websocket_connect("/stream-audio") as ws:
            ws.send_bytes(audio_chunk())


class TestSingleSessionGuard:
    """
    The audio buffer is a process-wide singleton, so a second concurrent stream
    would corrupt the first. Reject it instead of corrupting silently.
    """

    def test_rejects_a_second_concurrent_stream(self, client):
        with client.websocket_connect("/stream-audio"):
            with pytest.raises(WebSocketDisconnect) as excinfo:
                with client.websocket_connect("/stream-audio"):
                    pass

        assert excinfo.value.code == 1013

    def test_releases_the_guard_after_a_clean_disconnect(self, client):
        with client.websocket_connect("/stream-audio") as ws:
            ws.send_bytes(audio_chunk())

        with client.websocket_connect("/stream-audio") as ws:
            ws.send_bytes(audio_chunk())

    def test_releases_the_guard_after_an_error(self, client, stream_use_case):
        async def explode(chunk, audio_config):
            raise RuntimeError("engine exploded")

        original_execute = stream_use_case.execute
        stream_use_case.execute = explode
        with client.websocket_connect("/stream-audio") as ws:
            ws.send_bytes(audio_chunk())

        # A leaked flag would make the service refuse every later connection.
        stream_use_case.execute = original_execute
        with client.websocket_connect("/stream-audio") as ws:
            ws.send_bytes(audio_chunk())


class TestStreamingBehaviour:
    def test_relays_audio_and_returns_transcriptions(self, client, stream_use_case):
        stream_use_case.next_text = "hello world"
        chunk = audio_chunk()

        with client.websocket_connect("/stream-audio") as ws:
            ws.send_bytes(chunk)
            assert ws.receive_json() == {"transcription": "hello world"}

        assert stream_use_case.received_chunks == [chunk]

    def test_does_not_reset_the_buffer_on_connect(self, client, stream_use_case):
        """
        Resetting on connect is what made a second connection destructive to the
        first. The `finally` reset alone keeps the next session clean.
        """
        with client.websocket_connect("/stream-audio") as ws:
            ws.send_bytes(audio_chunk())
            assert stream_use_case.reset_calls == 0

        assert stream_use_case.reset_calls == 1

    def test_rejected_handshakes_never_touch_the_pipeline(
        self, client, stream_use_case
    ):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/stream-audio", headers={"origin": "https://chronus.cc-an.com"}
            ):
                pass

        assert stream_use_case.received_chunks == []
        assert stream_use_case.reset_calls == 0
