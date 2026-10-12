from __future__ import annotations

import json
import mimetypes
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .olmoe_client import OlmoeLocalClient
from .speech_service import SpeechService

ROOT = Path(__file__).resolve().parents[2]
WEB_DIST = ROOT / "third_party/colibri/web/dist"
CLIENT = OlmoeLocalClient(
    ROOT / "models/olmoe_i4",
    ROOT / "third_party/colibri/c/olmoe",
    ROOT / "models/olmoe_hf/tokenizer.json",
)
SPEECH = SpeechService()

INFERENCE_LOCK = threading.Lock()
STATS = {
    "admitted": 0,
    "completed": 0,
    "active": 0,
}


class Handler(BaseHTTPRequestHandler):
    server_version = "ColibriOlmoe/0.1"

    def _set_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, Accept, X-Requested-With, x-colibri-queue-wait-ms, x-request-id",
        )
        self.send_header("Access-Control-Max-Age", "86400")

    def _send_json(self, status: int, body: dict[str, object]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._set_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _clean_path(self) -> str:
        parsed = urlparse(self.path)
        return parsed.path.rstrip("/") or "/"

    def _serve_file(self, file_path: Path) -> None:
        if not file_path.is_file():
            self._send_json(404, {"error": "not found"})
            return
        content = file_path.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type is None:
            mime_type = "application/octet-stream"
        if mime_type.startswith("text/") or mime_type in (
            "application/javascript",
            "application/json",
        ):
            mime_type += "; charset=utf-8"

        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(content)))
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        clean = self._clean_path()

        if clean in ("/health", "/v1/health"):
            self._send_json(
                200,
                {
                    "status": "ok",
                    "model": "olmoe",
                    "kv_slots": 1,
                    "scheduler": {
                        "active": STATS["active"],
                        "capacity": 1,
                        "queued": 0,
                        "max_queue": 8,
                        "queue_timeout_seconds": 300,
                        "admitted": STATS["admitted"],
                        "completed": STATS["completed"],
                        "rejected": 0,
                        "timed_out": 0,
                        "cancelled": 0,
                    },
                },
            )
            return

        if clean in ("/v1/models", "/models"):
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "olmoe",
                            "object": "model",
                            "owned_by": "local",
                            "permission": [],
                            "root": "olmoe",
                        }
                    ],
                },
            )
            return

        if clean in ("/v1/audio/voices", "/audio/voices"):
            self._send_json(
                200,
                {
                    "voices": [
                        {"id": "local", "name": "Piper Lessac (100% Offline Local)", "lang": "en-US"},
                        {"id": "en-US-JennyNeural", "name": "Jenny (US Female)", "lang": "en-US"},
                        {"id": "en-US-GuyNeural", "name": "Guy (US Male)", "lang": "en-US"},
                        {"id": "en-IN-NeerjaNeural", "name": "Neerja (India English)", "lang": "en-IN"},
                        {"id": "en-GB-SoniaNeural", "name": "Sonia (UK Female)", "lang": "en-GB"},
                        {"id": "hi-IN-SwaraNeural", "name": "Swara (Hindi)", "lang": "hi-IN"},
                    ]
                },
            )
            return

        # Serve static UI if available
        if WEB_DIST.is_dir():
            if clean in ("/", "/index.html"):
                self._serve_file(WEB_DIST / "index.html")
                return

            candidate = (WEB_DIST / clean.lstrip("/")).resolve()
            if candidate.is_relative_to(WEB_DIST) and candidate.is_file():
                self._serve_file(candidate)
                return

            # Client-side router fallback to index.html for non-asset GET requests
            if "." not in clean:
                self._serve_file(WEB_DIST / "index.html")
                return

        self._send_json(404, {"error": "not found"})

    def _handle_audio_transcription(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self._send_json(400, {"error": "Empty audio payload"})
            return

        content_type = self.headers.get("Content-Type", "")
        raw_body = self.rfile.read(length)

        filename = "audio.webm"
        audio_data = raw_body

        if "multipart/form-data" in content_type:
            boundary_match = re.search(r"boundary=([^\s;]+)", content_type)
            if boundary_match:
                boundary = boundary_match.group(1).encode("utf-8").strip(b'"')
                parts = raw_body.split(b"--" + boundary)
                for part in parts:
                    if b'name="file"' in part:
                        fn_match = re.search(rb'filename="([^"]+)"', part)
                        if fn_match:
                            filename = fn_match.group(1).decode("utf-8", errors="ignore")
                        header_end = part.find(b"\r\n\r\n")
                        if header_end != -1:
                            audio_data = part[header_end + 4 :].rstrip(b"\r\n-")
                            break

        if not audio_data:
            self._send_json(400, {"error": "No audio data found in request"})
            return

        t0 = time.time()
        try:
            text = SPEECH.transcribe(audio_data, filename=filename)
            print(f"[STT] Transcribed {len(audio_data)} bytes in {time.time() - t0:.2f}s: {text!r}")
            self._send_json(200, {"text": text})
        except Exception as exc:
            print(f"[STT] Error transcribing: {exc}")
            self._send_json(500, {"error": f"Transcription failed: {exc}"})

    def _handle_audio_speech(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            req = json.loads(raw_body)
        except Exception as exc:
            self._send_json(400, {"error": f"Invalid JSON payload: {exc}"})
            return

        text = req.get("input", "").strip()
        voice = req.get("voice", "en-US-JennyNeural")
        if not text:
            self._send_json(400, {"error": "Missing 'input' text"})
            return

        t0 = time.time()
        try:
            audio_bytes, mime_type = SPEECH.synthesize(text, voice=voice)
            print(f"[TTS] Synthesized {len(text)} chars to {len(audio_bytes)} bytes ({mime_type}) in {time.time() - t0:.2f}s")
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(audio_bytes)))
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(audio_bytes)
        except Exception as exc:
            print(f"[TTS] Error synthesizing: {exc}")
            self._send_json(500, {"error": f"Speech synthesis failed: {exc}"})

    def do_POST(self) -> None:
        clean = self._clean_path()
        if clean in ("/v1/audio/transcriptions", "/audio/transcriptions"):
            self._handle_audio_transcription()
            return

        if clean in ("/v1/audio/speech", "/audio/speech"):
            self._handle_audio_speech()
            return

        if clean not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            request = json.loads(raw_body)
        except Exception as exc:
            self._send_json(400, {"error": f"Invalid JSON payload: {exc}"})
            return

        messages = request.get("messages", [])
        model = request.get("model", "olmoe")
        stream = bool(request.get("stream", False))
        max_tokens = int(
            request.get("max_completion_tokens")
            or request.get("max_tokens")
            or 256
        )
        # Cap max_tokens to a reasonable limit for interactive generation
        max_tokens = max(1, min(max_tokens, 1024))

        prompt_str = CLIENT.format_chat_prompt(messages)
        prompt_tokens = len(CLIENT.tokenizer.encode(prompt_str).ids)
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
        last_query = user_msgs[-1][:60] if user_msgs else "(none)"
        print(f"[{req_id}] Admitted: {len(messages)} messages (prompt_tokens={prompt_tokens}, query={last_query!r}, stream={stream})")

        t_start = time.time()
        with INFERENCE_LOCK:
            STATS["active"] += 1
            STATS["admitted"] += 1
            try:
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.send_header("X-Accel-Buffering", "no")
                    self._set_cors_headers()
                    self.end_headers()

                    token_count = 0
                    try:
                        for delta in CLIENT.stream_generate(messages, max_new_tokens=max_tokens):
                            token_count += 1
                            chunk = {
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": delta},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                            self.wfile.flush()

                        final_chunk = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": token_count,
                                "total_tokens": prompt_tokens + token_count,
                            },
                        }
                        self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8"))
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        self.close_connection = True
                        elapsed = max(0.001, time.time() - t_start)
                        print(f"[{req_id}] Done: {token_count} tokens in {elapsed:.2f}s ({token_count / elapsed:.1f} tok/s)")
                    except (BrokenPipeError, ConnectionResetError):
                        # Client disconnected prematurely
                        elapsed = max(0.001, time.time() - t_start)
                        print(f"[{req_id}] Cancelled by client after {token_count} tokens in {elapsed:.2f}s")
                else:
                    tokens = list(CLIENT.stream_generate(messages, max_new_tokens=max_tokens))
                    token_count = len(tokens)
                    full_text = "".join(tokens)
                    elapsed = max(0.001, time.time() - t_start)
                    print(f"[{req_id}] Done (non-stream): {token_count} tokens in {elapsed:.2f}s ({token_count / elapsed:.1f} tok/s)")
                    self._send_json(
                        200,
                        {
                            "id": req_id,
                            "object": "chat.completion",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": full_text},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": token_count,
                                "total_tokens": prompt_tokens + token_count,
                            },
                        },
                    )
            finally:
                STATS["active"] -= 1
                STATS["completed"] += 1


def run_server() -> None:
    host = os.getenv("COLIBRI_HOST", "0.0.0.0")
    port = int(os.getenv("COLIBRI_PORT", "8001"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Colibri OLMoE server listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        CLIENT.close()
        server.server_close()


if __name__ == "__main__":
    run_server()