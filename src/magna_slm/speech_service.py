from __future__ import annotations

import asyncio
import io
import os
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel
import edge_tts
from piper import PiperVoice

ROOT = Path(__file__).resolve().parents[2]
PIPER_MODEL_PATH = ROOT / "models/piper/en_US-lessac-medium.onnx"


class SpeechService:
    def __init__(self, model_size: str | None = None, device: str = "cuda") -> None:
        self.model_size = model_size or os.getenv("STT_MODEL", "tiny")
        self.device = device
        self._stt_model: WhisperModel | None = None
        self._stt_lock = threading.Lock()
        self._piper_voice: PiperVoice | None = None
        self._piper_lock = threading.Lock()

    def _get_stt_model(self) -> WhisperModel:
        if self._stt_model is None:
            with self._stt_lock:
                if self._stt_model is None:
                    dev = self.device
                    comp_type = "float16" if dev == "cuda" else "int8"
                    try:
                        self._stt_model = WhisperModel(
                            self.model_size, device=dev, compute_type=comp_type
                        )
                    except Exception:
                        self._stt_model = WhisperModel(
                            self.model_size, device="cpu", compute_type="int8"
                        )
        return self._stt_model

    def _get_piper_voice(self) -> PiperVoice | None:
        if self._piper_voice is None and PIPER_MODEL_PATH.is_file():
            with self._piper_lock:
                if self._piper_voice is None and PIPER_MODEL_PATH.is_file():
                    try:
                        self._piper_voice = PiperVoice.load(str(PIPER_MODEL_PATH))
                    except Exception as exc:
                        print(f"[TTS] Warning: Could not load Piper voice: {exc}")
        return self._piper_voice

    def transcribe(
        self,
        audio_data: bytes | io.BytesIO,
        filename: str = "audio.webm",
        language: str | None = None,
    ) -> str:
        """Transcribe audio bytes (WebM, WAV, MP3, etc.) using faster-whisper on GPU."""
        model = self._get_stt_model()
        suffix = Path(filename).suffix or ".webm"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            if isinstance(audio_data, io.BytesIO):
                tmp.write(audio_data.getvalue())
            else:
                tmp.write(audio_data)
            tmp_path = tmp.name

        try:
            segments, _ = model.transcribe(
                tmp_path,
                language=language,
                beam_size=5,
                vad_filter=True,
            )
            text_parts = [segment.text.strip() for segment in segments]
            return " ".join(t for t in text_parts if t).strip()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def synthesize_local(self, text: str) -> tuple[bytes, str]:
        """Synthesize text using local offline Piper neural voice."""
        voice = self._get_piper_voice()
        if voice is None:
            raise RuntimeError("Local Piper voice model is not available")

        chunks = list(voice.synthesize(text))
        if not chunks:
            return b"", "audio/wav"

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(chunks[0].sample_channels)
            wav.setsampwidth(chunks[0].sample_width)
            wav.setframerate(chunks[0].sample_rate)
            for c in chunks:
                wav.writeframes(c.audio_int16_bytes)

        return buf.getvalue(), "audio/wav"

    async def synthesize_async(
        self,
        text: str,
        voice: str = "local",
    ) -> tuple[bytes, str]:
        """Synthesize text to audio (MP3 or WAV bytes)."""
        if not text.strip():
            return b"", "audio/mpeg"

        # Explicit local offline voice requested
        if voice in ("local", "offline", "piper", "en_US-lessac-medium"):
            return self.synthesize_local(text)

        # Attempt cloud neural synthesis via edge-tts with local fallback
        try:
            communicate = edge_tts.Communicate(text, voice)
            out = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    out.write(chunk["data"])
            audio_data = out.getvalue()
            if audio_data:
                return audio_data, "audio/mpeg"
        except Exception as exc:
            print(f"[TTS] Cloud voice failed ({exc}), falling back to local Piper voice.")

        # Fallback to local offline TTS
        return self.synthesize_local(text)

    def synthesize(
        self,
        text: str,
        voice: str = "local",
    ) -> tuple[bytes, str]:
        """Synchronous wrapper for synthesize_async returning (audio_bytes, mime_type)."""
        return asyncio.run(self.synthesize_async(text, voice))

