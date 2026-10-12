from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    ram_gb: float
    vram_gb: float
    llm_on_gpu: bool
    speech_on_gpu: bool


COLIBRI_PROFILE = RuntimeProfile(
    name="colibri-t2000-16gb",
    ram_gb=16.0,
    vram_gb=4.0,
    llm_on_gpu=False,
    speech_on_gpu=True,
)


MODEL_IDS = {
    "llm": "allenai/OLMoE-1B-7B-0125-Instruct",
    "stt": "bodhan-ai/indic-transcribe-core",
    "tts": "bodhan-ai/indic-speak",
}