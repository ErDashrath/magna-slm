from __future__ import annotations

from langchain_openai import ChatOpenAI


def colibri_model(
    base_url: str = "http://127.0.0.1:8001/v1",
    model: str = "olmoe",
) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=base_url,
        api_key="local",
        model=model,
        temperature=0,
    )