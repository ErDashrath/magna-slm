from __future__ import annotations

import json
from urllib.request import Request, urlopen


class ColibriClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8001/v1") -> None:
        self.base_url = base_url.rstrip("/")

    def health(self) -> bool:
        request = Request(self.base_url.removesuffix("/v1") + "/health")
        try:
            with urlopen(request, timeout=3) as response:
                return 200 <= response.status < 300
        except OSError:
            return False

    def chat(self, message: str, model: str = "olmoe") -> str:
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": message}],
                "stream": False,
            }
        ).encode()
        request = Request(
            self.base_url + "/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
        return result["choices"][0]["message"]["content"]