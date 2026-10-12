from __future__ import annotations

import atexit
import json
import os
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer


class OlmoeLocalClient:
    def __init__(self, model_dir: Path, engine: Path, tokenizer: Path) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.engine = Path(engine).resolve()
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.req_counter = 0
        atexit.register(self.close)
        self._ensure_worker()

    def _ensure_worker(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        env = {
            **os.environ,
            "SNAP": str(self.model_dir),
            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS", "6"),
        }
        self.proc = subprocess.Popen(
            [str(self.engine), "64", "8", "-"],
            cwd=self.engine.parent,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self.proc.stdout is not None:
            for line in self.proc.stdout:
                line_str = line.strip()
                if line_str == "READY":
                    break

    def format_chat_prompt(self, messages_or_text: list[dict[str, str]] | str) -> str:
        if isinstance(messages_or_text, str):
            if "<|user|>" in messages_or_text or "<|assistant|>" in messages_or_text:
                return messages_or_text
            return f"|||IP_ADDRESS|||<|user|>\n{messages_or_text.strip()}\n<|assistant|>\n"

        parts: list[str] = ["|||IP_ADDRESS|||"]
        for msg in messages_or_text:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if not content:
                continue
            if role == "system":
                parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}|||IP_ADDRESS|||\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def stream_generate(
        self,
        messages_or_text: list[dict[str, str]] | str,
        max_new_tokens: int = 256,
    ) -> Iterator[str]:
        prompt = self.format_chat_prompt(messages_or_text)
        prompt_ids = self.tokenizer.encode(prompt).ids

        with self.lock:
            self._ensure_worker()
            if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
                raise RuntimeError("Failed to start OLMoE worker process")

            self.req_counter += 1
            req_id = self.req_counter
            req_str = f"REQ {req_id} {max_new_tokens} {' '.join(str(i) for i in prompt_ids)}\n"

            try:
                self.proc.stdin.write(req_str)
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self.proc = None
                self._ensure_worker()
                if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
                    raise RuntimeError("Failed to restart OLMoE worker process")
                self.proc.stdin.write(req_str)
                self.proc.stdin.flush()

            stopped = False
            try:
                while True:
                    line = self.proc.stdout.readline()
                    if not line:
                        break
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if parts[0] == "TOKEN" and len(parts) >= 3:
                        cur_req = int(parts[1])
                        if cur_req != req_id:
                            continue
                        if not stopped:
                            tok_id = int(parts[2])
                            if tok_id in (0, 50279):
                                stopped = True
                            else:
                                delta = self.tokenizer.decode([tok_id])
                                if delta:
                                    yield delta
                    elif parts[0] == "DONE" and len(parts) >= 2:
                        cur_req = int(parts[1])
                        if cur_req == req_id:
                            break
            except BaseException:
                try:
                    if self.proc and self.proc.stdin and not self.proc.stdin.closed:
                        self.proc.stdin.write(f"CANCEL {req_id}\n")
                        self.proc.stdin.flush()
                        while True:
                            line = self.proc.stdout.readline()
                            if not line:
                                break
                            parts = line.strip().split()
                            if parts and parts[0] == "DONE" and len(parts) >= 2:
                                if int(parts[1]) == req_id:
                                    break
                except Exception:
                    pass
                raise

    def generate(
        self,
        messages_or_text: list[dict[str, str]] | str,
        max_new_tokens: int = 256,
    ) -> str:
        return "".join(self.stream_generate(messages_or_text, max_new_tokens))

    def close(self) -> None:
        if self.proc is not None:
            try:
                if self.proc.stdin and not self.proc.stdin.closed:
                    self.proc.stdin.write("QUIT\n")
                    self.proc.stdin.flush()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None