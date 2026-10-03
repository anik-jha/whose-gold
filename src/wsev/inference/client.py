"""Concurrent client for a local llama.cpp OpenAI-compatible server.

Deliberately not an async framework: a bounded thread pool over urllib is enough
for a local server with N parallel slots, and it keeps the failure modes obvious.

Every request is deterministic (temperature 0, fixed seed) and every response is
persisted verbatim, because the playbook's discriminator for a suspicious
aggregate is always inspecting the raw sample outputs.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence


@dataclass
class ServerConfig:
    base_url: str = "http://127.0.0.1:8081"
    concurrency: int = 8
    timeout_s: float = 300.0
    max_retries: int = 3
    retry_backoff_s: float = 2.0
    seed: int = 1234
    temperature: float = 0.0
    disable_thinking: bool = True


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_calls: int = 0
    n_retries: int = 0
    wall_s: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, pt: int, ct: int, retries: int) -> None:
        with self._lock:
            self.prompt_tokens += pt
            self.completion_tokens += ct
            self.n_calls += 1
            self.n_retries += retries


class LlamaClient:
    """Minimal OpenAI-compatible chat client with retries and usage accounting."""

    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self.usage = Usage()

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.cfg.base_url}/health", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def model_name(self) -> str:
        with urllib.request.urlopen(f"{self.cfg.base_url}/v1/models", timeout=10) as r:
            d = json.load(r)
        models = d.get("models") or d.get("data") or []
        return (models[0].get("name") or models[0].get("id")) if models else "unknown"

    def chat(self, messages: Sequence[dict], *, max_tokens: int = 64,
             json_schema: dict | None = None, grammar: str | None = None,
             top_logprobs: int = 0) -> dict:
        body: dict[str, Any] = {
            "model": "local",
            "messages": list(messages),
            "temperature": self.cfg.temperature,
            "seed": self.cfg.seed,
            "max_tokens": max_tokens,
        }
        if self.cfg.disable_thinking:
            # Verified against llama-server 9307 on 2026-08-09: reasoning_effort="none"
            # still emits reasoning_content and leaves content empty; this kwarg does not.
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "out", "strict": True, "schema": json_schema},
            }
        if grammar is not None:
            body["grammar"] = grammar
        if top_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs

        payload = json.dumps(body).encode()
        last: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                req = urllib.request.Request(
                    f"{self.cfg.base_url}/v1/chat/completions",
                    data=payload, headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as r:
                    d = json.load(r)
                u = d.get("usage", {})
                self.usage.add(u.get("prompt_tokens", 0), u.get("completion_tokens", 0), attempt)
                return d
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
                last = e
                time.sleep(self.cfg.retry_backoff_s * (attempt + 1))
        raise RuntimeError(f"chat failed after {self.cfg.max_retries} attempts: {last}")

    def map(self, items: Iterable[Any], fn: Callable[[Any], Any], *,
            on_result: Callable[[Any], None] | None = None) -> list[Any]:
        """Run `fn` over items with bounded concurrency, preserving input order.

        `on_result` is called from the main thread as results arrive, so callers
        can stream to disk and stay resumable if the run dies.
        """
        t0 = time.time()
        out: list[Any] = []
        with ThreadPoolExecutor(max_workers=self.cfg.concurrency) as ex:
            for res in ex.map(fn, items):
                out.append(res)
                if on_result is not None:
                    on_result(res)
        self.usage.wall_s += time.time() - t0
        return out
