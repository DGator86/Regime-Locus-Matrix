"""
LLM client supporting Groq (free API) and Ollama (free local).

Groq  → sign up at console.groq.com, set GROQ_API_KEY in .env
         default model: llama-3.3-70b-versatile (free tier, 14k req/day)
Ollama → install ollama, run `ollama pull llama3.2`, set LLM_BACKEND=ollama
         model auto-selects; override with LLM_MODEL env var
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMConfig:
    backend: str = field(default_factory=lambda: os.environ.get("LLM_BACKEND", "groq"))
    model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", ""))
    api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    ollama_host: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    temperature: float = 0.3
    max_tokens: int = 1024
    timeout_sec: int = 60

    def __post_init__(self) -> None:
        if not self.model:
            self.model = "llama-3.3-70b-versatile" if self.backend == "groq" else "llama3.2"

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls()


class LLMClient:
    """Thin wrapper around Groq or Ollama chat-completion endpoints."""

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.cfg = config or LLMConfig.from_env()

    def chat(self, messages: list[Message], system: str = "") -> str:
        payload_msgs: list[dict] = []
        if system:
            payload_msgs.append({"role": "system", "content": system})
        for m in messages:
            payload_msgs.append({"role": m.role, "content": m.content})

        if self.cfg.backend == "groq":
            return self._call_groq(payload_msgs)
        return self._call_ollama(payload_msgs)

    def quick(self, prompt: str, system: str = "") -> str:
        return self.chat([Message("user", prompt)], system=system)

    # ------------------------------------------------------------------
    # Groq
    # ------------------------------------------------------------------
    def _call_groq(self, messages: list[dict]) -> str:
        if not self.cfg.api_key:
            raise RuntimeError("GROQ_API_KEY not set. Get a free key at console.groq.com and add it to .env")
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_sec) as resp:
                result = json.loads(resp.read().decode())
                return str(result["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RuntimeError(f"Groq HTTP {exc.code}: {body}") from exc

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------
    def _call_ollama(self, messages: list[dict]) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.cfg.temperature},
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.cfg.ollama_host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_sec) as resp:
                result = json.loads(resp.read().decode())
                return str(result["message"]["content"])
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama unreachable at {self.cfg.ollama_host}: {exc}. " "Is Ollama running? `ollama serve`"
            ) from exc
