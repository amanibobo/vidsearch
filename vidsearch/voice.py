"""Voice search: single-use ElevenLabs realtime speech-to-text tokens, minted here and rate-limited.

The browser streams microphone audio straight to ElevenLabs with a token from GET /api/voice/token, so the
API key stays on the server. Every token spends the account's credits, so minting counts against sliding
windows per browser identity, per hashed client IP and across everyone (in memory: one web container).
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Protocol

import httpx

from .config import Settings

log = logging.getLogger("vidsearch.voice")

TOKEN_URL = "https://api.elevenlabs.io/v1/single-use-token/realtime_scribe"
TOKEN_TTL_S = 900


class VoiceUnavailable(Exception):
    """ElevenLabs refused or failed to mint a token. The message is safe to show the user."""


class TokenMinter(Protocol):
    def mint(self) -> str:
        """A fresh single-use realtime token. Raises VoiceUnavailable."""


class ElevenLabsMinter:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def mint(self) -> str:
        try:
            r = httpx.post(TOKEN_URL, headers={"xi-api-key": self.api_key}, timeout=10)
            r.raise_for_status()
            return r.json()["token"]
        except (httpx.HTTPError, KeyError, ValueError) as e:
            # Type and status only: nothing from the request (it carries the key) reaches a log or a response.
            status = e.response.status_code if isinstance(e, httpx.HTTPStatusError) else None
            log.warning("ElevenLabs token mint failed: %s %s", type(e).__name__, status)
            raise VoiceUnavailable("the voice service is unavailable; try again or type your search") from None


@dataclass(frozen=True)
class Window:
    """At most `limit` tokens per `seconds` for one bucket; `refusal` is the 429 message."""
    key: str
    limit: int
    seconds: float
    refusal: str


class RateLimited(Exception):
    pass


class SlidingLimiter:
    """Timestamps of recent grants per bucket. A grant takes a slot in every window or in none."""

    def __init__(self):
        self.grants: dict[str, deque[float]] = {}
        self.lock = threading.Lock()

    def take(self, windows: list[Window], now: float) -> None:
        with self.lock:
            for w in windows:
                q = self.grants.setdefault(w.key, deque())
                while q and q[0] <= now - w.seconds:
                    q.popleft()
                if len(q) >= w.limit:
                    raise RateLimited(w.refusal)
            for w in windows:
                self.grants[w.key].append(now)


def voice_windows(s: Settings, user: str | None, ip_hash: str) -> list[Window]:
    """Every limit a token counts against, most specific first, so a refusal names the nearest one."""
    hour, day = 3600.0, 86400.0
    later = "try again later or type your search"
    per_user = [Window(f"user:{user}", s.voice_tokens_per_user_per_hour, hour,
                       f"voice search limit of {s.voice_tokens_per_user_per_hour} an hour reached; {later}")] if user else []
    return [
        *per_user,
        Window(f"ip:{ip_hash}", s.voice_tokens_per_ip_per_hour, hour,
               f"voice search limit of {s.voice_tokens_per_ip_per_hour} an hour from this network reached; {later}"),
        Window("all", s.voice_tokens_global_per_day, day,
               f"the archive's daily voice search limit is reached; {later}"),
    ]
