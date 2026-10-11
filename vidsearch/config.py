"""Settings, read from environment variables (VS_*) with sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    # Where per-film artifacts (frames, atlases, embeddings) are written.
    data_dir: Path = field(default_factory=lambda: Path(_env("VS_DATA", "data")))

    # Frame sampling
    sample_fps: float = field(default_factory=lambda: float(_env("VS_SAMPLE_FPS", "1.0")))
    frame_width: int = field(default_factory=lambda: int(_env("VS_FRAME_WIDTH", "384")))

    # Shot detection: "content" (PySceneDetect default, HSV frame diff) or "adaptive"
    # (rolling-average version; more tolerant of camera motion and flicker, but can miss cuts).
    # Threshold "" = default (content 18, adaptive 3.0). Lower = more cuts.
    shot_detector: str = field(default_factory=lambda: _env("VS_SHOT_DETECTOR", "content"))
    shot_threshold: str = field(default_factory=lambda: _env("VS_SHOT_THRESHOLD", ""))

    # Thumbnail atlases for the frontend (one texture page holds many cells)
    thumb_width: int = field(default_factory=lambda: int(_env("VS_THUMB_WIDTH", "96")))
    atlas_max_px: int = field(default_factory=lambda: int(_env("VS_ATLAS_MAX", "4096")))

    # Sliding windows inside each shot; these are the "moments" search returns
    window_sec: float = field(default_factory=lambda: float(_env("VS_WINDOW_SEC", "3.0")))
    window_stride: float = field(default_factory=lambda: float(_env("VS_WINDOW_STRIDE", "1.0")))

    # Captions are generated per shot chunk; long shots are split into chunks
    caption_chunk_sec: float = field(default_factory=lambda: float(_env("VS_CAPTION_CHUNK_SEC", "12")))

    # Models
    image_model: str = field(default_factory=lambda: _env("VS_IMAGE_MODEL", "ViT-B-16-SigLIP2"))
    image_pretrained: str = field(default_factory=lambda: _env("VS_IMAGE_PRETRAINED", "webli"))
    text_model: str = field(default_factory=lambda: _env("VS_TEXT_MODEL", "BAAI/bge-small-en-v1.5"))
    device: str = field(default_factory=lambda: _env("VS_DEVICE", ""))  # "" = auto

    # OpenAI-compatible vision endpoint for captions (OpenAI, Gemini, OpenRouter, vLLM, Ollama...)
    caption_base_url: str = field(default_factory=lambda: _env("VS_CAPTION_BASE_URL", "https://api.openai.com/v1"))
    caption_model: str = field(default_factory=lambda: _env("VS_CAPTION_MODEL", "gpt-4o-mini"))
    caption_api_key: str = field(default_factory=lambda: _env("VS_CAPTION_API_KEY", ""))

    # Built frontend (web/dist) to serve at /, so one server hosts API, media and app. "" = API only.
    web_dist: str = field(default_factory=lambda: _env("VS_WEB_DIST", ""))

    # Auth0 sign-in for uploads. Empty domain = auth off: everyone is anonymous and uploads return 503.
    auth0_domain: str = field(default_factory=lambda: _env("VS_AUTH0_DOMAIN", ""))
    auth0_audience: str = field(default_factory=lambda: _env("VS_AUTH0_AUDIENCE", "https://vidsearch-api"))
    # Local testing only: with no Auth0 domain, accept `Bearer dev:<sub>` as user <sub>. Ignored when a domain is set.
    dev_auth: bool = field(default_factory=lambda: _env("VS_DEV_AUTH", "") == "1")

    # Uploaded and downloaded source videos land here as <film_id><ext>.
    videos_dir: Path = field(default_factory=lambda: Path(_env("VS_VIDEOS", "videos")))
    uploads_per_day: int = field(default_factory=lambda: int(_env("VS_UPLOADS_PER_DAY", "3")))
    max_upload_bytes: int = field(default_factory=lambda: int(_env("VS_MAX_UPLOAD_BYTES", str(2 * 1024**3))))
    max_upload_minutes: float = field(default_factory=lambda: float(_env("VS_MAX_UPLOAD_MINUTES", "60")))
    upload_part_bytes: int = field(default_factory=lambda: int(_env("VS_UPLOAD_PART_BYTES", str(8 * 1024**2))))

    @property
    def films_dir(self) -> Path:
        return self.data_dir / "films"

    @property
    def uploads_dir(self) -> Path:
        """Parts of in-flight chunked uploads, one dir per upload. Not served."""
        return self.data_dir / "uploads"
