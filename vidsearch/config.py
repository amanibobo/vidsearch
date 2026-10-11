"""Settings, read from environment variables (VS_*) with sensible defaults.

VIDSEARCH_COLLECTION picks which archive's defaults apply (COLLECTIONS); empty means traffic."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Collection:
    """One archive's identity and defaults. Each VS_* setting below can still override its field."""
    title: str
    noun: str           # what one item is called in the UI; the plural adds "s" unless VS_ITEM_NOUN_PLURAL says otherwise
    caption_style: str  # key into caption.PROMPTS
    sample_fps: float
    # Phrasings of a search query whose SigLIP text vectors are averaged into the query vector.
    query_templates: tuple[str, ...]


DEFAULT_COLLECTION = "traffic"
COLLECTIONS = {
    "traffic": Collection(title="The Traffic Archive", noun="clip", caption_style="driving", sample_fps=2.0,
                          query_templates=("{}", "a dashcam photo of {}", "a traffic camera image of {}")),
}


def collection_name() -> str:
    """VIDSEARCH_COLLECTION, checked against COLLECTIONS; empty means the default."""
    name = os.environ.get("VIDSEARCH_COLLECTION", "") or DEFAULT_COLLECTION
    if name not in COLLECTIONS:
        raise ValueError(f"unknown VIDSEARCH_COLLECTION {name!r}; known: {', '.join(COLLECTIONS)}")
    return name


def _collection() -> Collection:
    return COLLECTIONS[collection_name()]


def _noun() -> str:
    return _env("VS_ITEM_NOUN", _collection().noun)


@dataclass
class Settings:
    # Where per-film artifacts (frames, atlases, embeddings) are written.
    data_dir: Path = field(default_factory=lambda: Path(_env("VS_DATA", "data")))

    # Frame sampling
    sample_fps: float = field(default_factory=lambda: float(_env("VS_SAMPLE_FPS", str(_collection().sample_fps))))
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
    # Which prompt describes a chunk (caption.PROMPTS).
    caption_style: str = field(default_factory=lambda: _env("VS_CAPTION_STYLE", _collection().caption_style))

    # What the frontend calls the archive and its items (masthead, tab title, stats line, upload copy).
    archive_title: str = field(default_factory=lambda: _env("VS_ARCHIVE_TITLE", _collection().title))
    item_noun: str = field(default_factory=_noun)
    item_noun_plural: str = field(default_factory=lambda: _env("VS_ITEM_NOUN_PLURAL", _noun() + "s"))

    # Models
    image_model: str = field(default_factory=lambda: _env("VS_IMAGE_MODEL", "ViT-B-16-SigLIP2"))
    image_pretrained: str = field(default_factory=lambda: _env("VS_IMAGE_PRETRAINED", "webli"))
    text_model: str = field(default_factory=lambda: _env("VS_TEXT_MODEL", "BAAI/bge-small-en-v1.5"))
    device: str = field(default_factory=lambda: _env("VS_DEVICE", ""))  # "" = auto
    query_templates: tuple[str, ...] = field(default_factory=lambda: _collection().query_templates)

    # OpenAI-compatible vision endpoint for captions (OpenAI, Gemini, OpenRouter, vLLM, Ollama...)
    caption_base_url: str = field(default_factory=lambda: _env("VS_CAPTION_BASE_URL", "https://api.openai.com/v1"))
    caption_model: str = field(default_factory=lambda: _env("VS_CAPTION_MODEL", "gpt-4o-mini"))
    caption_api_key: str = field(default_factory=lambda: _env("VS_CAPTION_API_KEY", ""))

    # Ingested videos are re-encoded to H.264 MP4 at most this wide (see jobs.normalize_video).
    max_video_width: int = field(default_factory=lambda: int(_env("VS_MAX_VIDEO_WIDTH", "1280")))

    # Built frontend (web/dist) to serve at /, so one server hosts API, media and app. "" = API only.
    web_dist: str = field(default_factory=lambda: _env("VS_WEB_DIST", ""))

    # Auth0 sign-in for uploads. Empty domain (and no anon_uploads) = auth off: everyone is anonymous and uploads return 503.
    auth0_domain: str = field(default_factory=lambda: _env("VS_AUTH0_DOMAIN", ""))
    auth0_audience: str = field(default_factory=lambda: _env("VS_AUTH0_AUDIENCE", "https://vidsearch-api"))
    # Local testing only: with no Auth0 domain, accept `Bearer dev:<sub>` as user <sub>. Ignored when a domain is set.
    dev_auth: bool = field(default_factory=lambda: _env("VS_DEV_AUTH", "") == "1")

    # Uploads without sign-in: `Bearer device:<32 lowercase hex>`, a random id the browser keeps, is a user.
    # Works with or without Auth0.
    anon_uploads: bool = field(default_factory=lambda: _env("VS_ANON_UPLOADS", "") == "1")

    # Uploaded and downloaded source videos land here as <film_id><ext>.
    videos_dir: Path = field(default_factory=lambda: Path(_env("VS_VIDEOS", "videos")))
    # Daily upload limits (UTC days), counted when an upload starts; a cancelled upload gives its slot back.
    uploads_per_day: int = field(default_factory=lambda: int(_env("VS_UPLOADS_PER_DAY", "3")))  # per user or device
    uploads_per_ip_per_day: int = field(default_factory=lambda: int(_env("VS_UPLOADS_PER_IP_PER_DAY", "5")))
    uploads_global_per_day: int = field(default_factory=lambda: int(_env("VS_UPLOADS_GLOBAL_PER_DAY", "20")))
    # Request header carrying the client IP behind a proxy, e.g. X-Forwarded-For. Its right-most entry is used,
    # since the proxy appends and anything left of that is whatever the client sent. "" = the TCP peer.
    client_ip_header: str = field(default_factory=lambda: _env("VS_CLIENT_IP_HEADER", ""))
    # Client IPs are only stored as sha256(salt, ip). "" = a random salt per process, so per-IP counts reset on restart.
    ip_salt: str = field(default_factory=lambda: _env("VS_IP_SALT", ""))
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
