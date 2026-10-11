"""Optional caption layer: a vision LLM describes each shot chunk, then captions are embedded
with a sentence-transformer. This is what makes action queries ("being chased", "falls off a
ladder") work, since a single frame embedding can't see motion.

Talks to any OpenAI-compatible /chat/completions endpoint that accepts image_url parts:
OpenAI, Gemini (https://generativelanguage.googleapis.com/v1beta/openai), OpenRouter,
or a local vLLM / Ollama server running Qwen2.5-VL etc. Model names starting with "claude-"
go to the Claude API through the anthropic SDK instead (key from ANTHROPIC_API_KEY).

Writes into each film dir:
    captions.jsonl     progress log, one {"i": chunk_index, "text": ...} per line (resumable)
    captions.json      [[start, end, shot, f0, f1, text], ...]
    emb_captions.npy   float16 caption embeddings
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from .config import Settings
from .ingest import Log, Progress, frame_path, frame_ranges, no_progress

# Keyed by Settings.caption_style (VS_CAPTION_STYLE).
PROMPTS = {
    "film": (
        "These are {n} frames, in order, from one continuous shot of a film. "
        "In one or two plain sentences, describe what happens: who is visible, what they are doing, "
        "the setting, and any movement or action (running, falling, chasing, fighting, etc.). "
        "Be concrete and visual. Do not guess character or actor names. Do not mention 'frames'."
    ),
    "driving": (
        "These are {n} frames, in order, from one continuous clip of road or traffic footage. "
        "In one or two plain sentences, describe what happens: the vehicles (type, colour) and what they do "
        "(turning, merging, braking, stopping, overtaking, cutting in, running a light, a near miss or collision), "
        "any pedestrians or cyclists, traffic signals and signs, the kind of road (intersection, highway, roundabout, "
        "parking lot), and the conditions (day or night, rain, snow, fog, glare). Say whether the camera is a dashcam "
        "or in-car view or a fixed roadside camera. Be concrete and visual. Do not speculate about who anyone is. "
        "Do not mention 'frames'."
    ),
}
MAX_IMAGES = 4


def _encode_frame(p: Path, width: int = 384) -> str:
    """Base64 JPEG, downscaled to at most `width` px wide."""
    with Image.open(p) as im:
        im = im.convert("RGB")
        if im.width > width:
            im = im.resize((width, int(im.height * width / im.width)))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def _pick_frames(f0: int, f1: int, k: int = MAX_IMAGES) -> list[int]:
    n = f1 - f0
    if n <= k:
        return list(range(f0, f1))
    return [f0 + int(round(i * (n - 1) / (k - 1))) for i in range(k)]


class Captioner:
    def __init__(self, s: Settings):
        if s.caption_style not in PROMPTS:
            raise ValueError(f"unknown VS_CAPTION_STYLE {s.caption_style!r}; known: {', '.join(PROMPTS)}")
        self.prompt = PROMPTS[s.caption_style]
        self.model = s.caption_model
        if self.model.startswith("claude-"):
            import anthropic

            self.claude = anthropic.Anthropic(api_key=s.caption_api_key or None, max_retries=6)
            return
        self.claude = None
        if not s.caption_api_key and "localhost" not in s.caption_base_url and "127.0.0.1" not in s.caption_base_url:
            raise SystemExit("Set VS_CAPTION_API_KEY (or point VS_CAPTION_BASE_URL at a local server).")
        self.url = s.caption_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {s.caption_api_key}"} if s.caption_api_key else {}
        self.client = httpx.Client(headers=headers, timeout=120)

    def caption(self, frames: list[Path]) -> str:
        if self.claude:
            return self._caption_claude(frames)
        content = [{"type": "text", "text": self.prompt.format(n=len(frames))}]
        content += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _encode_frame(p)}} for p in frames]
        body = {"model": self.model, "messages": [{"role": "user", "content": content}], "max_tokens": 120, "temperature": 0.2}
        delay = 2.0
        for attempt in range(6):
            r = self.client.post(self.url, json=body)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
            return " ".join(text.split())
        r.raise_for_status()
        raise RuntimeError(f"caption request failed: {r.status_code}")

    def _caption_claude(self, frames: list[Path]) -> str:
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _encode_frame(p)}}
                   for p in frames]
        content.append({"type": "text", "text": self.prompt.format(n=len(frames))})
        # Thinking is on by default and counts toward max_tokens; low effort keeps it short.
        msg = self.claude.messages.create(model=self.model, max_tokens=1024, output_config={"effort": "low"},
                                          messages=[{"role": "user", "content": content}])
        if msg.stop_reason == "refusal":
            raise RuntimeError(f"refused: {msg.stop_details.category if msg.stop_details else 'unknown'}")
        text = " ".join(b.text for b in msg.content if b.type == "text")
        if not text.strip():
            raise RuntimeError(f"no caption text (stop_reason={msg.stop_reason})")
        return " ".join(text.split())


def caption_film(film_dir: Path, s: Settings, *, workers: int = 8, captioner: Captioner | None = None,
                 text_embedder=None, log: Log = print, progress: Progress = no_progress) -> int:
    meta = json.loads((film_dir / "film.json").read_text())
    fid = meta["id"]
    shots = json.loads((film_dir / "shots.json").read_text())
    times = np.load(film_dir / "frame_times.npy")
    chunks = frame_ranges(shots, times, s.caption_chunk_sec, s.caption_chunk_sec, meta["sample_fps"], sliding=False)

    done_log = film_dir / "captions.jsonl"
    done: dict[int, str] = {}
    if done_log.exists():
        for line in done_log.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["i"]] = rec["text"]
    todo = [i for i in range(len(chunks)) if i not in done]
    log(f"[{fid}] {len(chunks)} caption chunks, {len(todo)} to go")

    progress("captioning", 0, len(todo))
    if todo:
        captioner = captioner or Captioner(s)
        lock = threading.Lock()
        with done_log.open("a") as fh, ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(captioner.caption, [frame_path(film_dir, f) for f in _pick_frames(chunks[i][3], chunks[i][4])]): i
                for i in todo
            }
            for k, fut in enumerate(as_completed(futs), 1):
                i = futs[fut]
                progress("captioning", k, len(futs))
                try:
                    text = fut.result()
                except Exception as e:  # keep going; rerun the command to retry failures
                    log(f"[{fid}]   chunk {i} failed: {e}")
                    continue
                with lock:
                    done[i] = text
                    fh.write(json.dumps({"i": i, "text": text}) + "\n")
                    fh.flush()
                if k % 25 == 0 or k == len(futs):
                    log(f"[{fid}]   {k}/{len(futs)}")

    rows = [chunks[i] + [done[i]] for i in range(len(chunks)) if done.get(i)]
    if not rows:
        log(f"[{fid}] no captions written")
        return 0
    if text_embedder is None:
        from .embed import text_embedder as get_te
        text_embedder = get_te(s)
    emb = text_embedder.encode_passages([r[5] for r in rows]).astype(np.float16)
    (film_dir / "captions.json").write_text(json.dumps(rows))
    np.save(film_dir / "emb_captions.npy", emb)
    meta["text_model"] = text_embedder.name
    meta["n_captions"] = len(rows)
    (film_dir / "film.json").write_text(json.dumps(meta, indent=2))
    missing = len(chunks) - len(rows)
    log(f"[{fid}] {len(rows)} captions embedded" + (f" ({missing} failed, rerun to retry)" if missing else ""))
    return len(rows)
