"""Turn one video file into searchable, renderable artifacts.

Output for each film lives in  <data>/films/<film_id>/ :

    film.json          metadata (written last; its presence means "ingest finished")
    frames/000000.jpg  sampled frames at VS_SAMPLE_FPS, VS_FRAME_WIDTH wide
    atlas/000.jpg      thumbnail texture pages, row-major cells, for the 3D view
    frame_times.npy    float32 timestamp (s) of each sampled frame
    shots.json         [[start, end], ...] shot boundaries in seconds
    frame_shot.npy     int32 shot index of each frame
    emb_frames.npy     float16 image embeddings, one per frame (L2-normalised)
    windows.json       [[start, end, shot, f0, f1], ...] sliding windows inside shots
    emb_windows.npy    float16 mean-pooled window embeddings
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .config import Settings

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".ogv"}

Log = Callable[[str], None]
# Structured progress for job tracking: (stage, done, total), e.g. ("embedding", 640, 3600).
Progress = Callable[[str, int, int], None]


def no_progress(stage: str, done: int, total: int) -> None:
    pass


# ---------------------------------------------------------------- helpers


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return s or "film"


def pretty_title(stem: str) -> str:
    t = re.sub(r"[_.\-]+", " ", stem).strip()
    return re.sub(r"\s+", " ", t).title() if t.islower() or t.isupper() else t


def find_videos(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXTS)
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(p)
    return out


def film_id_for(src: Path, films_dir: Path) -> str:
    """Slug of the filename; disambiguated with a path hash if another file already owns it."""
    base = slugify(src.stem)
    meta = films_dir / base / "film.json"
    if meta.exists():
        owner = json.loads(meta.read_text()).get("source")
        if owner and Path(owner) != src.resolve():
            return f"{base}-{hashlib.sha1(str(src.resolve()).encode()).hexdigest()[:6]}"
    return base


def probe(src: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height:format=duration", "-of", "json", str(src)],
        check=True, capture_output=True, text=True,
    ).stdout
    info = json.loads(out)
    stream = info["streams"][0]
    return {"duration": float(info["format"]["duration"]), "width": int(stream["width"]), "height": int(stream["height"])}


def is_grayscale(src: Path, samples: int = 24) -> bool:
    """True when sampled frames carry no colour (median HSV saturation under 1/255)."""
    import cv2

    cap = cv2.VideoCapture(str(src))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sats = []
    for i in np.linspace(0, max(n - 1, 0), samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            sats.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 1].mean()))
    cap.release()
    return bool(sats) and float(np.median(sats)) < 1.0


def detect_shots(src: Path, duration: float, detector: str = "content", threshold: str = "") -> list[list[float]]:
    """Shot boundaries via PySceneDetect."""
    from scenedetect import AdaptiveDetector, ContentDetector, detect

    if detector == "adaptive":
        det = AdaptiveDetector(adaptive_threshold=float(threshold)) if threshold else AdaptiveDetector()
    elif detector == "content":
        # 18 rather than PySceneDetect's 27: low-contrast silent-era film hides many cuts at 27.
        # On a grayscale print hue and saturation never change, so the default score (the mean of
        # hue, sat and luma deltas) shrinks to a third and cuts slip under any sane threshold.
        det = ContentDetector(threshold=float(threshold or 18), luma_only=is_grayscale(src))
    else:
        raise ValueError(f"unknown shot detector {detector!r} (use content or adaptive)")
    scenes = detect(str(src), det, start_in_scene=True)

    def sec(tc) -> float:
        return float(tc.seconds) if hasattr(tc, "seconds") else tc.get_seconds()

    shots = [[sec(s), sec(e)] for s, e in scenes]
    if not shots:
        shots = [[0.0, duration]]
    shots[0][0] = 0.0
    shots[-1][1] = max(shots[-1][1], duration)
    return shots


def extract_frames(src: Path, out_dir: Path, fps: float, width: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-vf", f"fps={fps},scale={width}:-2", "-q:v", "3", "-start_number", "0",
         str(out_dir / "%06d.jpg")],
        check=True,
    )
    return len(list(out_dir.glob("*.jpg")))


def frame_path(film_dir: Path, i: int) -> Path:
    return film_dir / "frames" / f"{i:06d}.jpg"


def build_atlas(film_dir: Path, n_frames: int, thumb_w: int, max_px: int) -> dict:
    """Pack frame thumbnails into texture pages (row-major) for instanced rendering."""
    with Image.open(frame_path(film_dir, 0)) as im:
        w, h = im.size
    cell_w = thumb_w
    cell_h = max(2, int(round(thumb_w * h / w / 2)) * 2)
    cols, rows = max_px // cell_w, max_px // cell_h
    per_page = cols * rows
    pages = math.ceil(n_frames / per_page)

    atlas_dir = film_dir / "atlas"
    atlas_dir.mkdir(exist_ok=True)
    for p in range(pages):
        count = min(per_page, n_frames - p * per_page)
        used_rows = math.ceil(count / cols)
        sheet = Image.new("RGB", (cols * cell_w, used_rows * cell_h))
        for k in range(count):
            with Image.open(frame_path(film_dir, p * per_page + k)) as im:
                sheet.paste(im.convert("RGB").resize((cell_w, cell_h), Image.BILINEAR),
                            ((k % cols) * cell_w, (k // cols) * cell_h))
        sheet.save(atlas_dir / f"{p:03d}.jpg", quality=82)
    return {"cell_w": cell_w, "cell_h": cell_h, "cols": cols, "rows": rows, "per_page": per_page, "pages": pages}


def frame_ranges(shots: list[list[float]], times: np.ndarray, length_sec: float, stride_sec: float,
                 fps: float, sliding: bool = True) -> list[list]:
    """Split each shot into frame ranges. Returns [[start, end, shot, f0, f1], ...] (f1 exclusive).

    sliding=True  -> overlapping windows of length_sec every stride_sec (search moments)
    sliding=False -> back-to-back chunks of at most length_sec (caption units)
    """
    wf = max(1, int(round(length_sec * fps)))
    sf = max(1, int(round(stride_sec * fps))) if sliding else wf
    out = []
    for si, (s, e) in enumerate(shots):
        f_lo = int(np.searchsorted(times, s, side="left"))
        f_hi = int(np.searchsorted(times, e, side="left"))
        if f_hi <= f_lo:
            continue  # shot shorter than the sampling interval
        if f_hi - f_lo <= wf:
            starts = [f_lo]
        else:
            starts = list(range(f_lo, f_hi - wf + 1, sf))
            if starts[-1] + wf < f_hi:
                starts.append(f_hi - wf if sliding else starts[-1] + sf)
        for f0 in starts:
            f1 = min(f0 + wf, f_hi)
            t0 = max(s, float(times[f0]))
            t1 = min(e, float(times[f1 - 1]) + 1.0 / fps)
            out.append([round(t0, 3), round(t1, 3), si, f0, f1])
    return out


def pool(emb: np.ndarray, ranges: list[list]) -> np.ndarray:
    """Mean-pool frame embeddings over each [.., f0, f1] range and re-normalise."""
    e = emb.astype(np.float32)
    pooled = np.stack([e[r[3]:r[4]].mean(axis=0) for r in ranges]) if ranges else np.zeros((0, e.shape[1]), np.float32)
    return pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)


# ---------------------------------------------------------------- main entry


def ingest_video(src: Path, s: Settings, *, title: str | None = None, force: bool = False,
                 embedder=None, owner: str | None = None, log: Log = print,
                 progress: Progress = no_progress) -> str:
    """owner: user id that may see this film; None = public."""
    src = src.resolve()
    s.films_dir.mkdir(parents=True, exist_ok=True)
    fid = film_id_for(src, s.films_dir)
    film_dir = s.films_dir / fid
    if (film_dir / "film.json").exists():
        if not force:
            log(f"[{fid}] already ingested, skipping (use --force to redo)")
            return fid
        # A re-ingest (e.g. after changing the image model) must not make a private film public.
        owner = owner or json.loads((film_dir / "film.json").read_text()).get("owner")
    if film_dir.exists():
        shutil.rmtree(film_dir)
    film_dir.mkdir(parents=True)

    progress("probing", 0, 1)
    info = probe(src)
    log(f"[{fid}] {info['duration']/60:.1f} min, {info['width']}x{info['height']}")

    log(f"[{fid}] detecting shots ({s.shot_detector})...")
    progress("detecting shots", 0, 1)
    shots = detect_shots(src, info["duration"], s.shot_detector, s.shot_threshold)
    (film_dir / "shots.json").write_text(json.dumps([[round(a, 3), round(b, 3)] for a, b in shots]))
    log(f"[{fid}] {len(shots)} shots")

    log(f"[{fid}] sampling frames at {s.sample_fps} fps...")
    progress("sampling frames", 0, 1)
    n = extract_frames(src, film_dir / "frames", s.sample_fps, s.frame_width)
    if n == 0:
        raise RuntimeError(f"no frames extracted from {src}")
    times = np.minimum(np.arange(n, dtype=np.float32) / s.sample_fps, info["duration"]).astype(np.float32)
    np.save(film_dir / "frame_times.npy", times)
    starts = np.array([a for a, _ in shots], dtype=np.float32)
    frame_shot = (np.searchsorted(starts, times, side="right") - 1).clip(0).astype(np.int32)
    np.save(film_dir / "frame_shot.npy", frame_shot)

    log(f"[{fid}] building thumbnail atlas for {n} frames...")
    progress("building atlas", 0, 1)
    atlas = build_atlas(film_dir, n, s.thumb_width, s.atlas_max_px)

    if embedder is None:
        from .embed import image_text_embedder
        embedder = image_text_embedder(s)
    log(f"[{fid}] embedding frames with {embedder.name}...")
    chunks = []
    progress("embedding", 0, n)
    for i in range(0, n, 64):
        ims = [Image.open(frame_path(film_dir, j)) for j in range(i, min(i + 64, n))]
        chunks.append(embedder.encode_images(ims))
        for im in ims:
            im.close()
        log(f"[{fid}]   {min(i + 64, n)}/{n}")
        progress("embedding", min(i + 64, n), n)
    emb = np.concatenate(chunks).astype(np.float16)
    np.save(film_dir / "emb_frames.npy", emb)

    windows = frame_ranges(shots, times, s.window_sec, s.window_stride, s.sample_fps, sliding=True)
    (film_dir / "windows.json").write_text(json.dumps(windows))
    np.save(film_dir / "emb_windows.npy", pool(emb, windows).astype(np.float16))
    log(f"[{fid}] {len(windows)} search windows")

    meta = {
        "id": fid,
        "title": title or pretty_title(src.stem),
        "source": str(src),
        **info,
        "sample_fps": s.sample_fps,
        "n_frames": n,
        "frame_width": s.frame_width,
        "atlas": atlas,
        "image_model": embedder.name,
        "embed_dim": int(emb.shape[1]),
        **({"owner": owner} if owner else {}),
    }
    (film_dir / "film.json").write_text(json.dumps(meta, indent=2))
    log(f"[{fid}] done")
    return fid
