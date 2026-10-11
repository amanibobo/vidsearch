"""Upload jobs: a user's video goes from bytes (or a URL) to a searchable film.

One record per job, advanced only through `advance`, which enforces the lifecycle:

    uploading ─► queued ─► processing ─► done
        └──────────┴───────────┴───────► failed

A `queued` or `processing` job whose record hasn't changed for STALE_AFTER_S (its worker died: a
crash, a container lost, the GPU function's timeout) reads back as failed, "processing timed out".

`uploading` exists only for browser uploads (the job is created when the upload starts, so the
daily quotas count uploads at creation; `total` is then the number of parts). Cancelling an upload
fails it with error "cancelled" and gives its quota slots back. While processing, done/total count
units of the current `stage`.

The job body (`run_job`) runs in two places behind the `Runner` interface: a background thread
here (`LocalRunner`), or a GPU function on Modal (see modal_app.py). Records live in a key-value
mapping: a dict locally, a modal.Dict on Modal. After `start`, only the job body writes a job's
record; the API only reads it.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import secrets
import shutil
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, MutableMapping, Protocol, Sequence
from urllib.parse import urlsplit

import httpx

from .config import Settings
from .ingest import VIDEO_EXTS, Log, probe, slugify

State = Literal["uploading", "queued", "processing", "done", "failed"]
NEXT: dict[str, set[str]] = {
    "uploading": {"queued", "failed"},
    "queued": {"processing", "failed"},
    "processing": {"processing", "done", "failed"},
    "done": set(),
    "failed": set(),
}
# process_job's Modal timeout is 60 min, so a record untouched for 70 min has no live worker behind it
# (the margin covers waiting for a GPU between queued and processing).
STALE_AFTER_S = 70 * 60
TIMED_OUT = "processing timed out"
CANCELLED = "cancelled"


@dataclass(frozen=True)
class Job:
    id: str
    owner: str | None          # Auth0 sub or device:<hex>; None = admin job, film is public
    title: str
    film_id: str
    source: Literal["file", "url"]
    video: str                 # file name under Settings.videos_dir
    state: State
    stage: str | None = None   # downloading, probing, converting, detecting shots, ..., captioning
    done: int = 0
    total: int = 0
    error: str | None = None
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)  # last state or progress write
    url: str | None = None     # source == "url"
    size: int | None = None    # source == "file": declared byte size
    quota: tuple[str, ...] = ()  # the quota buckets this job holds a slot in; internal, never sent to clients


def advance(job: Job, state: State, **changes) -> Job:
    if state not in NEXT[job.state]:
        raise ValueError(f"job {job.id}: {job.state} -> {state} is not a valid transition")
    return replace(job, state=state, updated=time.time(), **changes)


class JobError(Exception):
    """A failure whose message is meant for the uploader."""


class QuotaExceeded(Exception):
    pass


def new_job(owner: str | None, title: str, source: Literal["file", "url"], ext: str, *,
            url: str | None = None, size: int | None = None, part_size: int = 0) -> Job:
    """A fresh job. Owned films get an unguessable id, since media routes are unauthenticated.
    Admin jobs (no owner) keep the old add_film naming, <title><ext>, so existing videos are reused."""
    slug = slugify(title)[:40].strip("-") or "film"
    film_id = f"{slug}-{secrets.token_hex(6)}" if owner else slugify(title)
    ext = ext.lower() if ext.lower() in VIDEO_EXTS else ".mp4"
    state: State = "uploading" if source == "file" else "queued"
    return Job(id=secrets.token_hex(12), owner=owner, title=title, film_id=film_id, source=source,
               video=(film_id if owner else title) + ext, state=state, url=url, size=size,
               total=math.ceil(size / part_size) if size else 0)


@dataclass(frozen=True)
class Quota:
    """One daily limit: `key` names a bucket of the job ids that count against it today."""
    key: str
    limit: int
    refusal: str  # the 429 message, naming the limit


def _today(now: float) -> str:
    return datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d")


def user_bucket(owner: str, now: float) -> str:
    return f"quota:{_today(now)}:user:{owner}"


def hash_ip(ip: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}\0{ip}".encode()).hexdigest()


def upload_quotas(s: Settings, owner: str, ip_hash: str, now: float) -> list[Quota]:
    """Every limit a new upload counts against, most specific first, so a refusal names the nearest one."""
    day, retry = _today(now), "try again after 00:00 UTC"
    return [
        Quota(user_bucket(owner, now), s.uploads_per_day,
              f"daily limit of {s.uploads_per_day} uploads reached; {retry}"),
        Quota(f"quota:{day}:ip:{ip_hash}", s.uploads_per_ip_per_day,
              f"daily limit of {s.uploads_per_ip_per_day} uploads from this network reached; {retry}"),
        Quota(f"quota:{day}:all", s.uploads_global_per_day,
              f"the archive's daily limit of {s.uploads_global_per_day} uploads across everyone is reached; {retry}"),
    ]


class JobStore:
    """Job records in a key-value mapping (dict or modal.Dict).

        "job:<id>"               -> the job as a dict
        "owner:<sub>"            -> that user's job ids, oldest first
        "quota:<utc day>:<who>"  -> ids of that day's uploads counting against one limit (see upload_quotas)
    """

    def __init__(self, kv: MutableMapping | None = None, stale_after: float = STALE_AFTER_S):
        self.kv = {} if kv is None else kv
        self.stale_after = stale_after
        self._quota_lock = threading.Lock()

    def get(self, job_id: str) -> Job | None:
        rec = self.kv.get(f"job:{job_id}")
        if not rec:
            return None
        job = Job(**{"updated": rec["created"], **rec})  # records written before `updated` existed
        if job.state in ("queued", "processing") and time.time() - job.updated > self.stale_after:
            return advance(job, "failed", error=TIMED_OUT)
        return job

    def put(self, job: Job) -> None:
        self.kv[f"job:{job.id}"] = asdict(job)

    def for_owner(self, owner: str, limit: int = 20) -> list[Job]:
        """The owner's most recent jobs, newest first."""
        ids = self.kv.get(f"owner:{owner}", [])[-limit:]
        return [j for j in (self.get(i) for i in reversed(ids)) if j]

    def count(self, bucket: str) -> int:
        return len(self.kv.get(bucket, []))

    def create(self, job: Job, quotas: Sequence[Quota]) -> Job:
        """Record a new job, taking a slot in every quota, or raise QuotaExceeded naming the first one full.
        Checks and writes share a lock; on Modal the web class runs one container, so this is the only
        writer of the owner index and the quota buckets."""
        with self._quota_lock:
            for q in quotas:
                if self.count(q.key) >= q.limit:
                    raise QuotaExceeded(q.refusal)
            job = replace(job, quota=tuple(q.key for q in quotas))
            self.put(job)
            for key in (*job.quota, *([f"owner:{job.owner}"] if job.owner else [])):
                self.kv[key] = [*self.kv.get(key, []), job.id]
            return job

    def cancel(self, job_id: str) -> Job:
        """uploading -> failed "cancelled", giving its quota slots back. Safe to repeat, also after a crash
        between the two writes. ValueError when the upload already finished."""
        with self._quota_lock:
            job = self.get(job_id)
            if not (job.state == "failed" and job.error == CANCELLED):
                if job.state != "uploading":
                    raise ValueError(f"upload is {job.state}")
                job = advance(job, "failed", error=CANCELLED)
                self.put(job)
            for key in job.quota:
                self.kv[key] = [i for i in self.kv.get(key, []) if i != job.id]
            return job


class Runner(Protocol):
    store: JobStore

    def start(self, job_id: str) -> None:
        """Run the job body for a job already in the store, in the background."""


# ---------------------------------------------------------------- chunked uploads


def part_path(s: Settings, job_id: str, n: int) -> Path:
    return parts_dir(s, job_id) / f"{n:05d}"


def parts_dir(s: Settings, job_id: str) -> Path:
    return s.uploads_dir / job_id


def write_part(s: Settings, job_id: str, n: int, data: bytes) -> None:
    """Store part n. Re-sending a part replaces it, so the browser can retry or resume freely."""
    p = part_path(s, job_id, n)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(p)


def assemble(s: Settings, job: Job) -> None:
    """Concatenate the parts into the job's video file and drop them. Raises JobError when the
    parts don't add up to the declared upload."""
    parts = [part_path(s, job.id, n) for n in range(job.total)]
    missing = [n for n, p in enumerate(parts) if not p.exists()]
    if missing:
        raise JobError(f"missing parts: {missing[:20]}")
    got = sum(p.stat().st_size for p in parts)
    if got != job.size:
        raise JobError(f"parts add up to {got} bytes, expected {job.size}")
    dest = s.videos_dir / job.video
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with tmp.open("wb") as out:
        for p in parts:
            with p.open("rb") as fh:
                shutil.copyfileobj(fh, out, 1 << 20)
    tmp.replace(dest)
    shutil.rmtree(parts_dir(s, job.id), ignore_errors=True)


def discard(s: Settings, job: Job) -> None:
    """Delete what a failed or cancelled job left on disk: upload parts, and for a user's job its video and
    film dir (an admin's video may predate the job, so it stays). Safe to repeat."""
    shutil.rmtree(parts_dir(s, job.id), ignore_errors=True)
    if job.owner:
        (s.videos_dir / job.video).unlink(missing_ok=True)
        shutil.rmtree(s.films_dir / job.film_id, ignore_errors=True)


# ---------------------------------------------------------------- the job body


# Wikimedia (and others) throttle anonymous clients; identify the project as their policies ask.
USER_AGENT = "vidsearch/1.0 (+https://github.com/amanibobo/vidsearch)"


def _refuse_private_hosts(request: httpx.Request) -> None:
    # Runs for the first request and every redirect, so a public URL can't bounce to internal services.
    host = request.url.host
    try:
        addrs = {info[4][0] for info in socket.getaddrinfo(host, None)}
    except socket.gaierror as e:
        raise JobError(f"can't resolve {host}: {e}") from None
    if not all(ipaddress.ip_address(a.split("%")[0]).is_global for a in addrs):
        raise JobError(f"refusing to download from a private address ({host})")


def download(url: str, dest: Path, max_bytes: int, progress: Callable[[int, int], None]) -> None:
    tmp = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    too_big = JobError(f"the video is larger than the {max_bytes / 1024**3:g} GB limit")
    try:
        with httpx.Client(follow_redirects=True, timeout=120, headers={"User-Agent": USER_AGENT},
                          event_hooks={"request": [_refuse_private_hosts]}) as c, \
                c.stream("GET", url) as r:
            if r.status_code >= 400:
                raise JobError(f"download failed: HTTP {r.status_code}")
            total = int(r.headers.get("content-length") or 0)
            if total > max_bytes:
                raise too_big
            got = 0
            with tmp.open("wb") as fh:
                for chunk in r.iter_bytes(1 << 20):
                    got += len(chunk)
                    if got > max_bytes:
                        raise too_big
                    fh.write(chunk)
                    progress(got, total)
        tmp.replace(dest)
    except httpx.HTTPError as e:
        raise JobError(f"download failed: {e}") from None
    finally:
        tmp.unlink(missing_ok=True)


def check_video(path: Path, max_minutes: float) -> None:
    try:
        duration = probe(path)["duration"]
    except (subprocess.CalledProcessError, KeyError, IndexError, ValueError):
        raise JobError("not a video: ffprobe found no readable video stream") from None
    if duration > max_minutes * 60:
        raise JobError(f"the video is {duration / 60:.1f} min long; the limit is {max_minutes:g} min")


def _video_info(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,width,pix_fmt:format=format_name,duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    video = next(st for st in info["streams"] if st.get("codec_type") == "video")
    return {"codec": video.get("codec_name"), "width": int(video.get("width") or 0), "pix_fmt": video.get("pix_fmt"),
            "audio": any(st.get("codec_type") == "audio" for st in info["streams"]),
            "mp4": "mp4" in info["format"].get("format_name", ""), "duration": float(info["format"].get("duration") or 0)}


def normalize_video(path: Path, max_width: int, progress: Callable[[int, int], None] = lambda d, t: None) -> Path:
    """Re-encode to H.264 MP4 (yuv420p, at most max_width wide, no audio, moov first) unless it already is.
    Browsers then play every source the same way: Safari lacks VP8/VP9/Theora/AV1, and nothing here uses
    audio (some sources carry music under stricter licences). Returns the final path (always .mp4)."""
    info = _video_info(path)
    if (info["codec"] == "h264" and info["pix_fmt"] == "yuv420p" and info["width"] <= max_width
            and not info["audio"] and info["mp4"] and path.suffix.lower() == ".mp4"):
        return path
    out = path.with_suffix(".mp4")
    tmp = path.with_name(path.stem + ".converting.mp4")
    total = max(1, int(info["duration"]))
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
           "-vf", f"scale='min({max_width},trunc(iw/2)*2)':-2", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-progress", "pipe:1", str(tmp)]
    try:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
            for line in proc.stdout:
                if line.startswith("out_time_us=") and line.strip() != "out_time_us=N/A":
                    progress(min(total, int(line.split("=")[1]) // 1_000_000), total)
            err = proc.stderr.read()
        if proc.returncode != 0:
            raise JobError(f"could not convert the video: {err.strip()[-300:] or 'ffmpeg failed'}")
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    if out != path:
        path.unlink(missing_ok=True)
    progress(total, total)
    return out


def run_job(job_id: str, s: Settings, store: JobStore, *, caption: bool = True, embedder=None, captioner=None,
            text_embedder=None, sync: Callable[[], None] = lambda: None, log: Log = print) -> None:
    """Download (URL jobs), probe, ingest and caption one job, recording progress in the store.
    sync publishes written files to other containers (volume.commit on Modal).
    Safe to rerun: finished jobs are left alone, and ingest/captions resume from what's on disk."""
    from .caption import caption_film
    from .ingest import ingest_video

    job = store.get(job_id)
    if job is None or job.state in ("done", "failed"):
        return
    video = s.videos_dir / job.video
    last_put = 0.0

    def progress(stage: str, done: int, total: int) -> None:
        nonlocal job, last_put
        # Embedding and captioning report per batch; a modal.Dict write per batch is wasteful.
        now = time.monotonic()
        if stage == job.stage and done != total and now - last_put < 1.0:
            return
        last_put = now
        job = advance(job, "processing", stage=stage, done=done, total=total)
        store.put(job)

    try:
        if job.source == "url" and not video.exists():
            progress("downloading", 0, 0)
            download(job.url, video, s.max_upload_bytes, lambda d, t: progress("downloading", d, t))
            sync()
        progress("probing", 0, 1)
        check_video(video, s.max_upload_minutes)
        progress("converting", 0, 1)
        converted = normalize_video(video, s.max_video_width, lambda d, t: progress("converting", d, t))
        if converted != video:
            video = converted
            job = replace(job, video=converted.name)
            store.put(job)
            sync()
        fid = ingest_video(video, s, title=job.title, owner=job.owner, embedder=embedder,
                           log=log, progress=progress)
        job = replace(job, film_id=fid)
        sync()
        note = None
        if caption:
            # Captioner raises SystemExit when no caption API is configured. The film is still
            # searchable by its frames, so the job succeeds and says why captions are missing.
            try:
                caption_film(s.films_dir / fid, s, captioner=captioner, text_embedder=text_embedder,
                             log=log, progress=progress)
            except (Exception, SystemExit) as e:
                note = f"captions skipped: {e}"
                log(f"[{fid}] {note}")
            sync()
        store.put(advance(job, "done", stage=None, done=0, total=0, error=note))
    except Exception as e:
        msg = str(e) if isinstance(e, JobError) else f"processing failed: {type(e).__name__}: {e}"
        log(f"[job {job.id}] {msg}")
        discard(s, job)
        sync()
        store.put(advance(job, "failed", error=msg))


class LocalRunner:
    """Runs job bodies on one background thread in this process (ingest saturates the machine)."""

    def __init__(self, s: Settings, store: JobStore | None = None, **job_kwargs):
        self.s = s
        self.store = store or JobStore()
        self.job_kwargs = job_kwargs
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vidsearch-job")

    def start(self, job_id: str) -> None:
        self.pool.submit(run_job, job_id, self.s, self.store, **self.job_kwargs)


def url_ext(url: str) -> str:
    """File extension of an http(s) video URL's path. JobError for anything that isn't http(s)."""
    u = urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise JobError("url must be an http(s) link to a video file")
    return Path(u.path).suffix
