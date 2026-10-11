"""Film media (videos, atlas pages, frame stills) on Cloudflare R2, served through presigned GET URLs.

Keys mirror the local layout:

    videos/<file>                        Settings.videos_dir/<file>
    films/<film_id>/atlas/NNN.jpg        Settings.films_dir/<film_id>/atlas/NNN.jpg
    films/<film_id>/frames/NNNNNN.jpg    Settings.films_dir/<film_id>/frames/NNNNNN.jpg

A film is served from R2 once its film.json says `"media": "r2"`, which `publish_film` writes after every
object is uploaded. Until then (no R2 configured, or not migrated yet) the API serves the files from disk.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import mimetypes
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Callable, Sequence
from urllib.parse import quote

from .config import Settings

# A presigned URL is signed at the start of its hour and stays valid one hour past that hour's end, so every
# request in the hour gets the same URL (the browser's cache keeps the bytes) and a URL handed out late in the
# hour still has at least an hour to play.
WINDOW_S = 3600
UPLOAD_THREADS = 16


def video_key(video: str) -> str:
    return f"videos/{video}"


def film_prefix(film_id: str) -> str:
    return f"films/{film_id}/"


class R2:
    """One bucket, reached through boto3 for uploads, listings and deletes. `client` replaces boto3 (tests)."""

    def __init__(self, account_id: str, access_key_id: str, secret_access_key: str, bucket: str, client=None):
        self.host = f"{account_id}.r2.cloudflarestorage.com"
        self.access_key_id = access_key_id
        self._secret = secret_access_key
        self.bucket = bucket
        if client is not None:
            self.client = client

    def __repr__(self) -> str:
        return f"R2(bucket={self.bucket!r})"

    @cached_property
    def client(self):
        import boto3
        from botocore.config import Config

        return boto3.client("s3", endpoint_url=f"https://{self.host}", region_name="auto",
                            aws_access_key_id=self.access_key_id, aws_secret_access_key=self._secret,
                            config=Config(s3={"addressing_style": "path"}, max_pool_connections=UPLOAD_THREADS * 2))

    # ------------------------------------------------------------ reads

    def presign(self, key: str, now: float | None = None) -> tuple[str, int]:
        """(presigned GET URL, seconds a client may cache the redirect to it). The URL depends only on the key
        and the hour `now` falls in."""
        now = time.time() if now is None else now
        start = int(now // WINDOW_S * WINDOW_S)
        return _presign_get(self.host, self.bucket, key, self.access_key_id, self._secret, start, 2 * WINDOW_S), \
            max(1, start + WINDOW_S - int(now))

    def sizes(self, prefix: str) -> dict[str, int]:
        """Every object under prefix: key -> size in bytes."""
        out, token = {}, None
        while True:
            page = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix,
                                               **({"ContinuationToken": token} if token else {}))
            out.update({o["Key"]: o["Size"] for o in page.get("Contents", [])})
            if not page.get("IsTruncated"):
                return out
            token = page["NextContinuationToken"]

    # ------------------------------------------------------------ writes

    def upload_missing(self, files: dict[str, Path], progress: Callable[[int, int], None] = lambda d, t: None) -> int:
        """Upload each key's file unless an object of the same size is already there. Returns how many were uploaded."""
        have: dict[str, int] = {}
        for prefix in {k.rsplit("/", 1)[0] + "/" for k in files}:
            have.update(self.sizes(prefix))
        todo = [(k, p) for k, p in files.items() if have.get(k) != p.stat().st_size]
        done = len(files) - len(todo)
        progress(done, len(files))

        def put(item: tuple[str, Path]) -> None:
            key, path = item
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.client.upload_file(str(path), self.bucket, key, ExtraArgs={"ContentType": ctype})

        with ThreadPoolExecutor(UPLOAD_THREADS) as pool:
            for _ in pool.map(put, todo):
                done += 1
                progress(done, len(files))
        return len(todo)

    def delete(self, keys: Sequence[str] = (), prefixes: Sequence[str] = ()) -> None:
        """Delete these keys and everything under these prefixes. Absent objects are fine."""
        keys = [*keys, *(k for p in prefixes for k in self.sizes(p))]
        for i in range(0, len(keys), 1000):
            self.client.delete_objects(Bucket=self.bucket,
                                       Delete={"Objects": [{"Key": k} for k in keys[i:i + 1000]], "Quiet": True})


def r2_store(s: Settings) -> R2 | None:
    """The configured bucket, or None when the R2 settings are absent (media is served from disk)."""
    if not (s.r2_account_id and s.r2_access_key_id and s.r2_secret_access_key and s.r2_bucket):
        return None
    return R2(s.r2_account_id, s.r2_access_key_id, s.r2_secret_access_key, s.r2_bucket)


def on_r2(meta: dict) -> bool:
    return meta.get("media") == "r2"


def film_files(film_dir: Path) -> dict[str, Path]:
    """Every object a film is served from: key -> local file."""
    meta = json.loads((film_dir / "film.json").read_text())
    video = Path(meta["source"])
    files = {video_key(video.name): video}
    for kind in ("atlas", "frames"):
        files.update({f"{film_prefix(meta['id'])}{kind}/{p.name}": p for p in sorted((film_dir / kind).glob("*.jpg"))})
    return files


def publish_film(r2: R2, film_dir: Path, progress: Callable[[int, int], None] = lambda d, t: None) -> int:
    """Upload a film's media that isn't in R2 yet, then mark film.json so the API serves it from there.
    Safe to repeat. Returns how many objects were uploaded."""
    uploaded = r2.upload_missing(film_files(film_dir), progress)
    meta_path = film_dir / "film.json"
    meta = json.loads(meta_path.read_text())
    if not on_r2(meta):
        tmp = meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({**meta, "media": "r2"}, indent=2))
        tmp.replace(meta_path)
    return uploaded


def delete_film(r2: R2, film_id: str, video: str) -> None:
    r2.delete(keys=[video_key(video)], prefixes=[film_prefix(film_id)])


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def _presign_get(host: str, bucket: str, key: str, access_key_id: str, secret: str, signed_at: int,
                 expires: int) -> str:
    """AWS SigV4 query-string signature for GET https://host/bucket/key, signed at a fixed time (boto3 always
    signs at the current time, which would give every request a new URL and defeat the browser cache)."""
    t = datetime.fromtimestamp(signed_at, timezone.utc)
    amz_date, day = t.strftime("%Y%m%dT%H%M%SZ"), t.strftime("%Y%m%d")
    scope = f"{day}/auto/s3/aws4_request"
    path = "/" + quote(f"{bucket}/{key}", safe="/~")
    query = "&".join(f"{quote(k, safe='~')}={quote(v, safe='~')}" for k, v in sorted({
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{access_key_id}/{scope}",
        "X-Amz-Date": amz_date,
        "X-Amz-Expires": str(expires),
        "X-Amz-SignedHeaders": "host",
    }.items()))
    canonical = f"GET\n{path}\n{query}\nhost:{host}\n\nhost\nUNSIGNED-PAYLOAD"
    to_sign = f"AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{hashlib.sha256(canonical.encode()).hexdigest()}"
    k = _hmac(_hmac(_hmac(_hmac(f"AWS4{secret}".encode(), day), "auto"), "s3"), "aws4_request")
    return f"https://{host}{path}?{query}&X-Amz-Signature={hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()}"
