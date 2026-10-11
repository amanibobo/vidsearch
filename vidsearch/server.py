"""FastAPI server for the frontend.

    GET    /api/library                 films the caller can see, shot boundaries, atlas layout + urls,
                                        and `uploads`: {mode: off | device | account, limits}
    GET    /api/search?q=...&k=24       ranked moments (film, timestamp, best frame, atlas cell...)
    GET    /api/heatmap?q=...           per-frame relevance for every visible film (glowing strip ticks)
    GET    /api/video/{film_id}         the source video (supports Range, so <video> can seek)
    POST   /api/reload                  pick up newly ingested / captioned films
    GET    /media/{film_id}/...         frames/*.jpg and atlas/*.jpg

  signed in (Authorization: Bearer <Auth0 access token>, or device:<32 hex> with VS_ANON_UPLOADS=1):
    POST   /api/uploads                 {title, filename, size} -> {upload_id, part_size}
    PUT    /api/uploads/{id}/parts/{n}  raw bytes of part n (0-based); re-sending replaces it
    POST   /api/uploads/{id}/complete   {parts} -> {job_id}
    DELETE /api/uploads/{id}            cancel an unfinished upload: drops its parts, gives its quota slots back
    POST   /api/uploads/url             {title, url} -> {job_id}
    GET    /api/jobs                    the caller's recent jobs, uploads used today, limits
    GET    /api/jobs/{id}               one job
    DELETE /api/films/{film_id}         remove one of the caller's films

Run:  python -m vidsearch serve   (or)   uvicorn vidsearch.server:app --reload
"""

from __future__ import annotations

import logging
import secrets
import shutil
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .auth import uploads_mode, user_dependencies
from .config import Settings
from .jobs import (TIMED_OUT, Job, JobError, LocalRunner, QuotaExceeded, Runner, advance, assemble, discard, hash_ip,
                   new_job, parts_dir, upload_quotas, url_ext, user_bucket, write_part)
from .library import Library

settings = Settings()
log = logging.getLogger("vidsearch.server")


class NewUpload(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    filename: str = Field(min_length=1, max_length=500)
    size: int = Field(gt=0)


class CompleteUpload(BaseModel):
    parts: int = Field(gt=0)


class UrlUpload(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    url: str = Field(min_length=1, max_length=4000)


def create_app(lib: Library | None = None, s: Settings | None = None,
               before_reload: Callable[[], None] | None = None,
               after_write: Callable[[], None] | None = None,
               runner: Runner | None = None) -> FastAPI:
    """before_reload runs ahead of every library reload, e.g. to pull a remote volume's latest commit.
    after_write runs after the API writes files other processes need (assembled uploads, deletes),
    e.g. a volume commit. runner runs upload jobs; default is a background thread in this process."""
    s = s or settings
    runner = runner or LocalRunner(s)
    jobs = runner.store
    optional_user, required_user = user_dependencies(s)
    reload_lock = threading.Lock()
    upload_lock = threading.Lock()  # completing and cancelling an upload must not interleave
    picked_up: set[str] = set()  # finished jobs whose film this process has already loaded
    reaped: set[str] = set()     # timed-out jobs whose files this process has already deleted
    state: dict = {}  # per app, so two apps in one process (tests) don't share a library
    ip_salt = s.ip_salt
    if not ip_salt:
        ip_salt = secrets.token_hex(16)
        if s.anon_uploads:
            log.warning("VS_IP_SALT is empty: per-IP upload counts use a random salt and reset when this process restarts")
    limits = {"uploads_per_day": s.uploads_per_day, "uploads_per_ip_per_day": s.uploads_per_ip_per_day,
              "uploads_global_per_day": s.uploads_global_per_day, "max_bytes": s.max_upload_bytes,
              "max_minutes": s.max_upload_minutes, "part_size": s.upload_part_bytes}
    uploads = {"mode": uploads_mode(s), "limits": limits}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["lib"] = lib or Library(s)
        state["lib"].warm()
        yield

    app = FastAPI(title="vidsearch", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def L() -> Library:
        return state["lib"]

    def reload_library() -> None:
        with reload_lock:
            if before_reload:
                before_reload()
            L().reload()

    def observe(seen: list[Job]) -> None:
        """Act once per process on outcomes no worker will act on. A finished job's film isn't loaded: reload,
        so new films appear without /api/reload. A job timed out: its worker is gone, so delete what it left."""
        done = [j for j in seen if j.state == "done" and j.id not in picked_up]
        dead = [j for j in seen if j.state == "failed" and j.error == TIMED_OUT and j.id not in reaped]
        for j in dead:
            discard(s, j)
        if dead and after_write:
            after_write()
        if any(j.film_id not in L().by_id for j in done) or any(j.film_id in L().by_id for j in dead):
            reload_library()
        picked_up.update(j.id for j in done)
        reaped.update(j.id for j in dead)

    def owned_job(job_id: str, user: str) -> Job:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        if job.owner != user:
            raise HTTPException(403, "not your job")
        return job

    def client_ip(request: Request) -> str:
        forwarded = request.headers.get(s.client_ip_header, "") if s.client_ip_header else ""
        return forwarded.split(",")[-1].strip() or (request.client.host if request.client else "")

    def create_job(job: Job, user: str, request: Request) -> Job:
        try:
            return jobs.create(job, upload_quotas(s, user, hash_ip(client_ip(request), ip_salt), time.time()))
        except QuotaExceeded as e:
            raise HTTPException(429, str(e)) from None

    def job_json(job: Job) -> dict:
        out = asdict(job)
        del out["quota"]
        return out

    @app.get("/api/library")
    def library(user: str | None = Depends(optional_user)):
        if user:  # a few recent jobs is enough: each is a store read (a modal.Dict call on Modal)
            observe(jobs.for_owner(user, limit=5))
        return {**L().library_json(user), "uploads": uploads}

    @app.get("/api/search")
    def search(
        q: str = Query(..., min_length=1),
        k: int = Query(24, ge=1, le=200),
        films: str | None = Query(None, description="comma-separated film ids to restrict to"),
        gap: float | None = Query(None, description="min seconds between results from the same film"),
        use: str = Query("frame,window,caption", description="signals to fuse"),
        heatmap: bool = Query(False, description="also return /api/heatmap's payload, sharing the query encoding"),
        user: str | None = Depends(optional_user),
    ):
        t0 = time.perf_counter()
        results = L().search(
            q, k=k,
            films=[f for f in films.split(",") if f] if films else None,
            min_gap=gap,
            use=tuple(u.strip() for u in use.split(",") if u.strip()),
            user=user,
        )
        out = {"query": q, "took_ms": round((time.perf_counter() - t0) * 1000, 1), "results": results}
        if heatmap:
            out["heatmap"] = L().heatmap(q, user)
        return out

    @app.get("/api/heatmap")
    def heatmap(q: str = Query(..., min_length=1), user: str | None = Depends(optional_user)):
        return {"query": q, **L().heatmap(q, user)}

    # Media and video stay unauthenticated: <video> and WebGL texture loads can't send an
    # Authorization header. A private film is protected by its unguessable id instead.
    @app.get("/api/video/{film_id}")
    def video(film_id: str):
        lib = L()
        if film_id not in lib.by_id:
            raise HTTPException(404, "unknown film")
        src = Path(lib.films[lib.by_id[film_id]].meta["source"])
        if not src.exists():
            raise HTTPException(404, f"source video moved or missing: {src}")
        return FileResponse(src)

    @app.post("/api/reload")
    def reload():
        reload_library()
        return {"films": len(L().films)}

    @app.post("/api/uploads")
    def start_upload(body: NewUpload, request: Request, user: str = Depends(required_user)):
        if body.size > s.max_upload_bytes:
            raise HTTPException(413, f"file is larger than the {s.max_upload_bytes / 1024**3:g} GB limit")
        job = create_job(new_job(user, body.title, "file", Path(body.filename).suffix, size=body.size,
                                 part_size=s.upload_part_bytes), user, request)
        return {"upload_id": job.id, "part_size": s.upload_part_bytes, "parts": job.total}

    # async so the body can be read with a cap; blocking store and disk work goes to the threadpool.
    @app.put("/api/uploads/{upload_id}/parts/{n}")
    async def put_part(upload_id: str, n: int, request: Request, user: str = Depends(required_user)):
        job = await run_in_threadpool(owned_job, upload_id, user)
        if job.state != "uploading":
            raise HTTPException(409, f"upload is {job.state}")
        if not 0 <= n < job.total:
            raise HTTPException(400, f"part {n} is out of range 0..{job.total - 1}")
        data = bytearray()
        async for chunk in request.stream():
            data += chunk
            if len(data) > s.upload_part_bytes:
                raise HTTPException(413, f"parts are at most {s.upload_part_bytes} bytes")
        await run_in_threadpool(write_part, s, job.id, n, bytes(data))
        # A cancel that landed while this part was in flight has already deleted the parts; don't leave this one.
        now = await run_in_threadpool(jobs.get, job.id)
        if now.state != "uploading":
            await run_in_threadpool(shutil.rmtree, parts_dir(s, job.id), True)
            raise HTTPException(409, f"upload is {now.state}")
        return {"part": n, "bytes": len(data)}

    @app.post("/api/uploads/{upload_id}/complete")
    def complete_upload(upload_id: str, body: CompleteUpload, user: str = Depends(required_user)):
        with upload_lock:
            job = owned_job(upload_id, user)
            if job.state != "uploading":
                raise HTTPException(409, f"upload is {job.state}")
            if body.parts != job.total:
                raise HTTPException(400, f"this upload has {job.total} parts, not {body.parts}")
            try:
                assemble(s, job)
            except JobError as e:
                raise HTTPException(400, str(e)) from None
            jobs.put(advance(job, "queued", total=0))
        if after_write:
            after_write()  # the job may run in another container; it must see the assembled file
        runner.start(job.id)
        return {"job_id": job.id}

    @app.delete("/api/uploads/{upload_id}")
    def cancel_upload(upload_id: str, user: str = Depends(required_user)):
        with upload_lock:
            owned_job(upload_id, user)
            try:
                job = jobs.cancel(upload_id)
            except ValueError as e:
                raise HTTPException(409, str(e)) from None
            discard(s, job)
        return {"cancelled": job.id}

    @app.post("/api/uploads/url")
    def url_upload(body: UrlUpload, request: Request, user: str = Depends(required_user)):
        try:
            ext = url_ext(body.url)
        except JobError as e:
            raise HTTPException(400, str(e)) from None
        job = create_job(new_job(user, body.title, "url", ext, url=body.url), user, request)
        runner.start(job.id)
        return {"job_id": job.id}

    @app.get("/api/jobs")
    def list_jobs(user: str = Depends(required_user)):
        mine = jobs.for_owner(user)
        observe(mine)
        return {"jobs": [job_json(j) for j in mine], "uploads_today": jobs.count(user_bucket(user, time.time())),
                "limits": limits}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, user: str = Depends(required_user)):
        job = owned_job(job_id, user)
        observe([job])
        return job_json(job)

    @app.delete("/api/films/{film_id}")
    def delete_film(film_id: str, user: str = Depends(required_user)):
        lib = L()
        if film_id not in lib.by_id:
            raise HTTPException(404, "unknown film")
        film = lib.films[lib.by_id[film_id]]
        owner = film.meta.get("owner")
        if owner is None:
            raise HTTPException(403, "public films can't be deleted through the API")
        if owner != user:
            raise HTTPException(403, "not your film")
        shutil.rmtree(film.dir)
        Path(film.meta["source"]).unlink(missing_ok=True)
        if after_write:
            after_write()
        reload_library()
        return {"deleted": film_id}

    @app.middleware("http")
    async def cache_media(request, call_next):
        # Frames, atlases and videos never change after ingest (a re-ingest is rare and gets a reload anyway).
        response = await call_next(request)
        if request.url.path.startswith(("/media/", "/api/video/")) and response.status_code in (200, 206):
            response.headers.setdefault("Cache-Control", "public, max-age=86400")
        return response

    s.films_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/media", StaticFiles(directory=s.films_dir), name="media")
    if s.web_dist:
        app.mount("/", StaticFiles(directory=s.web_dist, html=True), name="web")
    return app


app = create_app()
