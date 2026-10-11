"""Run vidsearch on Modal: the library lives on a Volume, the API + frontend are one web endpoint,
and new clips are ingested and captioned on a GPU without touching this machine.

    modal deploy modal_app.py                                        # API + app at https://<workspace>--vidsearch-traffic-web.modal.run
    modal run modal_app.py::add_film --url <video url> --name "Junction at dusk"  # download, ingest, caption (public)
    modal run modal_app.py::relocate_sources                         # after `modal volume put` from a Mac

Names follow the collection (vidsearch.config.COLLECTIONS; VIDSEARCH_COLLECTION, default traffic): app and Volume
"vidsearch-<collection>", web label "vidsearch-<collection>-web", Dict "vidsearch-<collection>-jobs".
The Volume holds /videos/<name>.mp4 and /data/films/<id>/... exactly as the local layout,
plus /data/uploads/<job id>/ for the parts of in-flight browser uploads.
Captions use the "vidsearch-anthropic" secret (ANTHROPIC_API_KEY, VS_CAPTION_MODEL):
    modal secret create vidsearch-anthropic --from-dotenv .env
The Web class uses the "vidsearch-auth0" secret: VS_AUTH0_DOMAIN (absent = no sign-in), optionally
VS_AUTH0_AUDIENCE, and VS_IP_SALT (random, keeps per-IP upload counts stable across container restarts):
    modal secret create vidsearch-auth0 VS_IP_SALT=$(openssl rand -hex 16)
Uploads without sign-in (VS_ANON_UPLOADS=1, set below) tie each upload to a random id the browser keeps.
Before relying on the per-IP limit, check what Modal's proxy puts in request.client.host and X-Forwarded-For,
and set VS_CLIENT_IP_HEADER if the client's address is only in a header (README, "Accounts and uploads").

Uploads: the web container stores parts on the Volume, assembles them, commits, and spawns
`process_job` on a GPU. Job records live in the jobs Dict; the web container reads
them to report progress and reloads the library when it sees a finished job.
"""

from __future__ import annotations

from pathlib import Path

import modal

from vidsearch.config import collection_name

ROOT = Path(__file__).parent
VOL = "/vol"
VIDEOS = f"{VOL}/videos"

COLLECTION = collection_name()
NAME = f"vidsearch-{COLLECTION}"
WEB_LABEL = f"{NAME}-web"

volume = modal.Volume.from_name(NAME, create_if_missing=True)
jobs_dict = modal.Dict.from_name(f"{NAME}-jobs", create_if_missing=True)


def _bake_models() -> None:
    from vidsearch.config import Settings
    from vidsearch.embed import image_text_embedder, text_embedder

    s = Settings()
    image_text_embedder(s)
    text_embedder(s)


image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("ffmpeg", "libglib2.0-0")
    .uv_pip_install(requirements=[str(ROOT / "requirements.txt")])
    .env({"VS_DATA": f"{VOL}/data", "VS_VIDEOS": VIDEOS, "VS_WEB_DIST": "/web", "HF_HOME": "/models"})
    .add_local_python_source("vidsearch", copy=True)
    .run_function(_bake_models)
    # Containers re-import this module, so they need the collection too, both for the names above and for
    # Settings' per-collection defaults. After the model bake, so every collection shares that layer.
    .env({"VIDSEARCH_COLLECTION": COLLECTION})
    .add_local_dir(ROOT / "web" / "dist", "/web")
)

app = modal.App(NAME, image=image)


# Query encoding (3 SigLIP text passes) measured 435 ms on 1 core, 77 ms on 4 with torch pinned to 4 threads.
WEB_CPUS = 4


def _pull() -> None:
    # Commit first: reload() fails while this container has uncommitted writes (upload parts).
    volume.commit()
    volume.reload()


class ModalRunner:
    def __init__(self):
        from vidsearch.jobs import JobStore

        self.store = JobStore(jobs_dict)

    def start(self, job_id: str) -> None:
        process_job.spawn(job_id)


# The host reports 20 cores; pin torch's pools to the 4 reserved so they don't oversubscribe.
# max_containers=1: an upload's parts arrive as separate requests and are assembled from this
# container's view of the Volume; a second container wouldn't see parts it hasn't committed and
# reloaded. One container also makes it the only writer of the quota buckets (per user, per IP, global).
@app.cls(volumes={VOL: volume}, cpu=float(WEB_CPUS), memory=4096, scaledown_window=600, timeout=600,
         max_containers=1, enable_memory_snapshot=True,
         env={"OMP_NUM_THREADS": str(WEB_CPUS), "MKL_NUM_THREADS": str(WEB_CPUS), "VS_ANON_UPLOADS": "1"},
         secrets=[modal.Secret.from_name("vidsearch-auth0")])
@modal.concurrent(max_inputs=32)
class Web:
    # Importing torch and loading both models takes ~30 s on one core; the snapshot restores them already loaded.
    @modal.enter(snap=True)
    def load_models(self) -> None:
        _bake_models()

    @modal.asgi_app(label=WEB_LABEL)
    def serve(self):
        from vidsearch.server import create_app

        return create_app(before_reload=_pull, after_write=volume.commit, runner=ModalRunner())


def _run(job_id: str, caption: bool = True) -> None:
    from vidsearch.config import Settings
    from vidsearch.jobs import JobStore, run_job

    volume.reload()
    run_job(job_id, Settings(), JobStore(jobs_dict), caption=caption, sync=volume.commit)


@app.function(volumes={VOL: volume}, gpu="L4", timeout=3600,
              secrets=[modal.Secret.from_name("vidsearch-anthropic")])
def process_job(job_id: str) -> None:
    _run(job_id)


@app.function(volumes={VOL: volume}, gpu="L4", timeout=3600,
              secrets=[modal.Secret.from_name("vidsearch-anthropic")])
def add_film(url: str, name: str, caption: bool = True) -> str:
    """Admin ingest: same job body as uploads, no owner, so the film is public."""
    from vidsearch.jobs import JobStore, new_job, url_ext

    job = new_job(None, name, "url", url_ext(url), url=url)
    JobStore(jobs_dict).put(job)
    _run(job.id, caption)
    done = JobStore(jobs_dict).get(job.id)
    if done.state == "failed":
        raise RuntimeError(done.error)
    if done.error:
        print(done.error)
    return done.film_id


@app.function(volumes={VOL: volume})
def relocate_sources() -> list[str]:
    """Point each film.json's source at /vol/videos/<file name> (uploads carry the uploader's local paths)."""
    import json

    moved = []
    for mp in sorted(Path(f"{VOL}/data/films").glob("*/film.json")):
        meta = json.loads(mp.read_text())
        target = Path(VIDEOS) / Path(meta["source"]).name
        if meta["source"] != str(target) and target.exists():
            meta["source"] = str(target)
            mp.write_text(json.dumps(meta, indent=2))
            moved.append(meta["id"])
    volume.commit()
    return moved
