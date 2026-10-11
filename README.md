# vidsearch

Semantic "find the moment" search over a video library: type *being chased* and get back
timestamps across every film, plus everything a frontend needs to draw the whole archive as
filmstrips (thumbnail atlases, shot boundaries, per-frame relevance heatmaps).

```
video ─► shots (PySceneDetect) ─► frames @1fps ─► SigLIP 2 frame embeddings ─┬─► index: frames
                                      │                                      └─► index: 3s windows (mean-pooled)
                                      ├─► thumbnail atlases (for the 3D view)
                                      └─► [optional] VLM shot captions ─► bge embeddings ─► index: captions

query ─► SigLIP text + bge text ─► search all three ─► reciprocal-rank fusion ─► de-dupe in time ─► moments
```

Frame embeddings are good at *things* (a streetcar, a crowd, a bride). Captions are what make
*actions* work ("being chased", "falls off a ladder"), since one frame can't show motion.
Search fuses both, so the caption step is optional but recommended.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch            # pick the right build for your machine (CUDA / MPS / CPU)
pip install -r requirements.txt
```

Needs `ffmpeg` on your PATH. The first run downloads SigLIP 2 ViT-B/16 and bge-small from
Hugging Face.

## Quickstart

```bash
# 1. put videos in ./videos (Keaton's 1920s features are public domain on archive.org)
python -m vidsearch ingest videos/

# 2. optional, recommended: caption every shot with a vision LLM
export VS_CAPTION_API_KEY=...            # see "Caption providers" below
python -m vidsearch caption

# 3. try it
python -m vidsearch search "being chased"
python -m vidsearch search "a streetcar" -k 5

# 4. serve the API for the frontend
python -m vidsearch serve                # http://127.0.0.1:8000/docs
```

Ingest is per film and skips films already done (`--force` redoes one). Captioning is
resumable: it logs progress to `captions.jsonl`, so rerunning picks up where it stopped and
retries failures. After ingesting or captioning while the server is running, `POST /api/reload`.

Rough cost, as a ballpark: on a laptop GPU / Apple Silicon: a 60-min film is ~3,600 frames, about 1–3 min to
embed. On CPU it's more like 10–15 min. Captioning a feature film is ~400–600 chunks.

## Caption providers

Any OpenAI-compatible `/chat/completions` endpoint that accepts images works:

| Provider | `VS_CAPTION_BASE_URL` | `VS_CAPTION_MODEL` |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` (default) | `gpt-4o-mini` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-2.5-flash` |
| OpenRouter | `https://openrouter.ai/api/v1` | any vision model |
| Ollama (local, free) | `http://localhost:11434/v1` | `qwen2.5vl` |
| vLLM (local) | `http://localhost:8000/v1` | `Qwen/Qwen2.5-VL-7B-Instruct` |

Claude models go through the Claude API instead: set `ANTHROPIC_API_KEY` and
`VS_CAPTION_MODEL=claude-haiku-5-5` (any `claude-*` model name switches over; needs the `anthropic` package).

Each chunk sends up to 4 frames from one shot (long shots are split into 12 s chunks).

## API

| Route | Returns |
|---|---|
| `GET /api/library` | films the caller can see: title, duration, `n_frames`, `sample_fps`, `shots`, `atlas` layout + `urls`, `video_url`, `owner_is_you`. Also `uploads`: `{mode, limits}`, where `mode` is `off`, `device` or `account` (below) |
| `GET /api/search?q=being+chased&k=24` | ranked moments (below). Optional `films=id1,id2`, `gap=` (sec), `use=frame,window,caption` |
| `GET /api/heatmap?q=...` | per visible film, base64 `uint8` per sampled frame (0–255 relevance) |
| `GET /api/video/{id}` | source video with Range support, so `<video>` can seek |
| `GET /media/{id}/frames/000123.jpg` | sampled frame (384 px wide) |
| `GET /media/{id}/atlas/000.jpg` | thumbnail texture page |
| `POST /api/reload` | re-read the data dir |

Signed-in routes (`Authorization: Bearer <Auth0 access token>`, or `Bearer device:<id>` when uploads without sign-in are on):

| Route | Does |
|---|---|
| `POST /api/uploads` `{title, filename, size}` | starts a browser upload: `{upload_id, part_size, parts}`. 413 over the size limit, 429 over a daily limit (the message names which) |
| `PUT /api/uploads/{id}/parts/{n}` | raw bytes of part `n` (0-based, `part_size` bytes except the last). Re-sending a part replaces it, so retry or resume freely |
| `POST /api/uploads/{id}/complete` `{parts}` | checks every part arrived, assembles the file, starts the job: `{job_id}` (the same id as `upload_id`) |
| `DELETE /api/uploads/{id}` | cancels an unfinished upload (owner only): the job fails with error `cancelled`, its parts are deleted and it stops counting against any limit. 409 once it has completed. Repeating it is harmless |
| `POST /api/uploads/url` `{title, url}` | the server downloads a direct http(s) video link: `{job_id}` |
| `GET /api/jobs` | `{jobs, uploads_today, limits}`, the caller's recent jobs, newest first. `uploads_today` counts the caller's own uploads |
| `GET /api/jobs/{id}` | one job (owner only) |
| `DELETE /api/films/{id}` | deletes one of the caller's films and its video. 403 for someone else's film or a public one |

### Accounts and uploads

Sign-in is Auth0. Set `VS_AUTH0_DOMAIN` (your tenant, e.g. `acme.us.auth0.com`) and
`VS_AUTH0_AUDIENCE` (the API identifier, default `https://vidsearch-api`). The server checks
each access token as an RS256 JWT against the tenant's JWKS, and the user id is the token's `sub`.
Without a domain, everyone is anonymous and the upload routes return 503. Anonymous visitors
can browse and search. A token that is present but invalid or expired gets a 401 on every
route, never a silent fallback to anonymous.

For local end-to-end testing without a tenant, start the API with `VS_DEV_AUTH=1` and no
`VS_AUTH0_DOMAIN`: it then accepts `Authorization: Bearer dev:<user>` as user `<user>` and
logs a warning at startup. A configured domain always wins, so dev auth can't be switched on
in a deployment that has sign-in. Never set it on a server other people can reach.

**Uploads without sign-in.** With `VS_ANON_UPLOADS=1`, the browser is the identity. On the first
upload the frontend generates a random 128-bit id with `crypto.getRandomValues`, keeps it in
`localStorage` and sends `Authorization: Bearer device:<32 lowercase hex>`; the API treats it as
user `device:<hex>`. The id is a bearer secret: whoever holds it sees and deletes those uploads,
and clearing site data loses them. Any other `device:` token is a 401. This works with or without
Auth0 (Auth0 subs never start with `device:`), and dev auth stays available for tests.

`GET /api/library` tells the frontend what to offer, so no build flag is needed: `uploads.mode` is
`account` when Auth0 (or dev auth) is configured, `device` when only `VS_ANON_UPLOADS=1` is, and
`off` otherwise. Accounts win, so setting an Auth0 domain later brings sign-in back even while
`VS_ANON_UPLOADS=1` stays set; uploads made by browser id then stay reachable only through the API.

Uploads are private to the uploader. Everyone sees the public films (no `owner` in
`film.json`: the admin-ingested ones), and each user also sees their own. Library, search and
heatmap all filter by this, and search applies it before each index pools candidates, so
private films can't crowd out results either.

Each upload is at most 2 GB and 60 minutes long. Three daily limits (UTC days) protect the bill,
counted when an upload starts; a cancelled upload gives its slot back to all three:

| Limit | Default | Env |
|---|---|---|
| per user or browser id | 3 | `VS_UPLOADS_PER_DAY` |
| per client IP | 5 | `VS_UPLOADS_PER_IP_PER_DAY` |
| across everyone | 20 | `VS_UPLOADS_GLOBAL_PER_DAY` |

Over any of them the start is a 429 whose message names the limit. The client IP is the TCP peer,
or, when `VS_CLIENT_IP_HEADER` names a header (e.g. `X-Forwarded-For`), that header's right-most
entry: the proxy appends the address it saw, and anything to its left is whatever the client sent.
IPs are stored only as `sha256(VS_IP_SALT, ip)`. Without `VS_IP_SALT` the server picks a random
salt per process, so per-IP counts reset on restart.

**Check the proxy at deploy time.** Behind Modal's proxy, `request.client.host` may be the proxy
rather than the visitor. Then every visitor shares one IP bucket (5 uploads a day for everyone,
which fails safe but is tight). Log `request.client.host` and `X-Forwarded-For` for a request from a
known address. If the header's right-most entry is the visitor, set
`VS_CLIENT_IP_HEADER=X-Forwarded-For`. If it isn't, find the header that is before trusting it:
a header the client can set is no limit at all.

A job record, from `GET /api/jobs/{id}`:

```jsonc
{
  "id": "5f0c…", "owner": "auth0|abc", "title": "My Film", "film_id": "my-film-3fa9c1d2e4b7",
  "source": "file",            // or "url"
  "state": "processing",       // uploading -> queued -> processing -> done; failed can follow any state but done
  "stage": "embedding",        // downloading, probing, detecting shots, sampling frames, building atlas, embedding, captioning
  "done": 640, "total": 3600,  // progress within the stage (while uploading, total = number of parts)
  "error": null,               // failed: why, for the uploader. done: set if captions were skipped
  "created": 1791670000.0,      // when the upload started
  "updated": 1791670420.0,      // last state or progress write
  "video": "my-film-3fa9c1d2e4b7.mp4", "url": null, "size": 734003200
}
```

Poll `GET /api/jobs/{id}` until `state` is `done` or `failed`. A `queued` or `processing`
job whose record hasn't changed for 70 minutes (the GPU function's 60-minute timeout plus a
margin) reads back as `failed` with error `processing timed out`: its worker is gone, so the API
deletes the job's leftover parts, video and film dir the first time it reports that. When the API sees a finished
job, it reloads the library, so the film is in `GET /api/library` with no `POST /api/reload`.
A video over 60 minutes, or a file ffprobe can't read as video, fails with a message saying so.
URL downloads stop at 2 GB and refuse hosts that resolve to private addresses.

**Privacy tradeoff.** `/media/...` and `/api/video/...` have no auth check, because `<video>`
elements and WebGL texture loads can't send an `Authorization` header. A private film is
protected only by its id: the title's slug plus 12 random hex characters (48 bits), e.g.
`my-film-3fa9c1d2e4b7`. Anyone who learns the id, say from a shared frame URL, can fetch that
film's frames and video, but can't list, search or delete it. Signed media URLs would close
this if it matters later.

### Modal

`modal_app.py` runs the API, the frontend and upload processing on Modal:

```bash
modal secret create vidsearch-anthropic --from-dotenv .env         # ANTHROPIC_API_KEY, VS_CAPTION_MODEL
modal secret create vidsearch-auth0 VS_IP_SALT=$(openssl rand -hex 16)  # add VS_AUTH0_DOMAIN once Auth0 is set up
modal deploy modal_app.py
modal run modal_app.py::add_film --url <video url> --name "Cops"  # admin: a public film
```

The `vidsearch` Volume holds `/videos` and `/data`, the same layout as local. The `Web` class
writes upload parts to the Volume, assembles them, commits, and spawns `process_job` on an L4
GPU. That function reloads the Volume, runs the same job body as local (download, probe,
ingest, caption) and commits after ingest and after captions. Job records live in the
`vidsearch-jobs` Dict, so the web container can report progress it didn't compute. `Web` runs
with `max_containers=1`: all parts of an upload must land in one container's view of the
Volume, and one container is the only writer of the per-user quota index. `add_film` uses the
same job body with no owner, so its films stay public. `Web` sets `VS_ANON_UPLOADS=1`, so uploads
work without sign-in; the per-IP limit needs the proxy check above.

A search result:

```jsonc
{
  "rank": 0,
  "film_id": "seven-chances",
  "title": "Seven Chances",
  "t": 2572.0,                 // best-matching frame in the moment (seconds)
  "start": 2570.0, "end": 2573.0,
  "frame": 2572,               // sampled-frame index within the film
  "frame_range": [2570, 2573],
  "shot": 412, "shot_range": [2566.4, 2581.9],
  "similarity": 0.142,         // SigLIP cosine of the best frame (SigLIP scores run low; compare relatively)
  "score": 0.0412,             // fused rank score used for ordering
  "sources": {"frame": 3, "window": 0, "caption": 1},   // rank within each signal
  "caption": "A man runs down a street pursued by a crowd of women in wedding dresses.",
  "frame_url": "/media/seven-chances/frames/002572.jpg",
  "atlas": {"page": 1, "col": 20, "row": 0}
}
```

## Frontend

`web/` is a Vite + React + react-three-fiber app that draws the library as filmstrips and searches it.

```bash
python -m vidsearch serve --port 8765               # any free port
cd web && npm install && VS_API_URL=http://127.0.0.1:8765 npm run dev
```

See `web/README.md` for the build and the browser verification script.

## Drawing it in the frontend

**Filmstrips.** Each film is `n_frames` cells. Cell `i` lives at
`page = floor(i / per_page)`, `k = i % per_page`, `col = k % cols`, `row = floor(k / cols)`.
In three.js, use one `InstancedMesh` per atlas page and set a per-instance UV offset:

```js
const u0 = col * cell_w / texW, v0 = 1 - (row + 1) * cell_h / texH;   // texture's actual size
```

Lay film `j` out as a column at `x = j * gap`, with frame `i` at `y = -i * cellHeight`, tilt the
camera back, and fade with distance. The search result's `atlas` field points straight at
the matching cell.

**Glowing ticks.** Decode the heatmap and feed it into a per-instance attribute that tints
or brightens the strip:

```js
const heat = Uint8Array.from(atob(res.films[id]), c => c.charCodeAt(0));   // length n_frames
```

**Hit cards.** For each search result, lift its instance off the plane, then on hover show a
card with a `<video src={video_url}>` and set `video.currentTime = result.start`.

## Tuning

All settings are environment variables (`vidsearch/config.py`):

| Var | Default | Notes |
|---|---|---|
| `VS_DATA` | `data` | output dir |
| `VS_SAMPLE_FPS` | `1.0` | 2.0 catches quick gags; doubles embed time |
| `VS_SHOT_DETECTOR` | `content` | `adaptive` tolerates flicker and camera motion but can miss cuts |
| `VS_SHOT_THRESHOLD` | content 18, adaptive 3.0 | lower = more cuts. 18 is below PySceneDetect's 27 because faded silent prints hide cuts; under ~15 flicker starts reading as cuts. Grayscale prints are detected and scored on brightness only |
| `VS_WINDOW_SEC` / `VS_WINDOW_STRIDE` | `3` / `1` | moment length and step |
| `VS_CAPTION_CHUNK_SEC` | `12` | long shots are split into chunks this long |
| `VS_IMAGE_MODEL` / `VS_IMAGE_PRETRAINED` | `ViT-B-16-SigLIP2` / `webli` | e.g. `ViT-SO400M-14-SigLIP2` for better recall (needs a GPU). Re-ingest after changing |
| `VS_TEXT_MODEL` | `BAAI/bge-small-en-v1.5` | caption embedder. Re-run `caption` after changing |
| `VS_DEVICE` | auto | `cuda`, `mps`, or `cpu` |
| `VS_AUTH0_DOMAIN` / `VS_AUTH0_AUDIENCE` | empty / `https://vidsearch-api` | empty domain turns sign-in and uploads off |
| `VS_DEV_AUTH` | empty | `1` with no Auth0 domain: `Bearer dev:<user>` signs in as `<user>`. Local testing only |
| `VS_VIDEOS` | `videos` | where uploaded and downloaded videos are stored |
| `VS_UPLOADS_PER_DAY` / `VS_MAX_UPLOAD_BYTES` / `VS_MAX_UPLOAD_MINUTES` | `3` / 2 GiB / `60` | per-user upload limits |
| `VS_UPLOAD_PART_BYTES` | 8 MiB | chunk size for browser uploads |

Fusion weights live at the top of `vidsearch/library.py` (`WEIGHTS`). Bump `caption` if
action queries matter most.

## Tests

```bash
pytest -q tests/
```

The tests use stand-in models (mean colour as the "embedding", hashed bag of words for text)
on generated clips, so they run in ~15 s without downloading weights. They cover ingest,
captions with resume, fused search, de-dupe, the heatmap and every API route.
`tests/test_uploads.py` signs tokens with a throwaway RSA key and covers auth, browser ids, private
visibility, the per-user, per-IP and global limits, size limits, chunked uploads, cancel, job
failures, the timed-out job cleanup and delete.

## Next steps

- **Masks** (the pink cutouts): run SAM 3 with a text prompt like "person", or GroundingDINO
  + SAM 2, on the top-k frames at query time. Cache PNGs under `data/films/<id>/masks/`.
- **Scale**: search is an exact numpy inner product (`_top` in `library.py`), fine up to ~1M frames.
  Past that, swap in an ANN index (e.g. faiss `IndexHNSWFlat`). On macOS, faiss-cpu and torch each
  bundle their own `libomp` and crash when both load in one process, so keep them apart or use one OpenMP.
- **Better action recall without captions**: replace window mean-pooling with a video encoder
  (e.g. InternVideo2 or VideoPrism) over 2–4 s clips.

## Layout

```
vidsearch/
  config.py    settings (VS_* env vars)
  auth.py      Auth0 access token -> user id, as FastAPI dependencies
  jobs.py      upload job record and lifecycle, chunked-upload parts, the job body, local runner
  ingest.py    shots, frames, atlases, frame + window embeddings
  caption.py   optional VLM captions + caption embeddings
  embed.py     SigLIP 2 (open_clip) and sentence-transformer wrappers
  library.py   loads all films, exact numpy search, fused search, heatmap
  server.py    FastAPI app
  __main__.py  CLI
tests/test_pipeline.py, tests/test_uploads.py
modal_app.py   Modal deployment: web class, GPU job function, admin add_film
web/           frontend (see web/README.md)
```
