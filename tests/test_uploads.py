"""Accounts, private uploads and upload jobs, through the HTTP API.

Auth0 is replaced by a locally generated RSA key whose JWKS is served by a patched fetch, and
jobs run on the in-process LocalRunner with the stand-in models from test_pipeline.py, so this
stays offline and fast. Browser identities (`device:<32 hex>`) are their own bearer tokens.
"""

import base64
import json
import subprocess
import secrets
import shutil
import time
from dataclasses import asdict, replace

import jwt
import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from test_pipeline import StubCaptioner, StubImage, StubText, _make_clip

import vidsearch.library
import vidsearch.server
from vidsearch.config import Settings
from vidsearch.ingest import ingest_video
from vidsearch.jobs import STALE_AFTER_S, JobStore, LocalRunner, new_job, user_bucket
from vidsearch.library import Library
from vidsearch.server import create_app

DOMAIN = "vidsearch-test.auth0.example"
ISSUER = f"https://{DOMAIN}/"
AUDIENCE = "https://vidsearch-api"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def token(sub, *, aud=AUDIENCE, iss=ISSUER, ttl=3600, key=KEY):
    now = int(time.time())
    claims = {"sub": sub, "aud": aud, "iss": iss, "iat": now, "exp": now + ttl}
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "test-key"})


def auth(sub, **kw):
    return {"Authorization": f"Bearer {sub if sub.startswith('device:') else token(sub, **kw)}"}


def device():
    return f"device:{secrets.token_hex(16)}"


class RecordingKV(dict):
    """The job store's mapping; remembers every (state, stage) a job was written with."""

    def __init__(self):
        super().__init__()
        self.history: dict[str, list] = {}

    def __setitem__(self, key, value):
        if key.startswith("job:"):
            self.history.setdefault(value["id"], []).append((value["state"], value["stage"]))
        super().__setitem__(key, value)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("uploads")
    clips = root / "clips"
    clips.mkdir()
    _make_clip(clips / "public-film.mp4", [("blue", 4), ("red", 3), ("green", 4)])
    _make_clip(clips / "all-red.mp4", [("red", 8), ("white", 2)])
    _make_clip(clips / "too-long.mp4", [("green", 12), ("blue", 12)])

    s = Settings()
    s.data_dir = root / "data"
    s.videos_dir = root / "videos"
    s.sample_fps = 1.0  # one frame per second keeps the frame counts below easy to read
    s.shot_threshold = "27"
    s.auth0_domain = DOMAIN
    s.auth0_audience = AUDIENCE
    s.upload_part_bytes = 64 * 1024
    s.max_upload_minutes = 0.25  # 15 s, so too-long.mp4 (24 s) is over and all-red.mp4 (10 s) is under
    s.anon_uploads = True  # browser identities beside Auth0
    s.uploads_per_ip_per_day = s.uploads_global_per_day = 1000  # every request here comes from "testclient"

    quiet = lambda m: None  # noqa: E731
    img, txt = StubImage(), StubText()
    ingest_video(clips / "public-film.mp4", s, embedder=img, log=quiet)
    lib = Library(s, image_embedder=img, text_embedder=txt, log=quiet)
    kv = RecordingKV()
    runner = LocalRunner(s, JobStore(kv), embedder=img, captioner=StubCaptioner(), text_embedder=txt, log=quiet)

    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(KEY.public_key()))
    jwks = {"keys": [{**jwk, "kid": "test-key", "alg": "RS256", "use": "sig"}]}
    mp = pytest.MonkeyPatch()
    mp.setattr(jwt.PyJWKClient, "fetch_data", lambda self: jwks)
    with TestClient(create_app(lib=lib, s=s, runner=runner)) as c:
        yield c, s, clips, kv
    mp.undo()


def wait_for(c, job_id, user, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = c.get(f"/api/jobs/{job_id}", headers=auth(user)).json()
        if job["state"] in ("done", "failed"):
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} still {job['state']} after {timeout}s")


def upload(c, path, user, title):
    data = path.read_bytes()
    r = c.post("/api/uploads", json={"title": title, "filename": path.name, "size": len(data)}, headers=auth(user))
    assert r.status_code == 200, r.text
    up = r.json()
    ps = up["part_size"]
    parts = [data[i:i + ps] for i in range(0, len(data), ps)]
    assert up["parts"] == len(parts)
    for n in range(len(parts)):
        assert c.put(f"/api/uploads/{up['upload_id']}/parts/{n}", content=parts[n], headers=auth(user)).status_code == 200
    r = c.post(f"/api/uploads/{up['upload_id']}/complete", json={"parts": len(parts)}, headers=auth(user))
    assert r.status_code == 200, r.text
    return r.json()["job_id"]


@pytest.fixture(scope="module")
def alice_film(env):
    """alice uploads all-red.mp4 in parts: out of order, with part 1 first sent corrupted then re-sent."""
    c, s, clips, kv = env
    src = clips / "all-red.mp4"
    data = src.read_bytes()
    r = c.post("/api/uploads", json={"title": "My Red Film", "filename": "all-red.MP4", "size": len(data)},
               headers=auth("alice"))
    up = r.json()
    ps, n = up["part_size"], up["parts"]
    assert n >= 3
    parts = [data[i:i + ps] for i in range(0, len(data), ps)]
    put = lambda i, body: c.put(f"/api/uploads/{up['upload_id']}/parts/{i}", content=body, headers=auth("alice"))  # noqa: E731
    assert put(1, b"\0" * len(parts[1])).status_code == 200
    for i in reversed(range(n)):
        assert put(i, parts[i]).status_code == 200
    r = c.post(f"/api/uploads/{up['upload_id']}/complete", json={"parts": n}, headers=auth("alice"))
    assert r.status_code == 200, r.text
    job = wait_for(c, r.json()["job_id"], "alice")
    assert job["state"] == "done", job
    return job


# ---------------------------------------------------------------- auth


def test_auth_tokens(env):
    c, *_ = env
    assert c.get("/api/jobs").status_code == 401
    assert c.get("/api/jobs", headers=auth("alice")).status_code == 200
    for bad in (dict(aud="https://someone-else"), dict(iss="https://evil.example/"), dict(ttl=-60),
                dict(key=OTHER_KEY)):
        assert c.get("/api/jobs", headers=auth("alice", **bad)).status_code == 401, bad
    assert c.get("/api/jobs", headers={"Authorization": "Bearer not-a-jwt"}).status_code == 401
    # anonymous may browse, but a bad token is rejected rather than treated as anonymous
    assert c.get("/api/library").status_code == 200
    assert c.get("/api/library", headers=auth("alice", ttl=-60)).status_code == 401
    assert c.get("/api/search", params={"q": "red"}, headers=auth("alice", aud="x")).status_code == 401


def test_uploads_off_without_auth0(env, tmp_path):
    _, s, *_ = env
    off = Settings()
    off.data_dir, off.auth0_domain, off.dev_auth = tmp_path, "", False
    lib = Library(off, image_embedder=StubImage(), log=lambda m: None)
    with TestClient(create_app(lib=lib, s=off)) as c:
        assert c.get("/api/library").status_code == 200
        r = c.post("/api/uploads", json={"title": "x", "filename": "x.mp4", "size": 10})
        assert r.status_code == 503 and r.json()["detail"] == "uploads not configured"


def dev(sub):
    return {"Authorization": f"Bearer dev:{sub}"}


def dev_app(tmp_path, domain):
    s = Settings()
    s.data_dir, s.auth0_domain, s.auth0_audience, s.dev_auth = tmp_path, domain, AUDIENCE, True
    lib = Library(s, image_embedder=StubImage(), log=lambda m: None)
    return TestClient(create_app(lib=lib, s=s))


def test_dev_auth_without_auth0(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="vidsearch.auth"), dev_app(tmp_path, domain="") as c:
        assert "VS_DEV_AUTH=1" in caplog.text
        r = c.post("/api/uploads", json={"title": "x", "filename": "x.mp4", "size": 10}, headers=dev("tester"))
        assert r.status_code == 200, r.text
        job_id = r.json()["upload_id"]
        assert c.get(f"/api/jobs/{job_id}", headers=dev("tester")).json()["owner"] == "tester"
        assert c.get(f"/api/jobs/{job_id}", headers=dev("someone-else")).status_code == 403
        assert c.get("/api/library").status_code == 200
        assert c.get("/api/jobs").status_code == 401
        for bad in ("dev:", "tester", "dev:a b", "dev:tester:x", token("tester")):
            assert c.get("/api/jobs", headers={"Authorization": f"Bearer {bad}"}).status_code == 401, bad
            assert c.get("/api/library", headers={"Authorization": f"Bearer {bad}"}).status_code == 401, bad


def test_dev_auth_is_off_when_auth0_is_configured(env, tmp_path, caplog):
    # env patches the JWKS fetch, so real tokens verify here
    with caplog.at_level("WARNING", logger="vidsearch.auth"), dev_app(tmp_path, domain=DOMAIN) as c:
        assert "VS_DEV_AUTH" not in caplog.text
        assert c.get("/api/jobs", headers=dev("tester")).status_code == 401
        assert c.get("/api/library", headers=dev("tester")).status_code == 401
        assert c.get("/api/jobs", headers=auth("tester")).status_code == 200


# ---------------------------------------------------------------- uploads and jobs


def test_chunked_upload_assembles_ingests_and_appears(env, alice_film):
    c, s, clips, kv = env
    assert alice_film["film_id"].startswith("my-red-film-")
    assert len(alice_film["film_id"].removeprefix("my-red-film-")) == 12
    assert (s.videos_dir / f"{alice_film['film_id']}.mp4").read_bytes() == (clips / "all-red.mp4").read_bytes()
    assert not (s.uploads_dir / alice_film["id"]).exists()

    states = kv.history[alice_film["id"]]
    assert states[0] == ("uploading", None) and states[1] == ("queued", None) and states[-1] == ("done", None)
    stages = [st for state, st in states if state == "processing"]
    for stage in ("probing", "detecting shots", "sampling frames", "building atlas", "embedding", "captioning"):
        assert stage in stages

    films = {f["id"]: f for f in c.get("/api/library", headers=auth("alice")).json()["films"]}
    mine = films[alice_film["film_id"]]
    assert mine["owner_is_you"] and mine["title"] == "My Red Film" and mine["n_captions"] > 0
    assert not films["public-film"]["owner_is_you"]
    assert c.get(f"/api/video/{alice_film['film_id']}").status_code == 200


def test_private_film_is_invisible_to_others(env, alice_film, monkeypatch):
    c, *_ = env
    fid = alice_film["film_id"]
    for headers in ({}, auth("bob")):
        lib = c.get("/api/library", headers=headers).json()
        assert [f["id"] for f in lib["films"]] == ["public-film"]
        assert "alice" not in json.dumps(lib)
        res = c.get("/api/search", params={"q": "red", "k": 10, "heatmap": 1}, headers=headers).json()
        assert res["results"] and all(r["film_id"] == "public-film" for r in res["results"])
        assert set(res["heatmap"]["films"]) == {"public-film"}
        assert set(c.get("/api/heatmap", params={"q": "red"}, headers=headers).json()["films"]) == {"public-film"}
        named = c.get("/api/search", params={"q": "red", "films": fid}, headers=headers).json()
        assert named["results"] == []
        assert c.get(f"/api/jobs/{alice_film['id']}", headers=headers).status_code in (401, 403)

    heat = c.get("/api/heatmap", params={"q": "red"}, headers=auth("alice")).json()["films"]
    v = np.frombuffer(base64.b64decode(heat[fid]), np.uint8)
    assert len(v) == 10 and v[:8].min() > v[8:].max()


def test_private_film_does_not_crowd_others_out_of_search(env, alice_film, monkeypatch):
    c, *_ = env
    # Pools smaller than alice's run of red frames, so her film can fill every slot.
    monkeypatch.setattr(vidsearch.library, "POOL", {"frame": 3, "window": 3, "caption": 1})
    q = {"q": "red", "k": 5, "use": "frame,window"}
    mine = c.get("/api/search", params=q, headers=auth("alice")).json()["results"]
    assert mine[0]["film_id"] == alice_film["film_id"]
    for headers in ({}, auth("bob")):
        res = c.get("/api/search", params=q, headers=headers).json()["results"]
        assert res and res[0]["film_id"] == "public-film" and 4 <= res[0]["t"] < 7


def test_daily_quota(env):
    c, *_ = env
    body = {"title": "x", "filename": "x.mp4", "size": 1000}
    for _ in range(3):
        assert c.post("/api/uploads", json=body, headers=auth("carol")).status_code == 200
    r = c.post("/api/uploads", json=body, headers=auth("carol"))
    assert r.status_code == 429 and "daily limit" in r.json()["detail"]
    assert c.post("/api/uploads/url", json={"title": "x", "url": "https://example.com/a.mp4"},
                  headers=auth("carol")).status_code == 429
    jobs = c.get("/api/jobs", headers=auth("carol")).json()
    assert jobs["uploads_today"] == 3 and len(jobs["jobs"]) == 3


def test_size_cap(env):
    c, s, *_ = env
    r = c.post("/api/uploads", json={"title": "big", "filename": "big.mp4", "size": s.max_upload_bytes + 1},
               headers=auth("dave"))
    assert r.status_code == 413
    up = c.post("/api/uploads", json={"title": "ok", "filename": "ok.mp4", "size": 10}, headers=auth("dave")).json()
    r = c.put(f"/api/uploads/{up['upload_id']}/parts/0", content=b"x" * (s.upload_part_bytes + 1), headers=auth("dave"))
    assert r.status_code == 413
    assert c.get("/api/jobs", headers=auth("dave")).json()["uploads_today"] == 1


def test_complete_rejects_missing_parts(env):
    c, s, *_ = env
    up = c.post("/api/uploads", json={"title": "gap", "filename": "g.mp4", "size": 3 * s.upload_part_bytes},
                headers=auth("erin")).json()
    for n in (0, 2):
        c.put(f"/api/uploads/{up['upload_id']}/parts/{n}", content=b"x" * s.upload_part_bytes, headers=auth("erin"))
    r = c.post(f"/api/uploads/{up['upload_id']}/complete", json={"parts": 3}, headers=auth("erin"))
    assert r.status_code == 400 and "missing parts: [1]" in r.json()["detail"]
    assert c.put(f"/api/uploads/{up['upload_id']}/parts/1", content=b"x", headers=auth("bob")).status_code == 403


def test_url_upload_rejects_non_http(env):
    c, *_ = env
    for url in ("ftp://example.com/a.mp4", "file:///etc/passwd", "javascript:alert(1)", "example.com/a.mp4"):
        r = c.post("/api/uploads/url", json={"title": "x", "url": url}, headers=auth("frank"))
        assert r.status_code == 400, url
    assert c.get("/api/jobs", headers=auth("frank")).json()["uploads_today"] == 0


def test_url_upload_refuses_private_hosts(env):
    c, s, *_ = env
    r = c.post("/api/uploads/url", json={"title": "local", "url": "http://127.0.0.1:9/a.mp4"}, headers=auth("gina"))
    job = wait_for(c, r.json()["job_id"], "gina")
    assert job["state"] == "failed" and "private address" in job["error"]
    assert not (s.videos_dir / job["video"]).exists()


def test_too_long_video_fails_with_clear_error(env):
    c, s, clips, _ = env
    job = wait_for(c, upload(c, clips / "too-long.mp4", "henry", "Too Long"), "henry")
    assert job["state"] == "failed"
    assert job["error"] == "the video is 0.4 min long; the limit is 0.25 min"
    assert not (s.videos_dir / job["video"]).exists() and not (s.films_dir / job["film_id"]).exists()
    assert job["film_id"] not in {f["id"] for f in c.get("/api/library", headers=auth("henry")).json()["films"]}


def test_not_a_video_fails_with_clear_error(env, tmp_path):
    c, s, *_ = env
    junk = tmp_path / "notes.mp4"
    junk.write_bytes(b"this is not a video\n" * 100)
    job = wait_for(c, upload(c, junk, "ivan", "Junk"), "ivan")
    assert job["state"] == "failed" and job["error"].startswith("not a video")


def test_forced_reingest_keeps_film_private(env, alice_film):
    _, s, *_ = env
    video = s.videos_dir / f"{alice_film['film_id']}.mp4"
    fid = ingest_video(video, s, force=True, embedder=StubImage(), log=lambda m: None)
    assert fid == alice_film["film_id"]
    assert json.loads((s.films_dir / fid / "film.json").read_text())["owner"] == "alice"


# ---------------------------------------------------------------- delete


def test_delete_film(env, alice_film):
    c, s, *_ = env
    fid = alice_film["film_id"]
    assert c.delete("/api/films/public-film", headers=auth("alice")).status_code == 403
    assert c.delete(f"/api/films/{fid}", headers=auth("bob")).status_code == 403
    assert c.delete(f"/api/films/{fid}").status_code == 401
    assert c.delete(f"/api/films/{fid}", headers=auth("alice")).status_code == 200
    assert fid not in {f["id"] for f in c.get("/api/library", headers=auth("alice")).json()["films"]}
    assert not (s.films_dir / fid).exists() and not (s.videos_dir / f"{fid}.mp4").exists()
    assert c.get(f"/api/video/{fid}").status_code == 404
    assert c.delete(f"/api/films/{fid}", headers=auth("alice")).status_code == 404


# ---------------------------------------------------------------- stale jobs


def test_stale_jobs_read_as_timed_out(env):
    c, s, clips, kv = env
    store, now = JobStore(kv), time.time()
    old, recent = now - STALE_AFTER_S - 60, now - STALE_AFTER_S + 60
    cases = {
        "processing, idle 71 min": (dict(state="processing", stage="embedding", done=5, total=9, updated=old), "failed"),
        "queued, idle 71 min": (dict(state="queued", updated=old), "failed"),
        "processing, idle 69 min": (dict(state="processing", stage="embedding", updated=recent), "processing"),
        "done long ago": (dict(state="done", updated=old), "done"),
        "uploading, idle 71 min": (dict(state="uploading", updated=old), "uploading"),
    }
    ids = {}
    for name, (fields, _) in cases.items():
        job = replace(new_job("kim", name, "url", ".mp4", url="https://example.com/a.mp4"), created=old, **fields)
        store.create(job, [])
        ids[name] = job.id
    legacy = new_job("kim", "legacy", "url", ".mp4", url="https://example.com/a.mp4")
    store.create(legacy, [])
    rec = asdict(replace(legacy, state="processing", stage="captioning", created=old))
    del rec["updated"]  # written before jobs had `updated`
    kv[f"job:{legacy.id}"] = rec
    cases["legacy processing record"], ids["legacy processing record"] = (rec, "failed"), legacy.id

    listed = {j["id"]: j for j in c.get("/api/jobs", headers=auth("kim")).json()["jobs"]}
    for name, (_, state) in cases.items():
        job = c.get(f"/api/jobs/{ids[name]}", headers=auth("kim")).json()
        assert job["state"] == state, name
        assert listed[ids[name]]["state"] == state, name
        assert (job["error"] == "processing timed out") == (state == "failed"), name
    assert kv[f"job:{ids['processing, idle 71 min']}"]["state"] == "processing"  # reported, not rewritten


# ---------------------------------------------------------------- uploads without sign-in (browser identities)

ID = "0123456789abcdef0123456789abcdef"


def device_app(tmp_path, **overrides):
    """An API with VS_ANON_UPLOADS=1 and no Auth0: device ids are the only identities."""
    s = Settings()
    s.data_dir, s.videos_dir = tmp_path / "data", tmp_path / "videos"
    s.auth0_domain, s.dev_auth, s.anon_uploads, s.ip_salt = "", False, True, "test-salt"
    s.upload_part_bytes = 1024
    for k, v in overrides.items():
        setattr(s, k, v)
    kv = {}
    lib = Library(s, image_embedder=StubImage(), log=lambda m: None)
    return TestClient(create_app(lib=lib, s=s, runner=LocalRunner(s, JobStore(kv), log=lambda m: None))), s, kv


def test_device_ids(tmp_path):
    c, *_ = device_app(tmp_path)
    with c:
        assert c.get("/api/jobs", headers=auth(f"device:{ID}")).status_code == 200
        assert c.get("/api/library").status_code == 200
        assert c.get("/api/jobs").status_code == 401
        for bad in (f"device:{ID[:-1]}", f"device:{ID}0", f"device:{ID.upper()}", ID, f"device:{'g' * 32}",
                    "device:", f"Device:{ID}", "dev:tester", token("tester")):
            for path in ("/api/jobs", "/api/library"):
                assert c.get(path, headers={"Authorization": f"Bearer {bad}"}).status_code == 401, (bad, path)


def test_library_advertises_the_upload_mode(env, tmp_path):
    c, s, *_ = env
    uploads = c.get("/api/library").json()["uploads"]
    assert uploads["mode"] == "account"  # Auth0 configured: sign-in wins, device ids still work
    assert c.get("/api/jobs", headers=auth(device())).status_code == 200
    assert uploads["limits"] == {"uploads_per_day": 3, "uploads_per_ip_per_day": 1000, "uploads_global_per_day": 1000,
                                 "max_bytes": s.max_upload_bytes, "max_minutes": 0.25, "part_size": 64 * 1024}
    for anon, dev_auth, mode in ((True, False, "device"), (False, False, "off"), (True, True, "account")):
        c2, *_ = device_app(tmp_path / mode, anon_uploads=anon, dev_auth=dev_auth)
        with c2:
            assert c2.get("/api/library").json()["uploads"]["mode"] == mode, (anon, dev_auth)


def test_device_upload_is_private_to_that_browser(env):
    c, s, clips, _ = env
    me, other = device(), device()
    job = wait_for(c, upload(c, clips / "all-red.mp4", me, "Browser Red"), me)
    assert job["state"] == "done" and job["owner"] == me and "quota" not in job
    fid = job["film_id"]
    assert {f["id"]: f for f in c.get("/api/library", headers=auth(me)).json()["films"]}[fid]["owner_is_you"]
    mine = c.get("/api/search", params={"q": "red", "k": 50}, headers=auth(me)).json()["results"]
    assert any(r["film_id"] == fid for r in mine)
    for headers in ({}, auth(other), auth("alice")):
        assert fid not in {f["id"] for f in c.get("/api/library", headers=headers).json()["films"]}
        res = c.get("/api/search", params={"q": "red", "k": 50, "heatmap": 1}, headers=headers).json()
        assert all(r["film_id"] != fid for r in res["results"]) and fid not in res["heatmap"]["films"]
        assert c.get(f"/api/jobs/{job['id']}", headers=headers).status_code in (401, 403)
    assert c.delete(f"/api/films/{fid}", headers=auth(other)).status_code == 403
    assert c.delete(f"/api/films/{fid}", headers=auth(me)).status_code == 200


def test_per_ip_and_global_daily_caps(tmp_path):
    c, s, kv = device_app(tmp_path, uploads_per_ip_per_day=2, uploads_global_per_day=3,
                          client_ip_header="X-Forwarded-For")
    body = {"title": "x", "filename": "x.mp4", "size": 10}

    def start(who, forwarded):
        return c.post("/api/uploads", json=body, headers={**auth(who), "X-Forwarded-For": forwarded})

    with c:
        # The proxy appends the client's address; entries left of it are whatever the client sent.
        assert start(device(), "10.0.0.1, 203.0.113.7").status_code == 200
        assert start(device(), "10.0.0.2, 203.0.113.7").status_code == 200
        r = start(device(), "10.0.0.3,203.0.113.7")
        assert r.status_code == 429
        assert r.json()["detail"] == "daily limit of 2 uploads from this network reached; try again after 00:00 UTC"
        assert start(device(), "203.0.113.7, 198.51.100.9").status_code == 200
        r = start(device(), "198.51.100.10")
        assert r.status_code == 429
        assert r.json()["detail"] == ("the archive's daily limit of 3 uploads across everyone is reached; "
                                      "try again after 00:00 UTC")
        assert not any(ip in repr(kv) for ip in ("203.0.113.7", "198.51.100.9", "198.51.100.10"))


def test_cancel_gives_the_slots_back_and_removes_parts(tmp_path, monkeypatch):
    c, s, kv = device_app(tmp_path, uploads_per_day=1, uploads_per_ip_per_day=1, uploads_global_per_day=1)
    body = {"title": "x", "filename": "x.mp4", "size": 3 * s.upload_part_bytes}
    part = b"x" * s.upload_part_bytes
    me, other = device(), device()
    with c:
        uid = c.post("/api/uploads", json=body, headers=auth(me)).json()["upload_id"]
        for n in (0, 1):
            assert c.put(f"/api/uploads/{uid}/parts/{n}", content=part, headers=auth(me)).status_code == 200
        assert (s.uploads_dir / uid).is_dir()
        assert c.post("/api/uploads", json=body, headers=auth(other)).status_code == 429
        assert c.delete(f"/api/uploads/{uid}").status_code == 401
        assert c.delete(f"/api/uploads/{uid}", headers=auth(other)).status_code == 403
        for _ in range(2):
            assert c.delete(f"/api/uploads/{uid}", headers=auth(me)).status_code == 200
        job = c.get(f"/api/jobs/{uid}", headers=auth(me)).json()
        assert job["state"] == "failed" and job["error"] == "cancelled"
        assert not (s.uploads_dir / uid).exists()
        assert c.get("/api/jobs", headers=auth(me)).json()["uploads_today"] == 0
        assert c.put(f"/api/uploads/{uid}/parts/2", content=part, headers=auth(me)).status_code == 409
        assert c.post(f"/api/uploads/{uid}/complete", json={"parts": 3}, headers=auth(me)).status_code == 409
        assert not (s.uploads_dir / uid).exists()

        # The user, IP and global slots are all free again: another browser on the same network can start.
        uid2 = c.post("/api/uploads", json=body, headers=auth(other)).json()["upload_id"]

        # A part in flight when the cancel lands must not leave a parts dir behind.
        def cancel_lands_first(s_, job_id, n, data):
            assert c.delete(f"/api/uploads/{job_id}", headers=auth(other)).status_code == 200
            real_write_part(s_, job_id, n, data)

        real_write_part = vidsearch.server.write_part
        monkeypatch.setattr(vidsearch.server, "write_part", cancel_lands_first)
        assert c.put(f"/api/uploads/{uid2}/parts/0", content=part, headers=auth(other)).status_code == 409
        assert not (s.uploads_dir / uid2).exists()


def test_cancel_refuses_a_finished_upload(tmp_path):
    c, *_ = device_app(tmp_path)
    me = device()
    with c:
        uid = c.post("/api/uploads", json={"title": "x", "filename": "x.mp4", "size": 10}, headers=auth(me)).json()["upload_id"]
        assert c.put(f"/api/uploads/{uid}/parts/0", content=b"not video!", headers=auth(me)).status_code == 200
        assert c.post(f"/api/uploads/{uid}/complete", json={"parts": 1}, headers=auth(me)).status_code == 200
        r = c.delete(f"/api/uploads/{uid}", headers=auth(me))
        assert r.status_code == 409 and r.json()["detail"].startswith("upload is ")
        assert c.get("/api/jobs", headers=auth(me)).json()["uploads_today"] == 1


def test_timed_out_job_leaves_nothing_behind(env):
    c, s, clips, kv = env
    me, quiet = device(), (lambda m: None)
    old = time.time() - STALE_AFTER_S - 60
    job = new_job(me, "Stuck", "file", ".mp4", size=10, part_size=10)
    # Its worker died while captioning: the film is ingested and loaded, and a stray parts dir is left too.
    shutil.copy(clips / "all-red.mp4", s.videos_dir / job.video)
    assert ingest_video(s.videos_dir / job.video, s, owner=me, embedder=StubImage(), log=quiet) == job.film_id
    (s.uploads_dir / job.id).mkdir(parents=True)
    (s.uploads_dir / job.id / "00000").write_bytes(b"x")
    JobStore(kv).create(replace(job, state="processing", stage="captioning", created=old, updated=old), [])
    c.post("/api/reload")
    assert c.get(f"/api/video/{job.film_id}").status_code == 200  # loaded (the library call would reap it)

    for _ in range(2):  # every read reports it; cleaning up again is harmless
        got = c.get(f"/api/jobs/{job.id}", headers=auth(me)).json()
        assert got["state"] == "failed" and got["error"] == "processing timed out"
    assert not (s.videos_dir / job.video).exists()
    assert not (s.films_dir / job.film_id).exists()
    assert not (s.uploads_dir / job.id).exists()
    assert job.film_id not in {f["id"] for f in c.get("/api/library", headers=auth(me)).json()["films"]}


def _make_webm_with_audio(path, width=1920, height=1080, seconds=3):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", f"testsrc2=s={width}x{height}:r=24:d={seconds}",
                    "-f", "lavfi", "-i", f"sine=f=440:d={seconds}",
                    "-c:v", "libvpx-vp9", "-deadline", "realtime", "-b:v", "500k", "-c:a", "libopus",
                    "-shortest", str(path)], check=True)


def test_normalize_reencodes_to_browser_safe_mp4(tmp_path):
    from vidsearch.jobs import _video_info, normalize_video

    src = tmp_path / "clip.webm"
    _make_webm_with_audio(src)
    seen = []
    out = normalize_video(src, 1280, lambda d, t: seen.append((d, t)))
    info = _video_info(out)
    assert out == tmp_path / "clip.mp4" and not src.exists()
    assert info["codec"] == "h264" and info["pix_fmt"] == "yuv420p" and info["mp4"]
    assert info["width"] == 1280 and not info["audio"]
    assert seen and seen[-1][0] == seen[-1][1]  # progress reaches its total
    assert normalize_video(out, 1280) == out  # already compliant: left alone


def test_webm_upload_is_converted_before_ingest(env, tmp_path):
    c, s, clips, kv = env
    me = device()
    src = tmp_path / "phone-clip.webm"
    _make_webm_with_audio(src, width=640, height=360, seconds=4)
    job = wait_for(c, upload(c, src, me, "Phone Clip"), me)
    assert job["state"] == "done", job
    assert job["video"].endswith(".mp4") and (s.videos_dir / job["video"]).exists()
    assert not (s.videos_dir / job["video"]).with_suffix(".webm").exists()
    assert "converting" in [st for state, st in kv.history[job["id"]] if state == "processing"]
    r = c.get(f"/api/video/{job['film_id']}", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and r.headers["content-type"] == "video/mp4"
    assert c.delete(f"/api/films/{job['film_id']}", headers=auth(me)).status_code == 200
