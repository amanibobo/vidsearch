"""Media on R2: jobs publish every object, the API redirects to presigned URLs for published films and serves
the rest from disk, and deletes reach the bucket. The bucket is an in-memory stand-in for the few S3 calls used."""

import json
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import boto3
import botocore.auth
import pytest
from botocore.config import Config
from fastapi.testclient import TestClient
from test_pipeline import StubCaptioner, StubImage, StubText, _make_clip
from test_uploads import RecordingKV, auth, device, upload, wait_for

import vidsearch.media
from vidsearch.config import Settings
from vidsearch.ingest import ingest_video
from vidsearch.jobs import STALE_AFTER_S, JobStore, LocalRunner, new_job
from vidsearch.library import Library
from vidsearch.media import R2, publish_film
from vidsearch.server import create_app

KEY_ID, SECRET = "AKIDTESTKEY0001", "s3cr3t-NEVER-in-a-response"
NOW = 1_760_000_000.0  # 2025-10-09 08:53:20 UTC


class FakeS3:
    PAGE = 7  # small, so listings take several pages

    def __init__(self):
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.puts: list[str] = []
        self.lock = threading.Lock()

    def upload_file(self, filename, bucket, key, ExtraArgs):
        with self.lock:
            self.objects[key] = (Path(filename).read_bytes(), ExtraArgs["ContentType"])
            self.puts.append(key)

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken="0"):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        i = int(ContinuationToken)
        page = keys[i:i + self.PAGE]
        return {"Contents": [{"Key": k, "Size": len(self.objects[k][0])} for k in page],
                "IsTruncated": i + self.PAGE < len(keys), "NextContinuationToken": str(i + self.PAGE)}

    def delete_objects(self, Bucket, Delete):
        for o in Delete["Objects"]:
            self.objects.pop(o["Key"], None)

    def under(self, prefix):
        return {k for k in self.objects if k.startswith(prefix)}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("media")
    _make_clip(root / "public-film.mp4", [("blue", 4), ("red", 3)])
    _make_clip(root / "all-red.mp4", [("red", 8), ("white", 2)])
    s = Settings()
    s.data_dir, s.videos_dir = root / "data", root / "videos"
    s.sample_fps, s.shot_threshold = 1.0, "27"
    s.auth0_domain, s.dev_auth, s.anon_uploads, s.ip_salt = "", False, True, "test-salt"
    s.upload_part_bytes = 64 * 1024
    quiet = lambda m: None  # noqa: E731
    img, txt = StubImage(), StubText()
    ingest_video(root / "public-film.mp4", s, embedder=img, log=quiet)  # never published: served from disk
    fake = FakeS3()
    r2 = R2("acct", KEY_ID, SECRET, "bucket", client=fake)
    kv = RecordingKV()
    runner = LocalRunner(s, JobStore(kv), embedder=img, captioner=StubCaptioner(), text_embedder=txt, log=quiet, r2=r2)
    with TestClient(create_app(lib=Library(s, image_embedder=img, text_embedder=txt, log=quiet), s=s,
                               runner=runner, r2=r2)) as c:
        me = device()
        job = wait_for(c, upload(c, root / "all-red.mp4", me, "Red"), me)
        assert job["state"] == "done" and job["error"] is None, job
        yield c, s, fake, r2, kv, me, job


def test_job_publishes_every_object_once(env):
    c, s, fake, r2, kv, me, job = env
    fid, film = job["film_id"], s.films_dir / job["film_id"]
    frames, atlas = sorted((film / "frames").glob("*.jpg")), sorted((film / "atlas").glob("*.jpg"))
    assert len(frames) == 10 and atlas
    expected = {f"videos/{fid}.mp4": s.videos_dir / f"{fid}.mp4",
                **{f"films/{fid}/frames/{p.name}": p for p in frames},
                **{f"films/{fid}/atlas/{p.name}": p for p in atlas}}
    assert set(fake.objects) == set(expected)
    for key, path in expected.items():
        assert fake.objects[key] == (path.read_bytes(), "video/mp4" if key.startswith("videos/") else "image/jpeg")
    assert sorted(fake.puts) == sorted(expected)
    assert json.loads((film / "film.json").read_text())["media"] == "r2"
    assert "public-film" not in json.dumps(sorted(fake.objects))

    stages = [st for state, st in kv.history[job["id"]] if state == "processing"]
    assert stages[-1] == "publishing" and stages.index("publishing") > stages.index("captioning")

    assert publish_film(r2, film) == 0
    assert sorted(fake.puts) == sorted(expected)


def test_published_media_redirects_to_a_signed_url(env, monkeypatch):
    c, s, fake, r2, kv, me, job = env
    fid = job["film_id"]
    monkeypatch.setattr(vidsearch.media.time, "time", lambda: NOW)
    for path, key in ((f"/api/video/{fid}", f"videos/{fid}.mp4"),
                      (f"/media/{fid}/frames/000003.jpg", f"films/{fid}/frames/000003.jpg"),
                      (f"/media/{fid}/atlas/000.jpg", f"films/{fid}/atlas/000.jpg")):
        r = c.get(path, follow_redirects=False)
        assert r.status_code == 302, path
        url = urlsplit(r.headers["location"])
        assert (url.netloc, url.path) == ("acct.r2.cloudflarestorage.com", f"/bucket/{key}")
        assert r.headers["location"] == r2.presign(key, NOW)[0]
        assert r.headers["cache-control"] == "private, max-age=400"  # until 09:00 UTC


def test_unpublished_media_is_served_from_disk(env):
    c, s, *_ = env
    film = s.films_dir / "public-film"
    r = c.get("/media/public-film/frames/000001.jpg", follow_redirects=False)
    assert r.status_code == 200 and r.content == (film / "frames" / "000001.jpg").read_bytes()
    assert c.get("/media/public-film/atlas/000.jpg", follow_redirects=False).status_code == 200
    r = c.get("/api/video/public-film", headers={"Range": "bytes=0-99"}, follow_redirects=False)
    assert r.status_code == 206 and len(r.content) == 100


def test_unknown_media_is_404(env):
    c, s, fake, r2, kv, me, job = env
    fid = job["film_id"]
    for path in ("/api/video/no-such-film", "/media/no-such-film/frames/000001.jpg", "/media/no-such-film/atlas/000.jpg",
                 "/media/public-film/frames/999999.jpg", f"/media/{fid}/emb/x.npy"):
        assert c.get(path, follow_redirects=False).status_code == 404, path


def test_signed_urls_are_stable_within_the_hour():
    r2 = R2("acct", KEY_ID, SECRET, "bucket")
    key = "films/f/frames/000001.jpg"
    hour = NOW // 3600 * 3600
    url, max_age = r2.presign(key, hour)
    assert max_age == 3600
    assert r2.presign(key, hour + 3599) == (url, 1)
    later, _ = r2.presign(key, hour + 3600)
    assert later != url
    q = parse_qs(urlsplit(url).query)
    assert q["X-Amz-Date"] == [datetime.fromtimestamp(hour, timezone.utc).strftime("%Y%m%dT%H%M%SZ")]
    assert q["X-Amz-Expires"] == ["7200"]  # valid until an hour after the window ends


def test_signature_matches_botocore(monkeypatch):
    signed_at = datetime.fromtimestamp(NOW // 3600 * 3600, timezone.utc)
    monkeypatch.setattr(botocore.auth, "get_current_datetime", lambda: signed_at.replace(tzinfo=None))
    client = boto3.client("s3", endpoint_url="https://acct.r2.cloudflarestorage.com", region_name="auto",
                          aws_access_key_id=KEY_ID, aws_secret_access_key=SECRET,
                          config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"))
    key = "films/a film+x/frames/000001.jpg"
    theirs = client.generate_presigned_url("get_object", Params={"Bucket": "bucket", "Key": key}, ExpiresIn=7200)
    ours, _ = R2("acct", KEY_ID, SECRET, "bucket").presign(key, NOW)
    split = lambda u: (urlsplit(u).path, parse_qs(urlsplit(u).query))  # noqa: E731
    assert split(ours) == split(theirs)


def test_secret_never_leaves_the_server(env):
    c, s, fake, r2, kv, me, job = env
    fid = job["film_id"]
    responses = [c.get("/api/library", headers=auth(me)), c.get("/api/search", params={"q": "red"}),
                 c.get("/api/jobs", headers=auth(me)), c.get(f"/api/jobs/{job['id']}", headers=auth(me)),
                 c.get(f"/api/video/{fid}", follow_redirects=False),
                 c.get(f"/media/{fid}/frames/000001.jpg", follow_redirects=False)]
    for r in responses:
        assert SECRET not in r.text and SECRET not in str(r.headers.raw)
    assert SECRET not in repr(r2)
    s.r2_secret_access_key = SECRET
    assert SECRET not in repr(s)


def test_timed_out_job_removes_its_objects(env):
    c, s, fake, r2, kv, me, job = env
    old = time.time() - STALE_AFTER_S - 60
    stuck = new_job(me, "Stuck", "file", ".mp4", size=10, part_size=10)
    for key in (f"videos/{stuck.video}", f"films/{stuck.film_id}/frames/000000.jpg", f"films/{stuck.film_id}/atlas/000.jpg"):
        fake.objects[key] = (b"x", "image/jpeg")
    JobStore(kv).create(replace(stuck, state="processing", stage="publishing", created=old, updated=old), [])
    got = c.get(f"/api/jobs/{stuck.id}", headers=auth(me)).json()
    assert got["error"] == "processing timed out"
    assert not fake.under(f"films/{stuck.film_id}/") and f"videos/{stuck.video}" not in fake.objects
    assert fake.under(f"films/{job['film_id']}/")


def test_delete_removes_objects(env):
    c, s, fake, r2, kv, me, job = env
    fid = job["film_id"]
    assert fake.under(f"films/{fid}/")
    assert c.delete(f"/api/films/{fid}", headers=auth(me)).status_code == 200
    assert not fake.under(f"films/{fid}/") and f"videos/{fid}.mp4" not in fake.objects
    assert c.get(f"/api/video/{fid}", follow_redirects=False).status_code == 404
    assert c.get(f"/media/{fid}/frames/000001.jpg", follow_redirects=False).status_code == 404
    assert c.delete(f"/api/films/{fid}", headers=auth(me)).status_code == 404
