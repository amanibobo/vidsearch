"""End-to-end test with stand-in models, so it runs in seconds without downloading weights.

Generates two clips of solid-colour shots with ffmpeg, then checks ingest, captions (with
resume), fused search, de-duplication, the heatmap, and every API route.

    pytest -q tests/
"""

import base64
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vidsearch.caption import caption_film
from vidsearch.config import Settings
from vidsearch.ingest import find_videos, ingest_video
from vidsearch.library import Library

COLORS = {"red": (255, 0, 0), "green": (0, 128, 0), "blue": (0, 0, 255), "white": (255, 255, 255)}


def _feat(rgb):
    v = np.concatenate([np.array(rgb, np.float32) / 255 - 0.33, [0.05]])
    return v / np.linalg.norm(v)


class StubImage:
    """Mean colour as the 'embedding'; a colour word as the 'query'."""
    name = "stub-color/v1"

    def encode_images(self, ims):
        return np.stack([_feat(np.asarray(im.convert("RGB")).reshape(-1, 3).mean(0)) for im in ims])

    def encode_query(self, q):
        return next((_feat(c) for w, c in COLORS.items() if w in q), _feat((128, 128, 128)))


class StubText:
    """Hashed bag of words."""
    name = "stub-bow"

    def _v(self, t):
        v = np.zeros(64, np.float32)
        for w in t.lower().replace(".", "").split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        return v / max(np.linalg.norm(v), 1e-9)

    def encode_passages(self, ts):
        return np.stack([self._v(t) for t in ts])

    def encode_query(self, q):
        return self._v(q)


class StubCaptioner:
    calls = 0

    def caption(self, frames):
        StubCaptioner.calls += 1
        m = np.asarray(Image.open(frames[0]).convert("RGB")).reshape(-1, 3).mean(0)
        if m[0] > 150 and m[1] < 100:
            return "A man is being chased down the street by a crowd."
        if m.min() > 200:
            return "A bright white wall."
        return "A quiet room, people standing still."


def _make_clip(path: Path, segments):
    args, filt = [], ""
    for i, (c, d) in enumerate(segments):
        args += ["-f", "lavfi", "-i", f"color=c={c}:s=320x240:r=24:d={d}"]
        filt += f"[{i}:v]noise=alls=12:allf=t[v{i}];"
    filt += "".join(f"[v{i}]" for i in range(len(segments))) + f"concat=n={len(segments)}:v=1:a=0[o]"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args, "-filter_complex", filt,
                    "-map", "[o]", "-pix_fmt", "yuv420p", str(path)], check=True)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    root = tmp_path_factory.mktemp("vs")
    vids = root / "vids"
    vids.mkdir()
    _make_clip(vids / "Seven_Chances_1925.mp4", [("blue", 6), ("red", 5), ("green", 7), ("blue", 4), ("red", 3)])
    _make_clip(vids / "sherlock-jr.mp4", [("green", 5), ("white", 6), ("red", 8), ("blue", 5)])

    s = Settings()
    s.data_dir = root / "data"
    # Noisy flat white has random hue, which scores ~22 every frame and swallows real cuts at the
    # default 18; real footage doesn't behave like this, so pin PySceneDetect's stock threshold.
    s.shot_threshold = "27"
    img = StubImage()
    for v in find_videos([vids]):
        ingest_video(v, s, embedder=img, log=lambda m: None)
    for d in sorted(s.films_dir.glob("*/film.json")):
        caption_film(d.parent, s, captioner=StubCaptioner(), text_embedder=StubText(), log=lambda m: None)
    return Library(s, image_embedder=img, text_embedder=StubText(), log=lambda m: None)


def test_ingest_artifacts(lib):
    assert [f.meta["id"] for f in lib.films] == ["seven-chances-1925", "sherlock-jr"]
    assert lib.films[0].meta["title"] == "Seven Chances 1925"
    assert lib.films[0].shots == [[0.0, 6.0], [6.0, 11.0], [11.0, 18.0], [18.0, 22.0], [22.0, 25.0]]
    assert len(lib.F) == 49 and lib.n_captions == 9


def test_caption_resume_skips_done(lib):
    before = StubCaptioner.calls
    caption_film(lib.films[1].dir, lib.s, captioner=StubCaptioner(), text_embedder=StubText(), log=lambda m: None)
    assert StubCaptioner.calls == before


def test_visual_search_finds_red_shots(lib):
    r = lib.search("red", k=6)
    red = {("Seven Chances 1925", 6, 11), ("Seven Chances 1925", 22, 25), ("Sherlock Jr", 11, 19)}
    for x in r[:3]:
        assert x["similarity"] > 0.99
        assert any(x["title"] == t and a <= x["t"] < b for t, a, b in red), x


def test_caption_search_finds_action(lib):
    r = lib.search("being chased", k=5, use=("caption",))
    assert all("chased" in x["caption"] for x in r[:3])


def test_film_filter_and_dedupe(lib):
    assert all(x["film_id"] == "sherlock-jr" for x in lib.search("blue", k=4, films=["sherlock-jr"]))
    r = lib.search("red", k=50)
    for i, a in enumerate(r):
        for b in r[i + 1:]:
            if a["film_id"] == b["film_id"]:
                assert abs((a["start"] + a["end"]) / 2 - (b["start"] + b["end"]) / 2) >= 4.0 - 1e-6


def test_heatmap(lib):
    v = np.frombuffer(base64.b64decode(lib.heatmap("red")["films"]["sherlock-jr"]), np.uint8)
    assert len(v) == 24 and v[11:19].min() > 200 and v[:5].max() < 50


def test_api(lib):
    from fastapi.testclient import TestClient

    from vidsearch.server import create_app

    with TestClient(create_app(lib=lib, s=lib.s)) as c:
        L = c.get("/api/library").json()
        assert L["has_captions"] and len(L["films"]) == 2
        R = c.get("/api/search", params={"q": "red", "k": 3}).json()["results"]
        assert len(R) == 3
        fr = c.get(R[0]["frame_url"])
        assert fr.status_code == 200 and fr.headers["content-type"] == "image/jpeg"
        at = c.get(L["films"][0]["atlas"]["urls"][0])
        assert Image.open(io.BytesIO(at.content)).size == (42 * 96, 72)
        vr = c.get("/api/video/seven-chances-1925", headers={"Range": "bytes=0-99"})
        assert vr.status_code == 206 and len(vr.content) == 100
        assert c.get("/api/heatmap", params={"q": "red"}).status_code == 200
        both = c.get("/api/search", params={"q": "red", "heatmap": 1}).json()
        alone = c.get("/api/heatmap", params={"q": "red"}).json()
        assert both["results"] and both["heatmap"] == {k: v for k, v in alone.items() if k != "query"}
        assert "max-age" in fr.headers["cache-control"] and "max-age" in vr.headers["cache-control"]
        assert c.post("/api/reload").json() == {"films": 2}
        assert c.get("/api/video/nope").status_code == 404


def test_query_encoded_once_across_search_and_heatmap(lib, monkeypatch):
    calls = []
    real = lib.image_embedder.encode_query
    monkeypatch.setattr(lib.image_embedder, "encode_query", lambda q: calls.append(q) or real(q))
    lib.search("a blue door")
    lib.heatmap("a blue door")
    lib.search("a blue door", k=3)
    assert calls == ["a blue door"]


def test_api_serves_frontend_and_runs_reload_hook(lib, tmp_path):
    from fastapi.testclient import TestClient

    from vidsearch.server import create_app

    (tmp_path / "index.html").write_text("<title>vidsearch</title>")
    lib.s.web_dist = str(tmp_path)
    calls = []
    try:
        with TestClient(create_app(lib=lib, s=lib.s, before_reload=lambda: calls.append(1))) as c:
            assert "<title>vidsearch</title>" in c.get("/").text
            assert c.get("/api/library").json()["films"]
            assert c.post("/api/reload").json() == {"films": 2} and calls == [1]
    finally:
        lib.s.web_dist = ""
