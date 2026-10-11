"""Collections: the traffic archive is the default and the only one. Its defaults, Modal names, caption prompt, the
archive identity the API hands the frontend, and a 2 fps ingest whose times stay in seconds."""

import base64
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_pipeline import StubCaptioner, StubImage, StubText, _make_clip

from vidsearch.caption import PROMPTS, Captioner, caption_film
from vidsearch.config import COLLECTIONS, Settings
from vidsearch.embed import ImageTextEmbedder
from vidsearch.ingest import ingest_video
from vidsearch.library import Library
from vidsearch.server import create_app

ROOT = Path(__file__).parent.parent

@pytest.fixture
def no_collection_env(monkeypatch):
    for k in ["VIDSEARCH_COLLECTION", "VS_SAMPLE_FPS", "VS_CAPTION_STYLE", "VS_ARCHIVE_TITLE", "VS_ITEM_NOUN",
              "VS_ITEM_NOUN_PLURAL"]:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def identity(s):
    return (s.archive_title, s.item_noun, s.item_noun_plural, s.caption_style, s.sample_fps)


def test_settings_follow_the_collection(no_collection_env):
    env = no_collection_env
    traffic = ("The Traffic Archive", "clip", "clips", "driving", 2.0)
    assert identity(Settings()) == traffic
    for name in ["", "traffic"]:
        env.setenv("VIDSEARCH_COLLECTION", name)
        assert identity(Settings()) == traffic
    env.setenv("VS_SAMPLE_FPS", "3")
    env.setenv("VS_ITEM_NOUN", "video")
    assert identity(Settings()) == ("The Traffic Archive", "video", "videos", "driving", 3.0)
    for name in ["trafic", "keaton"]:
        env.setenv("VIDSEARCH_COLLECTION", name)
        with pytest.raises(ValueError, match=f"unknown VIDSEARCH_COLLECTION '{name}'"):
            Settings()


def modal_app(collection):
    """modal_app imported in a fresh interpreter with VIDSEARCH_COLLECTION as given (None = unset)."""
    env = {k: v for k, v in os.environ.items() if k != "VIDSEARCH_COLLECTION"}
    if collection is not None:
        env["VIDSEARCH_COLLECTION"] = collection
    code = "import json, modal_app as m; print(json.dumps([m.app.name, m.volume.name, m.jobs_dict.name, m.WEB_LABEL]))"
    return subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)


def modal_names(collection):
    out = modal_app(collection)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_a_plain_deploy_uses_the_live_traffic_names():
    pytest.importorskip("modal")
    live = ["vidsearch-traffic", "vidsearch-traffic", "vidsearch-traffic-jobs", "vidsearch-traffic-web"]
    for collection in [None, "", "traffic"]:
        assert modal_names(collection) == live, collection
    unknown = modal_app("keaton")
    assert unknown.returncode != 0 and "unknown VIDSEARCH_COLLECTION 'keaton'" in unknown.stderr


def test_query_templates_are_the_collections(no_collection_env):
    assert Settings().query_templates == ("{}", "a dashcam photo of {}", "a traffic camera image of {}")


def test_query_vector_averages_the_given_templates():
    class Tower:
        seen = []

        def encode_texts(self, texts):
            Tower.seen = list(texts)
            return np.eye(4, dtype=np.float32)[: len(texts)]

    v = ImageTextEmbedder.encode_query(Tower(), "a red car", ("{}", "a dashcam photo of {}"))
    assert Tower.seen == ["a red car", "a dashcam photo of a red car"]
    assert np.allclose(v, [2 ** -0.5, 2 ** -0.5, 0, 0])


class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": "A red car turns left."}}]}


def test_captioner_sends_the_configured_prompt(tmp_path):
    from PIL import Image

    frame = tmp_path / "000000.jpg"
    Image.new("RGB", (64, 48), (200, 0, 0)).save(frame)
    s = replace(Settings(), caption_style="driving", caption_base_url="http://127.0.0.1:9/v1", caption_model="vlm")
    c = Captioner(s)
    sent = []
    c.client.post = lambda url, json: sent.append(json) or FakeResponse()
    assert c.caption([frame, frame]) == "A red car turns left."
    text = sent[0]["messages"][0]["content"][0]["text"]
    assert text == PROMPTS["driving"].format(n=2) and "traffic footage" in text


def test_unknown_caption_style_is_refused():
    for style in ["news", "film"]:
        with pytest.raises(ValueError, match=f"unknown VS_CAPTION_STYLE '{style}'"):
            Captioner(replace(Settings(), caption_style=style, caption_base_url="http://127.0.0.1:9/v1"))


@pytest.fixture(scope="module")
def traffic(tmp_path_factory):
    """A 16 s clip (blue 4 s, red 3 s, green 9 s) ingested and captioned with the default (traffic) settings."""
    mp = pytest.MonkeyPatch()
    for k in ["VIDSEARCH_COLLECTION", "VS_SAMPLE_FPS", "VS_CAPTION_STYLE", "VS_ARCHIVE_TITLE", "VS_ITEM_NOUN", "VS_ITEM_NOUN_PLURAL"]:
        mp.delenv(k, raising=False)
    s = Settings()
    mp.undo()
    root = tmp_path_factory.mktemp("traffic")
    _make_clip(root / "junction.mp4", [("blue", 4), ("red", 3), ("green", 9)])
    s.data_dir = root / "data"
    s.shot_threshold = "27"
    ingest_video(root / "junction.mp4", s, embedder=StubImage(), log=lambda m: None)
    caption_film(s.films_dir / "junction", s, captioner=StubCaptioner(), text_embedder=StubText(), log=lambda m: None)
    return Library(s, image_embedder=StubImage(), text_embedder=StubText(), log=lambda m: None)


def test_two_fps_ingest_times_and_windows(traffic):
    film = traffic.films[0]
    d = film.dir
    assert film.meta["sample_fps"] == 2.0 and film.meta["n_frames"] == 32
    assert film.shots == [[0.0, 4.0], [4.0, 7.0], [7.0, 16.0]]
    assert np.load(d / "frame_times.npy").tolist() == [i / 2 for i in range(32)]
    assert np.load(d / "frame_shot.npy").tolist() == [0] * 8 + [1] * 6 + [2] * 18
    # 3 s windows are 6 frames, one every 1 s (2 frames), inside their shot.
    assert json.loads((d / "windows.json").read_text()) == [
        [0.0, 3.0, 0, 0, 6], [1.0, 4.0, 0, 2, 8],
        [4.0, 7.0, 1, 8, 14],
        *[[7.0 + k, 10.0 + k, 2, 14 + 2 * k, 20 + 2 * k] for k in range(7)],
    ]
    # 12 s caption chunks are 24 frames, so the 9 s shot stays one chunk.
    assert [r[:5] for r in json.loads((d / "captions.json").read_text())] == [
        [0.0, 4.0, 0, 0, 8], [4.0, 7.0, 1, 8, 14], [7.0, 16.0, 2, 14, 32],
    ]


def test_two_fps_search_answers_in_seconds(traffic):
    r = traffic.search("red", k=3)
    top = r[0]
    # Red runs 4-7 s, which is frames 8-13: a frame index leaking into t/start/end would land at 8 or later.
    assert 4.0 <= top["t"] < 7.0 and 4.0 <= top["start"] < top["end"] <= 7.0, top
    assert top["t"] == top["frame"] / 2 and 8 <= top["frame"] < 14
    assert top["end"] - top["start"] == 3.0
    assert top["frame_url"] == f"/media/junction/frames/{top['frame']:06d}.jpg"
    assert "cyclist" in r[0]["caption"]
    heat = np.frombuffer(base64.b64decode(traffic.heatmap("red")["films"]["junction"]), np.uint8)
    assert len(heat) == 32 and heat[8:14].min() > 200 and heat[:8].max() < 50


def test_library_queries_with_its_collections_templates(traffic):
    class Recording(StubImage):
        templates = []

        def encode_query(self, q, templates):
            Recording.templates.append(tuple(templates))
            return super().encode_query(q, templates)

    lib = Library(traffic.s, image_embedder=Recording(), text_embedder=StubText(), log=lambda m: None)
    lib.search("a red car")
    assert Recording.templates == [COLLECTIONS["traffic"].query_templates]


def test_api_names_the_archive(traffic, tmp_path):
    (tmp_path / "index.html").write_text("<html><head><title>A Build Title</title></head></html>")
    s = replace(traffic.s, web_dist=str(tmp_path))
    with TestClient(create_app(lib=traffic, s=s)) as c:
        body = c.get("/api/library").json()
        assert body["archive"] == {"title": "The Traffic Archive", "noun": "clip", "noun_plural": "clips"}
        assert body["films"][0]["sample_fps"] == 2.0
        assert "<title>The Traffic Archive</title>" in c.get("/").text
    s = replace(s, archive_title="Night & Day", item_noun="drive", item_noun_plural="drives")
    with TestClient(create_app(lib=traffic, s=s)) as c:
        assert c.get("/api/library").json()["archive"] == {"title": "Night & Day", "noun": "drive", "noun_plural": "drives"}
        assert "<title>Night &amp; Day</title>" in c.get("/").text
