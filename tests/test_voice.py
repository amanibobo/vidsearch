"""Voice search tokens: GET /api/voice/token with a stubbed minter, and ElevenLabs failures with a stubbed httpx.
Nothing here touches the network."""

import logging
import secrets

import httpx
import pytest
from fastapi.testclient import TestClient
from test_pipeline import StubImage

from vidsearch.config import Settings
from vidsearch.library import Library
from vidsearch.server import create_app
from vidsearch.voice import ElevenLabsMinter, RateLimited, SlidingLimiter, Window

KEY = "sk_test_" + "k" * 40


class StubMinter:
    def __init__(self):
        self.minted = 0

    def mint(self) -> str:
        self.minted += 1
        return f"sutkn_stub{self.minted}"


def voice_app(tmp_path, minter=None, **overrides):
    s = Settings()
    s.data_dir, s.videos_dir = tmp_path / "data", tmp_path / "videos"
    s.auth0_domain, s.dev_auth, s.anon_uploads, s.ip_salt = "", False, True, "test-salt"
    s.elevenlabs_api_key = KEY
    s.client_ip_header = "X-Forwarded-For"
    for k, v in overrides.items():
        setattr(s, k, v)
    return TestClient(create_app(lib=Library(s, image_embedder=StubImage(), log=lambda m: None), s=s, minter=minter))


def device():
    return {"Authorization": f"Bearer device:{secrets.token_hex(16)}"}


def test_token_comes_from_the_minter_and_library_advertises_voice(tmp_path):
    minter = StubMinter()
    with voice_app(tmp_path, minter) as c:
        assert c.get("/api/library").json()["voice"] == {"available": True}
        r = c.get("/api/voice/token")
        assert r.status_code == 200
        assert r.json() == {"token": "sutkn_stub1", "expires_in": 900}
        assert KEY not in r.text


def test_no_key_means_no_voice(tmp_path):
    with voice_app(tmp_path, elevenlabs_api_key="") as c:
        assert c.get("/api/library").json()["voice"] == {"available": False}
        r = c.get("/api/voice/token")
        assert r.status_code == 503 and r.json()["detail"] == "voice search not configured"


@pytest.mark.parametrize("failure", ["status", "network", "body"])
def test_elevenlabs_failure_is_a_502_that_never_shows_the_key(tmp_path, monkeypatch, caplog, failure):
    sent = {}

    def post(url, headers, timeout):
        sent["key"] = headers["xi-api-key"]
        request = httpx.Request("POST", url, headers=headers)
        if failure == "network":
            raise httpx.ConnectError(f"cannot reach {url} with {headers}", request=request)
        if failure == "status":
            return httpx.Response(401, json={"detail": {"message": f"invalid api key {headers['xi-api-key']}"}}, request=request)
        return httpx.Response(200, text="not json", request=request)

    monkeypatch.setattr(httpx, "post", post)
    caplog.set_level(logging.DEBUG)
    with voice_app(tmp_path) as c:
        r = c.get("/api/voice/token")
    assert sent["key"] == KEY  # the default minter is the real ElevenLabs one, built from the setting
    assert r.status_code == 502
    assert r.json()["detail"] == "the voice service is unavailable; try again or type your search"
    assert KEY not in r.text and KEY not in caplog.text and "sk_" not in caplog.text


def test_minter_sends_the_key_only_as_a_header(monkeypatch):
    seen = {}

    def post(url, headers, timeout):
        seen.update(url=url, headers=headers)
        return httpx.Response(200, json={"token": "sutkn_abc"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    assert ElevenLabsMinter(KEY).mint() == "sutkn_abc"
    assert seen == {"url": "https://api.elevenlabs.io/v1/single-use-token/realtime_scribe", "headers": {"xi-api-key": KEY}}


def test_per_ip_limit_uses_the_proxys_entry_not_the_spoofable_left_one(tmp_path):
    minter = StubMinter()
    with voice_app(tmp_path, minter, voice_tokens_per_ip_per_hour=2, voice_tokens_global_per_day=4) as c:
        get = lambda forwarded: c.get("/api/voice/token", headers={**device(), "X-Forwarded-For": forwarded})  # noqa: E731
        assert get("10.0.0.1, 203.0.113.7").status_code == 200
        assert get("10.0.0.2, 203.0.113.7").status_code == 200
        r = get("10.0.0.3,203.0.113.7")
        assert r.status_code == 429
        assert r.json()["detail"] == "voice search limit of 2 an hour from this network reached; try again later or type your search"
        assert get("203.0.113.7, 198.51.100.9").status_code == 200
        assert get("198.51.100.10").status_code == 200
        r = get("198.51.100.11")
        assert r.status_code == 429
        assert r.json()["detail"] == "the archive's daily voice search limit is reached; try again later or type your search"
    assert minter.minted == 4  # a refused request never reaches ElevenLabs


def test_per_browser_limit(tmp_path):
    with voice_app(tmp_path, StubMinter(), voice_tokens_per_user_per_hour=1) as c:
        me = device()
        get = lambda headers, ip: c.get("/api/voice/token", headers={**headers, "X-Forwarded-For": ip})  # noqa: E731
        assert get(me, "198.51.100.1").status_code == 200
        r = get(me, "198.51.100.2")
        assert r.status_code == 429
        assert r.json()["detail"] == "voice search limit of 1 an hour reached; try again later or type your search"
        assert get(device(), "198.51.100.3").status_code == 200


def test_windows_slide():
    lim = SlidingLimiter()
    w = [Window("ip:x", 2, 3600, "ip full"), Window("all", 3, 86400, "all full")]
    lim.take(w, 0)
    lim.take(w, 10)
    with pytest.raises(RateLimited, match="ip full"):
        lim.take(w, 3599)
    lim.take(w, 3600)  # the first grant has left the hour
    with pytest.raises(RateLimited, match="all full"):
        lim.take([Window("ip:y", 2, 3600, "ip full"), w[1]], 3601)
