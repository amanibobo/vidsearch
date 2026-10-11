"""Load every ingested film into one in-memory index and answer queries.

Search returns *moments* (short windows inside a shot), ranked by reciprocal-rank fusion of
three signals, then de-duplicated so you don't get five hits from the same two seconds:

    frames    SigLIP query vs every sampled frame        (objects, people, settings)
    windows   SigLIP query vs mean of ~3 s of frames     (smoother, a bit of temporal context)
    captions  text query vs VLM shot captions (optional) (actions, narrative: "being chased")
"""

from __future__ import annotations

import base64
import json
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import Settings

RRF_K = 60
WEIGHTS = {"frame": 1.0, "window": 1.0, "caption": 1.5}
POOL = {"frame": 1000, "window": 500, "caption": 200}  # candidates pulled from each index


@dataclass
class Film:
    meta: dict
    dir: Path
    frame_offset: int
    shots: list


def _ranges_concat(f0: np.ndarray, f1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For ranges [f0, f1): (flat indices, owning range id)."""
    lens = (f1 - f0).astype(np.int64)
    owner = np.repeat(np.arange(len(f0)), lens)
    starts = np.repeat(f0.astype(np.int64), lens)
    within = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    return starts + within, owner


def _top(emb: np.ndarray, q: np.ndarray, k: int, rows: np.ndarray | None = None) -> np.ndarray:
    """Indices of the k rows of emb with the highest inner product with q, best first (exact).
    rows: optional bool mask; masked-out rows are never returned."""
    sims = emb @ q
    if rows is not None:
        sims = np.where(rows, sims, -np.inf)
        k = min(k, int(rows.sum()))
    k = min(k, len(sims))
    if k == 0:
        return np.zeros(0, np.int64)
    top = np.argpartition(-sims, k - 1)[:k]
    return top[np.argsort(-sims[top], kind="stable")]


class Library:
    def __init__(self, s: Settings, image_embedder=None, text_embedder=None, log=print):
        self.s = s
        self._image_embedder = image_embedder
        self._text_embedder = text_embedder
        self.log = log
        # Query vectors depend only on the model, not the library, so they survive reload().
        self.query_vector = lru_cache(maxsize=512)(self._encode_query)
        self.reload()

    # ------------------------------------------------------------ loading

    def reload(self) -> None:
        films: list[Film] = []
        F, f_film, f_t, f_shot = [], [], [], []
        W, w_rows = [], []
        C, c_rows, c_text = [], [], []
        model = None
        offset = 0
        metas = sorted(self.s.films_dir.glob("*/film.json")) if self.s.films_dir.exists() else []
        for mp in metas:
            meta = json.loads(mp.read_text())
            d = mp.parent
            if model is None:
                model = (meta["image_model"], meta["embed_dim"])
            elif (meta["image_model"], meta["embed_dim"]) != model:
                self.log(f"skipping {meta['id']}: embedded with {meta['image_model']}, library uses {model[0]}")
                continue
            fi = len(films)
            films.append(Film(meta, d, offset, json.loads((d / "shots.json").read_text())))
            n = meta["n_frames"]
            F.append(np.load(d / "emb_frames.npy"))
            f_film.append(np.full(n, fi, np.int32))
            f_t.append(np.load(d / "frame_times.npy"))
            f_shot.append(np.load(d / "frame_shot.npy"))
            for st, en, sh, a, b in json.loads((d / "windows.json").read_text()):
                w_rows.append((fi, st, en, sh, a + offset, b + offset))
            W.append(np.load(d / "emb_windows.npy"))
            if (d / "captions.json").exists() and (d / "emb_captions.npy").exists():
                for st, en, sh, a, b, text in json.loads((d / "captions.json").read_text()):
                    c_rows.append((fi, st, en, sh, a + offset, b + offset))
                    c_text.append(text)
                C.append(np.load(d / "emb_captions.npy"))
            offset += n

        self.films = films
        self.by_id = {f.meta["id"]: i for i, f in enumerate(films)}
        self.image_model = model[0] if model else None
        if not films:
            self.F = np.zeros((0, 1), np.float32)
            self.n_windows = self.n_captions = 0
            self.log("library is empty - run `python -m vidsearch ingest <videos>` first")
            return

        self.F = np.concatenate(F).astype(np.float32)
        self.f_film = np.concatenate(f_film)
        self.f_t = np.concatenate(f_t)
        self.f_shot = np.concatenate(f_shot)

        self.Wemb = np.concatenate(W).astype(np.float32)
        self.w = np.array(w_rows, dtype=np.float64)  # film, start, end, shot, gf0, gf1
        self.w_film = self.w[:, 0].astype(np.int64)
        self.n_windows = len(self.w)

        # each frame -> the window whose centre is closest (frames sit in several overlapping windows)
        wf0, wf1 = self.w[:, 4].astype(np.int64), self.w[:, 5].astype(np.int64)
        flat, owner = _ranges_concat(wf0, wf1)
        dist = np.abs((self.w[owner, 1] + self.w[owner, 2]) / 2 - self.f_t[flat])
        order = np.lexsort((dist, flat))
        flat, owner = flat[order], owner[order]
        first = np.ones(len(flat), bool)
        first[1:] = flat[1:] != flat[:-1]
        self.frame_to_window = np.full(len(self.F), -1, np.int64)
        self.frame_to_window[flat[first]] = owner[first]

        self.n_captions = len(c_rows)
        self.c = np.array(c_rows, dtype=np.float64) if c_rows else np.zeros((0, 6))
        self.c_film = self.c[:, 0].astype(np.int64)
        self.c_text = c_text
        self.Cemb = None
        self.caption_to_windows: list[np.ndarray] = []
        self.window_caption = np.full(self.n_windows, -1, np.int64)
        if c_rows:
            self.Cemb = np.concatenate(C).astype(np.float32)
            # windows overlapping each caption chunk (same film, overlapping global frame ranges)
            starts = wf0
            for ci, (_, _, _, _, a, b) in enumerate(c_rows):
                lo = np.searchsorted(starts, a - (wf1 - wf0).max(), side="left")
                hi = np.searchsorted(starts, b, side="left")
                cand = np.arange(lo, hi)
                cand = cand[(wf0[cand] < b) & (wf1[cand] > a)]
                self.caption_to_windows.append(cand)
                # a window's caption = the chunk containing its centre frame
                mids = (wf0[cand] + wf1[cand] - 1) // 2
                self.window_caption[cand[(mids >= a) & (mids < b)]] = ci

        self.log(f"loaded {len(films)} films, {len(self.F)} frames, {self.n_windows} windows, {self.n_captions} captions")

    # ------------------------------------------------------------ embedders

    @property
    def image_embedder(self):
        if self._image_embedder is None:
            from .embed import image_text_embedder

            self._image_embedder = image_text_embedder(self.s)
            if self.image_model and self._image_embedder.name != self.image_model:
                self.log(f"warning: library embedded with {self.image_model}, querying with {self._image_embedder.name}")
        return self._image_embedder

    @property
    def text_embedder(self):
        if self._text_embedder is None:
            from .embed import text_embedder

            self._text_embedder = text_embedder(self.s)
        return self._text_embedder

    def _encode_query(self, query: str) -> np.ndarray:
        return self.image_embedder.encode_query(query).astype(np.float32)

    def warm(self) -> None:
        """Load models now instead of on the first query."""
        if self.films:
            self.image_embedder
            if self.n_captions:
                self.text_embedder

    # ------------------------------------------------------------ urls / metadata

    @staticmethod
    def frame_url(fid: str, i: int) -> str:
        return f"/media/{fid}/frames/{i:06d}.jpg"

    def atlas_cell(self, film: Film, i: int) -> dict:
        a = film.meta["atlas"]
        page, k = divmod(i, a["per_page"])
        return {"page": page, "col": k % a["cols"], "row": k // a["cols"]}

    # ------------------------------------------------------------ visibility

    def visible(self, user: str | None, films: list[str] | None = None) -> np.ndarray | None:
        """Per-film bool mask of films this caller may see (public + their own), narrowed to `films`
        when given. None means every film, so callers can skip masking."""
        ok = np.array([f.meta.get("owner") in (None, user) for f in self.films], bool)
        if films is not None:
            ok &= np.isin(np.arange(len(self.films)), [self.by_id[f] for f in films if f in self.by_id])
        return None if ok.all() else ok

    def library_json(self, user: str | None = None) -> dict:
        films = []
        vis = self.visible(user)
        for fi, f in enumerate(self.films):
            if vis is not None and not vis[fi]:
                continue
            m = f.meta
            fid = m["id"]
            films.append({
                "id": fid,
                "title": m["title"],
                "duration": m["duration"],
                "width": m["width"],
                "height": m["height"],
                "sample_fps": m["sample_fps"],
                "n_frames": m["n_frames"],
                "atlas": {**m["atlas"], "urls": [f"/media/{fid}/atlas/{p:03d}.jpg" for p in range(m["atlas"]["pages"])]},
                "video_url": f"/api/video/{fid}",
                "shots": f.shots,
                "n_captions": m.get("n_captions", 0),
                "owner_is_you": user is not None and m.get("owner") == user,
            })
        return {"films": films, "image_model": self.image_model, "has_captions": any(f["n_captions"] for f in films)}

    # ------------------------------------------------------------ search

    def search(self, query: str, k: int = 24, films: list[str] | None = None, min_gap: float | None = None,
               use: tuple[str, ...] = ("frame", "window", "caption"), user: str | None = None) -> list[dict]:
        if not self.films or not query.strip():
            return []
        # Restrict every index before it pools candidates, so films the caller can't see never take
        # pool slots from ones they can.
        vis = self.visible(user, films)
        if vis is not None and not vis.any():
            return []
        f_rows = None if vis is None else vis[self.f_film]
        w_rows = None if vis is None else vis[self.w_film]
        c_rows = None if vis is None else vis[self.c_film]
        qi = self.query_vector(query)
        score: dict[int, float] = defaultdict(float)
        why: dict[int, dict] = defaultdict(dict)

        def add(src: str, w: int, rank: int) -> None:
            if src not in why[w]:
                why[w][src] = rank
                score[w] += WEIGHTS[src] / (RRF_K + rank)

        if "frame" in use:
            rank, seen = 0, set()
            for f in _top(self.F, qi, POOL["frame"], f_rows):
                w = int(self.frame_to_window[f])
                if w < 0 or w in seen:
                    continue
                seen.add(w)
                add("frame", w, rank)
                rank += 1

        if "window" in use:
            for rank, w in enumerate(_top(self.Wemb, qi, POOL["window"], w_rows)):
                add("window", int(w), rank)

        if "caption" in use and self.Cemb is not None:
            qt = self.text_embedder.encode_query(query).astype(np.float32)
            for rank, c in enumerate(_top(self.Cemb, qt, POOL["caption"], c_rows)):
                for w in self.caption_to_windows[int(c)]:
                    add("caption", int(w), rank)

        cand = np.fromiter(score.keys(), dtype=np.int64, count=len(score))
        if len(cand) == 0:
            return []
        fused = np.array([score[int(w)] for w in cand])
        cos = self.Wemb[cand] @ qi
        order = np.lexsort((-cos, -fused))  # fused score, then window cosine as tiebreak

        gap = min_gap if min_gap is not None else max(4.0, 1.5 * self.s.window_sec)
        kept: list[int] = []
        kept_at: dict[int, list[float]] = defaultdict(list)
        for j in order:
            w = int(cand[j])
            fi = int(self.w[w, 0])
            centre = (self.w[w, 1] + self.w[w, 2]) / 2
            if any(abs(centre - c) < gap for c in kept_at[fi]):
                continue
            kept.append(j)
            kept_at[fi].append(centre)
            if len(kept) >= k:
                break

        out = []
        for rank, j in enumerate(kept):
            w = int(cand[j])
            fi, st, en, sh, g0, g1 = self.w[w]
            fi, sh, g0, g1 = int(fi), int(sh), int(g0), int(g1)
            film = self.films[fi]
            sims = self.F[g0:g1] @ qi
            best = g0 + int(np.argmax(sims))
            local = best - film.frame_offset
            ci = int(self.window_caption[w])
            out.append({
                "rank": rank,
                "film_id": film.meta["id"],
                "title": film.meta["title"],
                "t": round(float(self.f_t[best]), 3),
                "start": round(float(st), 3),
                "end": round(float(en), 3),
                "frame": local,
                "frame_range": [g0 - film.frame_offset, g1 - film.frame_offset],
                "shot": sh,
                "shot_range": film.shots[sh],
                "score": round(float(fused[j]), 6),
                "similarity": round(float(sims.max()), 4),
                "sources": why[w],
                "caption": self.c_text[ci] if ci >= 0 else None,
                "frame_url": self.frame_url(film.meta["id"], local),
                "atlas": self.atlas_cell(film, local),
            })
        return out

    def heatmap(self, query: str, user: str | None = None) -> dict:
        """Per-frame relevance for every film the caller can see, as base64 uint8 per film.
        Drives the glowing ticks along every filmstrip, not just the top-k hits."""
        vis = self.visible(user) if self.films else None
        if not self.films or not query.strip() or (vis is not None and not vis.any()):
            return {"films": {}}
        q = self.query_vector(query)
        sims = self.F @ q
        seen = sims if vis is None else sims[vis[self.f_film]]
        lo, hi = np.percentile(seen, 50), np.percentile(seen, 99.9)
        v = (np.clip((sims - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        out = {}
        for fi, f in enumerate(self.films):
            if vis is not None and not vis[fi]:
                continue
            n = f.meta["n_frames"]
            out[f.meta["id"]] = base64.b64encode(v[f.frame_offset:f.frame_offset + n].tobytes()).decode()
        return {"films": out, "encoding": "base64 uint8, one value per sampled frame"}
