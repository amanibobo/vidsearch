"""CLI:  python -m vidsearch <command>

    ingest  VIDEO_OR_DIR...   shots, frames, atlases, embeddings
    caption [--film ID]       optional VLM captions for action-level search
    search  "being chased"    query from the terminal
    list                      show ingested films
    serve                     start the API
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="vidsearch", description="Semantic search over a video library.")
    p.add_argument("--data", type=Path, help="data directory (default: $VS_DATA or ./data)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="ingest video files or folders")
    pi.add_argument("paths", nargs="+", type=Path)
    pi.add_argument("--title", help="display title (single file only)")
    pi.add_argument("--fps", type=float, help="frames sampled per second (default 1)")
    pi.add_argument("--force", action="store_true", help="re-ingest films that already exist")

    pc = sub.add_parser("caption", help="caption shots with a vision LLM and embed the captions")
    pc.add_argument("--film", action="append", help="film id (repeatable; default: all)")
    pc.add_argument("--workers", type=int, default=8)

    ps = sub.add_parser("search", help="search from the terminal")
    ps.add_argument("query")
    ps.add_argument("-k", type=int, default=10)
    ps.add_argument("--json", action="store_true")

    sub.add_parser("list", help="list ingested films")

    pv = sub.add_parser("serve", help="run the API server")
    pv.add_argument("--host", default="127.0.0.1")
    pv.add_argument("--port", type=int, default=8000)

    a = p.parse_args(argv)
    s = Settings()
    if a.data:
        s.data_dir = a.data

    if a.cmd == "ingest":
        from .ingest import find_videos, ingest_video

        if a.fps:
            s.sample_fps = a.fps
        videos = find_videos(a.paths)
        if not videos:
            sys.exit("no video files found")
        if a.title and len(videos) > 1:
            sys.exit("--title only works with a single file")
        for v in videos:
            ingest_video(v, s, title=a.title, force=a.force)

    elif a.cmd == "caption":
        from .caption import caption_film

        dirs = sorted(d.parent for d in s.films_dir.glob("*/film.json"))
        if a.film:
            dirs = [d for d in dirs if d.name in set(a.film)]
        if not dirs:
            sys.exit("no matching ingested films")
        for d in dirs:
            caption_film(d, s, workers=a.workers)

    elif a.cmd == "search":
        from .library import Library

        lib = Library(s, log=lambda m: print(m, file=sys.stderr))
        res = lib.search(a.query, k=a.k)
        if a.json:
            print(json.dumps(res, indent=2))
            return
        for r in res:
            m, sec = divmod(r["t"], 60)
            src = ",".join(f"{k}#{v}" for k, v in r["sources"].items())
            print(f"{r['rank']+1:>3}. {r['title']} @ {int(m)}:{sec:04.1f}  sim={r['similarity']:.3f}  [{src}]")
            if r["caption"]:
                print(f"      {r['caption']}")

    elif a.cmd == "list":
        for mp in sorted(s.films_dir.glob("*/film.json")):
            m = json.loads(mp.read_text())
            print(f"{m['id']:<32} {m['title']:<36} {m['duration']/60:6.1f} min  {m['n_frames']:>6} frames  "
                  f"captions={m.get('n_captions', 0)}")

    elif a.cmd == "serve":
        import os

        import uvicorn

        os.environ["VS_DATA"] = str(s.data_dir)
        uvicorn.run("vidsearch.server:app", host=a.host, port=a.port)


if __name__ == "__main__":
    main()
