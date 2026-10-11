# vidsearch web

The whole library as strips of frame slices on a tilted plane, with one search pill. Results dim the archive,
light matching frames in lilac and lift the top 12 as cards. Hover a card or a frame to play it; arrow keys step
through results. The archive title comes from `VITE_ARCHIVE_TITLE` (default `The Keaton Archive`).

The API may be cold (it scales to zero): the library request retries transient failures (network errors, 5xx) with
backoff for up to 60 s behind a centred loader before offering a retry button, searches retry twice, and each atlas
page retries three times before its strip is skipped.

Sign-in is Auth0 (`@auth0/auth0-react`). Anyone can browse and search; signed-in users can upload a video file or a
direct video link, which becomes a private film only they see (marked with a lilac bar under its near end). The
avatar menu at the top right has `upload`, `my uploads` (with delete) and `sign out`. A chip beside the avatar shows
an upload's progress while it runs, even with the panel closed. Every API call goes through `src/api.ts`, which adds
the bearer token, retries a 401 once with a fresh token, then signs the user out with a message.

| Build env | Default | |
|---|---|---|
| `VITE_API_URL` | same origin | the API's origin, e.g. the Modal URL |
| `VITE_AUTH0_DOMAIN` | empty | the Auth0 tenant, e.g. `acme.us.auth0.com`. Empty hides sign-in |
| `VITE_AUTH0_CLIENT_ID` | empty | the Auth0 single-page application's client id |
| `VITE_AUTH0_AUDIENCE` | `https://vidsearch-api` | the Auth0 API identifier; must match the API's `VS_AUTH0_AUDIENCE` |
| `VITE_DEV_AUTH` | empty | dev server only: with no Auth0 domain, act signed in as this user and send `Bearer dev:<user>` (the API needs `VS_DEV_AUTH=1`). Production builds drop this code |

```bash
npm install
VS_API_URL=http://127.0.0.1:8765 npm run dev   # proxies /api and /media to the vidsearch API (default http://127.0.0.1:8000)
npm run build                                   # type-checks, then builds to dist/
```

`npm run verify -- <outDir> <label>` drives a running dev server in headless Chromium at 1920x1080 (`BASE_URL`,
default http://localhost:5199). It saves screenshots and checks the archive title and stats, the loaders (slow, flaky and dead
library API, atlas progress, failing atlas pages), the overview framing, atlas mapping on slices and lifted cards,
the hover card's placement, frame still and video seek, search states and retries, camera travel, frame timing, and
that a 3-film library still fits and draws. Then the account half: sign-in UI, quota, client-side refusals, a
chunked upload of a 30 s clip with an injected part failure, the job's stages, the new private strip and the camera
easing to it, search and privacy across two dev users, URL errors, delete, and an expired session. That half needs
the API on `VS_DEV_AUTH=1` and the dev server on `VITE_DEV_AUTH=tester`; `ONLY=archive` or `ONLY=account` runs one
half. It needs a Chromium in `~/Library/Caches/ms-playwright` or `CHROMIUM=/path/to/chrome`.

```bash
VS_DEV_AUTH=1 python -m vidsearch serve --port 8765                                       # from the repo root
VS_API_URL=http://127.0.0.1:8765 VITE_DEV_AUTH=tester npx vite --port 5199 --strictPort  # here
npm run verify -- /tmp/vidsearch-verify v5
```
