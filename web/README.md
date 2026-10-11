# vidsearch web

The whole library as strips of frame slices on a tilted plane, with one search pill. Results dim the archive,
light matching frames in lilac and lift the top 12 as cards. Hover a card or a frame to play it; arrow keys step
through results. The archive title comes from `VITE_ARCHIVE_TITLE` (default `The Keaton Archive`).

The API may be cold (it scales to zero): the library request retries transient failures (network errors, 5xx) with
backoff for up to 60 s behind a centred loader before offering a retry button, searches retry twice, and each atlas
page retries three times before its strip is skipped.

Anyone can browse and search. Uploads (a video file or a direct video link) become private films only their uploader
sees, marked with a lilac bar under the strip's near end. What the top-right corner offers comes from the API's
`uploads.mode` in `/api/library`, so the build needs no flag for it:

- `device` (the API runs `VS_ANON_UPLOADS=1`): `upload` and `my uploads` links, no sign-in. The first upload mints a
  random id (`src/device.ts`), kept in localStorage and sent as `Bearer device:<id>`. My uploads says that uploads are
  tied to this browser. A browser that has uploaded before asks for the library a second time as itself before
  anything shows, so its own strips arrive with the archive.
- `account`: Auth0 (`@auth0/auth0-react`). `sign in` when signed out; signed in, an avatar menu with `upload`,
  `my uploads` and `sign out`. Shown only when this build has `VITE_AUTH0_DOMAIN` (or `VITE_DEV_AUTH` in dev).
- `off`: no upload UI.

A chip in the corner shows an upload's progress while it runs, even with the panel closed. Cancelling an upload, or
a part that fails every retry, sends `DELETE /api/uploads/{id}` so it stops counting against the day's uploads.
Every API call goes through `src/api.ts`, which adds the bearer token (the account's, else the browser id), retries
a 401 once with a fresh token, then signs the user out with a message.

| Build env | Default | |
|---|---|---|
| `VITE_API_URL` | same origin | the API's origin, e.g. the Modal URL. The only one browser-id uploads need |
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
that a 3-film library still fits and draws. Then the upload half, in whichever mode the API advertises: quota,
client-side refusals, a chunked upload of a 30 s clip with an injected part failure, the job's stages, the new
private strip and the camera easing to it, search and privacy across two identities, URL errors and delete. Device
mode adds the corner links, the browser id and its note, getting the strip back after a reload, cancel freeing the
slot, and the per-IP limit's message. Account mode adds sign-in and an expired session. `ONLY=archive` or
`ONLY=uploads` runs one half; `ONLY=prod` checks a production build served by `vite preview`. The upload half fills
the API's per-IP quota, so start the API fresh before each run (counts live in its memory). It needs a Chromium in
`~/Library/Caches/ms-playwright` or `CHROMIUM=/path/to/chrome`.

```bash
VS_ANON_UPLOADS=1 python -m vidsearch serve --port 8765                  # from the repo root (VS_DEV_AUTH=1 for account mode)
VS_API_URL=http://127.0.0.1:8765 npx vite --port 5199 --strictPort       # here (add VITE_DEV_AUTH=tester for account mode)
npm run verify -- /tmp/vidsearch-verify v6

VITE_API_URL=http://127.0.0.1:8765 npx vite build --outDir /tmp/vs-prod && npx vite preview --outDir /tmp/vs-prod --port 5198
ONLY=prod BASE_URL=http://localhost:5198 npm run verify -- /tmp/vidsearch-verify prod
```
