# vidsearch web

The Traffic Archive's frontend: dashcam and traffic-camera clips as strips of frame slices on a tilted plane, with one search pill. Results dim the archive,
light matching frames in lilac and lift the top 12 as cards. Hover a card or a frame to play it; arrow keys step
through results. The archive's title and what it calls an item (`clip`) come from the API's `archive` in
`/api/library`, for the masthead, the tab title, the stats line and the upload copy. Until the API answers, the title is
`VITE_ARCHIVE_TITLE`, else the page's `<title>` (`The Traffic Archive` in `index.html`; the API rewrites it to its own
when it serves the build).

The API may be cold (it scales to zero): the library request retries transient failures (network errors, 5xx) with
backoff for up to 60 s behind a centred loader before offering a retry button, searches retry twice, and each atlas
page retries three times before its strip is skipped.

Anyone can browse and search. Uploads (a video file or a direct video link) become private clips only their uploader
sees, marked with a lilac bar under the strip's near end. What the top-right corner offers comes from the API's
`uploads.mode` in `/api/library`, so the build needs no flag for it:

- `device` (the API runs `VS_ANON_UPLOADS=1`): `upload` and `my uploads` links, no sign-in. The first upload mints a
  random id (`src/device.ts`), kept in localStorage and sent as `Bearer device:<id>`. The panel's footer says
  "saved to this browser", with the full caveat in its tooltip. A browser that has uploaded before asks for the library a second time as itself before
  anything shows, so its own strips arrive with the archive.
- `account`: Auth0 (`@auth0/auth0-react`). `sign in` when signed out; signed in, an avatar menu with `upload`,
  `my uploads` and `sign out`. Shown only when this build has `VITE_AUTH0_DOMAIN` (or `VITE_DEV_AUTH` in dev).
- `off`: no upload UI.

Both links open one panel (`src/UploadPanel.tsx`) with `upload` and `my uploads` tabs and a dot meter of the day's
uploads. The upload tab follows an upload through its steps under one overall progress bar that never goes back
(`stepProgress` in `src/uploads.ts` weighs each step by its rough share of the time), then offers `view in archive`.
My uploads lists running uploads first, then ready clips (a frame still, duration, age; click to fly to the strip;
delete asks inline) and failures with their reason until dismissed. A chip in the corner shows the same overall
progress while an upload runs, even with the panel closed. Cancelling an upload, or
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
that a library of up to 3 clips still fits and draws. Then the upload half, in whichever mode the API advertises: quota,
client-side refusals, a chunked upload of a 30 s clip (cut from the library's first video) with an injected part failure, the job's stages, the new
private strip and the camera easing to it, overall progress never going back, search and privacy across two
identities, URL errors, the panel's tabs (fixed position, eased height), reduced motion, my uploads rows (thumbnail,
duration, age, click to fly there), delete with its inline confirm, and the empty state. Device mode adds the corner
links, the browser id and its note, getting the strip back after a reload, a running upload's row, cancel freeing the
slot, a failed upload's row and dismiss, and the per-IP limit's message. Screenshots of panel states come with a 2x
crop of the corner. Account mode adds sign-in and an expired session. `ONLY=archive` or
`ONLY=uploads` runs one half; `ONLY=prod` checks a production build served by `vite preview`. `ONLY=collection` checks the
collection's identity: the title before and after the library loads, the stats and upload copy in its nouns, and hover seek
on every clip sampled at other than 1 fps (`SERVED_PAGE=<API URL>` adds the page the API serves with `VS_WEB_DIST`). The upload half fills
the API's per-IP quota, so start the API fresh before each run (counts live in its memory). Nothing in it names an
archive: the clip it checks atlases on, its frames, the upload clip and the identity all come from `/api/library`, so it
runs against any library of at least two clips. It needs `ffmpeg` and a Chromium in `~/Library/Caches/ms-playwright`
or `CHROMIUM=/path/to/chrome`.

```bash
VS_ANON_UPLOADS=1 python -m vidsearch serve --port 8765                  # from the repo root (VS_DEV_AUTH=1 for account mode)
VS_API_URL=http://127.0.0.1:8765 npx vite --port 5199 --strictPort       # here (add VITE_DEV_AUTH=tester for account mode)
npm run verify -- /tmp/vidsearch-verify v6

VITE_API_URL=http://127.0.0.1:8765 npx vite build --outDir /tmp/vs-prod && npx vite preview --outDir /tmp/vs-prod --port 5198
ONLY=prod BASE_URL=http://localhost:5198 npm run verify -- /tmp/vidsearch-verify prod
```
