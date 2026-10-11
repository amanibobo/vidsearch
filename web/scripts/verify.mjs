// Drives the running dev server in headless Chromium at 1920x1080: screenshots, the archive title and loaders (slow,
// failing and cold API; atlas progress), atlas pixel checks on plane slices and lifted cards, hover-card placement, frame
// still and video seek, search states and retries, camera travel, fps, and the overview at a library of up to 3 films. Then
// uploads, in whichever mode the API advertises: quota, client-side refusals, a chunked upload with an injected part
// failure, job stages and overall progress, the new private strip and camera moves, search and privacy across
// identities, URL errors, the panel's tabs and motion (and reduced motion), my uploads rows, delete, the empty state.
// Device mode (VS_ANON_UPLOADS=1, no auth env on the dev server) adds the browser-identity corner, running and failed
// rows, cancel giving the slot back and the per-IP cap; account mode (VS_DEV_AUTH=1 and VITE_DEV_AUTH=tester) adds sign-in and expired sessions.
// Usage: node scripts/verify.mjs <outDir> <label>   (env BASE_URL, default http://localhost:5199; CHROMIUM to override the binary;
// ONLY=archive or ONLY=uploads runs one half; ONLY=prod checks a production build served by `vite preview` instead;
// ONLY=collection checks the API's collection: its title, nouns, upload copy and hover seek on every film sampled at
// other than 1 fps; SERVED_PAGE=<the API's own URL, run with VS_WEB_DIST> adds the title it serves).
// Everything archive-specific comes from /api/library: title, nouns and stats from its `archive`, the atlas checks from
// its first film, and the upload clip from that film's video (with ffmpeg). Any archive works, given at least two films.
// The upload half fills the API's per-IP quota, so give it a freshly started API (job records and counts live in its memory).
import { chromium } from 'playwright-core'
import { spawnSync } from 'node:child_process'
import { closeSync, existsSync, ftruncateSync, mkdirSync, openSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

const [outDir = '.', label = 'run'] = process.argv.slice(2)
mkdirSync(outDir, { recursive: true })
const BASE = process.env.BASE_URL ?? 'http://localhost:5199'
const VIEWPORT = { width: 1920, height: 1080 }
const CARD_ATLAS_RANKS = [0, 1, 2]
const INPUT = 'input[aria-label="Search"]'
const LIBRARY = await (await fetch(new URL('/api/library', BASE))).json()
const IDENTITY = LIBRARY.archive
// Plain words any footage answers to with some ranking; the second also lifts a full set of cards.
const QUERY = 'a car'
const CARDS_QUERY = 'a road at night'
const WAITING_PLACEHOLDER = 'waiting for the archive…'
const ONLY = process.env.ONLY ?? ''
const DEV_USER = 'tester'
const DEVICE_NOTE = 'uploads are tied to this browser; clearing site data loses access'
const DEVICE_TOKEN = /^device:[0-9a-f]{32}$/
const POLL = 1500

/** Records every loader line (with whether the strips were showing) and every frame-still show/hide in the hover card. */
const instrument = () => {
  window.__loaderLines = []
  window.__stills = []
  const hasStill = (n) => n.nodeType === 1 && (n.matches('.card-still') || !!n.querySelector('.card-still'))
  new MutationObserver((records) => {
    const line = document.querySelector('.loader-line')?.textContent
    const prev = window.__loaderLines.at(-1)
    if (line && prev?.line !== line) window.__loaderLines.push({ line, stageShown: !!document.querySelector('.stage.shown') })
    for (const r of records) {
      if ([...r.addedNodes].some(hasStill)) window.__stills.push({ ev: 'shown' })
      if ([...r.removedNodes].some(hasStill)) {
        const v = document.querySelector('.card video')
        window.__stills.push({ ev: v ? 'hidden' : 'card-gone', video: v && { readyState: v.readyState, seeking: v.seeking, paused: v.paused } })
      }
    }
  }).observe(document, { subtree: true, childList: true, characterData: true })
}
const loaderLines = (page) => page.evaluate(() => window.__loaderLines.map((l) => l.line))
const libraryStats = (films) => {
  const minutes = Math.round(films.reduce((s, f) => s + f.duration, 0) / 60)
  const h = Math.floor(minutes / 60), m = minutes % 60
  const runtime = h && m ? `${h} h ${m} min` : h ? `${h} h` : `${m} min`
  const frames = films.reduce((n, f) => n + f.n_frames, 0)
  return `${films.length} ${films.length === 1 ? IDENTITY.noun : IDENTITY.noun_plural} · ${runtime} · ${frames.toLocaleString('en-US')} frames`
}
const ready = (page, timeout = 20000) => page.waitForSelector('.loader', { state: 'detached', timeout })

function findChromium() {
  if (process.env.CHROMIUM) return process.env.CHROMIUM
  const cache = join(homedir(), 'Library/Caches/ms-playwright')
  for (const dir of readdirSync(cache).filter((d) => /^chromium-\d+$/.test(d)).sort().reverse()) {
    const bin = join(cache, dir, 'chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing')
    if (existsSync(bin)) return bin
  }
  throw new Error('no cached chromium found')
}

const failures = []
const check = (ok, msg) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${msg}`); if (!ok) failures.push(msg) }
const shot = (page, name) => page.screenshot({ path: join(outDir, `${label}-${name}.png`) })
/** The full 1920x1080 view, plus the top-right corner (chip, links, panel) at 2x for reading the panel's details. */
async function shots(page, name) {
  await page.waitForTimeout(300)  // let the panel's fade, row hover and tab fade finish
  await shot(page, name)
  const panel = await page.evaluate(() => { const r = document.querySelector('.panel')?.getBoundingClientRect(); return r && { left: r.left, bottom: r.bottom } })
  const left = Math.max(0, (panel?.left ?? VIEWPORT.width - 460) - 40)
  const clip = { x: left, y: 0, width: VIEWPORT.width - left, height: Math.min(VIEWPORT.height, (panel?.bottom ?? 80) + 40) }
  const cdp = await page.context().newCDPSession(page)
  const { data } = await cdp.send('Page.captureScreenshot', { format: 'png', clip: { ...clip, scale: 2 } })
  await cdp.detach()
  writeFileSync(join(outDir, `${label}-${name}-panel2x.png`), Buffer.from(data, 'base64'))
}
const METER = '.panel-head .quota-meter'
const quotaOf = (page) => page.$eval(METER, (m) => ({ label: m.getAttribute('aria-label'), dots: m.querySelectorAll('i').length, used: m.querySelectorAll('i[data-used]').length }))
const clock = (s) => {
  const tenths = Math.max(0, Math.round(s * 10)), r = tenths % 600
  return `${Math.floor(tenths / 600)}:${String(Math.floor(r / 10)).padStart(2, '0')}.${r % 10}`
}
const frameUrl = (film, frame) => `/media/${film}/frames/${String(frame).padStart(6, '0')}.jpg`

/**
 * Samples the screenshot at `points` (a cols x rows grid the app projected over a slice or card) and compares it with
 * the same region of frame JPEGs: the frame itself and its neighbours. `band` is the [top, height] fraction of the JPEG
 * the grid covers. Returns mean abs RGB diffs per frame and a side-by-side composite PNG.
 */
async function compareRegion(page, { film, frame, nFrames, points, cols, rows, band }) {
  const png = await page.screenshot()
  return page.evaluate(async ([b64, film, frame, n, points, cols, rows, band]) => {
    const load = (src) => new Promise((ok, err) => { const i = new Image(); i.onload = () => ok(i); i.onerror = err; i.src = src })
    const W = 32, H = Math.max(2, Math.round((32 * rows) / cols)), TW = 288, TH = Math.max(24, Math.round((288 * rows) / cols))
    const sheet = document.createElement('canvas'); sheet.width = TW * 4 + 30; sheet.height = TH + 24
    const sctx = sheet.getContext('2d'); sctx.fillStyle = '#222'; sctx.fillRect(0, 0, sheet.width, sheet.height)
    sctx.fillStyle = '#fff'; sctx.font = '14px sans-serif'
    const name = (slot, text) => sctx.fillText(text, slot * (TW + 10) + 4, 17)
    const shotImg = await load(`data:image/png;base64,${b64}`)
    const sc = document.createElement('canvas'); sc.width = shotImg.naturalWidth; sc.height = shotImg.naturalHeight
    const scx = sc.getContext('2d'); scx.drawImage(shotImg, 0, 0)
    const shotPx = scx.getImageData(0, 0, sc.width, sc.height).data
    const scale = sc.width / innerWidth
    const sampled = new ImageData(cols, rows)
    points.forEach(({ x, y }, k) => {
      const o = (Math.round(y * scale) * sc.width + Math.round(x * scale)) * 4
      sampled.data.set([shotPx[o], shotPx[o + 1], shotPx[o + 2], 255], k * 4)
    })
    const tmp = document.createElement('canvas'); tmp.width = cols; tmp.height = rows
    tmp.getContext('2d').putImageData(sampled, 0, 0)
    const small = document.createElement('canvas'); small.width = W; small.height = H
    small.getContext('2d').drawImage(tmp, 0, 0, W, H)
    const rendered = small.getContext('2d').getImageData(0, 0, W, H).data
    sctx.imageSmoothingEnabled = false
    sctx.drawImage(tmp, 0, 24, TW, TH)
    name(0, `rendered ${frame}`)
    const diffs = {}
    for (const [slot, g] of [[1, frame], [2, frame - 1], [3, frame + 1]]) {
      if (g < 0 || g >= n) continue
      const img = await load(`/media/${film}/frames/${String(g).padStart(6, '0')}.jpg`)
      const crop = [img.naturalWidth * 0.1, img.naturalHeight * band[0], img.naturalWidth * 0.8, img.naturalHeight * band[1]]
      sctx.drawImage(img, ...crop, slot * (TW + 10), 24, TW, TH)
      name(slot, `frames/${g}.jpg`)
      const c = document.createElement('canvas'); c.width = W; c.height = H
      const ctx = c.getContext('2d')
      ctx.drawImage(img, ...crop, 0, 0, W, H)
      const ref = ctx.getImageData(0, 0, W, H).data
      let sum = 0
      for (let i = 0; i < ref.length; i += 4) sum += Math.abs(ref[i] - rendered[i]) + Math.abs(ref[i + 1] - rendered[i + 1]) + Math.abs(ref[i + 2] - rendered[i + 2])
      diffs[g] = +(sum / (W * H * 3)).toFixed(2)
    }
    return { diffs, composite: sheet.toDataURL('image/png').split(',')[1] }
  }, [png.toString('base64'), film, frame, nFrames, points, cols, rows, band])
}

function atlasVerdict(what, frame, { diffs, composite }, file) {
  writeFileSync(join(outDir, `${label}-${file}.png`), Buffer.from(composite, 'base64'))
  const own = diffs[frame]
  const others = Object.entries(diffs).filter(([g]) => +g !== frame).map(([, d]) => d)
  check(own < 20 && others.every((d) => own < d), `${what} frame ${frame}: mean abs diff vs own jpg ${own}, vs neighbours ${JSON.stringify(others)}`)
}

/** Mean luminance and count of lilac pixels inside a rectangle of a screenshot. */
async function tone(page, rect) {
  const png = await page.screenshot({ clip: rect, scale: 'css' })
  return page.evaluate(async (b64) => {
    const img = await new Promise((ok) => { const i = new Image(); i.onload = () => ok(i); i.src = `data:image/png;base64,${b64}` })
    const c = document.createElement('canvas'); c.width = img.naturalWidth; c.height = img.naturalHeight
    const ctx = c.getContext('2d'); ctx.drawImage(img, 0, 0)
    const d = ctx.getImageData(0, 0, c.width, c.height).data
    let lum = 0, lilac = 0
    for (let i = 0; i < d.length; i += 4) {
      lum += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]
      if (d[i + 2] > d[i + 1] + 25 && d[i] > d[i + 1] + 15) lilac++
    }
    return { lum: +(lum / (d.length / 4)).toFixed(1), lilac }
  }, png.toString('base64'))
}

async function measureFps(page) {
  const frames = await page.evaluate(() => new Promise((done) => {
    const dts = []
    let last = performance.now()
    const t0 = last
    const tick = (now) => { dts.push(now - last); last = now; if (now - t0 < 3000) requestAnimationFrame(tick); else done(dts) }
    const el = document.querySelector('canvas')
    const pan = setInterval(() => el.dispatchEvent(new WheelEvent('wheel', { deltaY: 40, bubbles: true, cancelable: true })), 16)
    requestAnimationFrame(tick)
    setTimeout(() => clearInterval(pan), 3000)
  }))
  const sorted = frames.slice(1).sort((a, b) => a - b)
  const pct = (p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))].toFixed(1)
  return { fps: frames.length / 3, p50: pct(0.5), p95: pct(0.95), max: sorted.at(-1).toFixed(1) }
}

const browser = await chromium.launch({ executablePath: findChromium(), headless: true, args: (process.env.CHROMIUM_ARGS ?? '').split(' ').filter(Boolean) })
try {
  if (ONLY === 'prod') await prodChecks()
  else if (ONLY === 'collection') await collectionChecks()
  else {
    if (ONLY !== 'uploads') await archiveChecks()
    if (ONLY !== 'archive') await uploadChecks()
  }
} finally {
  await browser.close()
}
if (failures.length) { console.log(`\n${failures.length} failure(s):\n${failures.join('\n')}`); process.exit(1) }
console.log('\nall checks passed')

async function archiveChecks() {
  const page = await browser.newPage({ viewport: VIEWPORT })
  page.on('pageerror', (e) => failures.push(`pageerror: ${e.message}`))
  page.on('console', (m) => { if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) console.log('console.error:', m.text()) })
  page.on('response', (r) => { if (r.status() >= 400 && !r.url().includes('/api/search')) console.log(`HTTP ${r.status()} ${r.url()}`) })
  await page.addInitScript(instrument)
  let atlasRequests = 0
  await page.route('**/media/*/atlas/*', async (r) => { await new Promise((ok) => setTimeout(ok, 150 * atlasRequests++)); await r.continue().catch(() => {}) })
  await page.goto(BASE)
  await page.waitForFunction(() => window.__vidsearch?.settled())
  await ready(page)
  await page.unroute('**/media/*/atlas/*')
  const renderer = await page.evaluate(() => {
    const gl = document.createElement('canvas').getContext('webgl2')
    const ext = gl?.getExtension('WEBGL_debug_renderer_info')
    return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : 'unknown'
  })
  console.log('webgl renderer:', renderer)
  const library = await page.evaluate(async () => (await (await fetch('/api/library')).json()).films)
  console.log('films:', library.map((f) => `${f.id} (${f.n_frames})`).join(', '))
  // The first film's first frames, its middle, and the two frames either side of its first atlas page break (or its end).
  const atlasFilm = library[0].id
  const edge = Math.min(library[0].atlas.per_page, library[0].n_frames - 1)
  const atlasFrames = [0, Math.floor(edge / 2), edge - 1, edge]
  await page.waitForTimeout(2500)
  await shot(page, '2-overview')

  const pages = library.reduce((n, f) => n + f.atlas.urls.length, 0)
  const progress = await page.evaluate(() => window.__loaderLines)
  const counts = progress.map((p) => /^loading frames (\d+) \/ (\d+)$/.exec(p.line)).filter(Boolean)
  console.log('loader lines:', progress.map((p) => p.line).join(' | '))
  check(counts.length >= 3 && counts.every((m, i) => +m[2] === pages && (i === 0 || +m[1] > +counts[i - 1][1])) && counts.at(-1)[1] === String(pages),
    `atlas progress counts up to ${pages} / ${pages}: ${counts.map((m) => m[1]).join(', ')}`)
  check(progress.filter((p) => !p.line.endsWith(`${pages} / ${pages}`)).every((p) => !p.stageShown) && await page.evaluate(() => getComputedStyle(document.querySelector('.stage')).opacity) === '1',
    'strips stay hidden while pages load, then fade in; the loader is gone')
  const masthead = { title: await page.textContent('.masthead-title'), stats: await page.textContent('.masthead-stats'), doc: await page.title() }
  check(masthead.title === IDENTITY.title && masthead.doc === IDENTITY.title, `archive title "${masthead.title}", document title "${masthead.doc}" (the API's "${IDENTITY.title}")`)
  check(masthead.stats === libraryStats(library), `archive stats from /api/library: "${masthead.stats}" (expected "${libraryStats(library)}")`)
  check(!(await page.isDisabled(INPUT)) && await page.getAttribute(INPUT, 'placeholder') === 'search the archive', 'search pill enabled once the archive is ready')

  const settle = async () => {
    await page.waitForTimeout(100)
    await page.waitForFunction(() => window.__vidsearch.settled(), null, { timeout: 15000 })
    await page.waitForTimeout(400)
  }
  const view = () => page.evaluate(() => window.__vidsearch.view())
  const overview = await view()

  const pill = await page.locator('form.pill').boundingBox()
  const ends = await page.evaluate((films) => films.map((f) => ({ id: f.id, first: window.__vidsearch.screenOf(f.id, 0), last: window.__vidsearch.screenOf(f.id, f.n_frames - 1) })), library)
  const inView = (p) => p && p.x > 0 && p.x < VIEWPORT.width && p.y > 0 && p.y < pill.y
  check(ends.every((e) => inView(e.first) && inView(e.last)), `overview shows every film's first and last frame above the search pill (pill top ${pill.y.toFixed(0)})`)
  const nearY = ends.map((e) => e.first.y)
  check(Math.max(...nearY) - Math.min(...nearY) < 2, `near ends (frame 0) line up: screen y ${nearY.map((y) => y.toFixed(1)).join(', ')}`)
  const perColumn = await page.evaluate((id) => { const a = window.__vidsearch.screenOf(id, 0), b = window.__vidsearch.screenOf(id, 1); return { a, b } }, atlasFilm)
  check(perColumn.b.y < perColumn.a.y, `time runs away from the viewer: frame 1 is above frame 0 on screen (${perColumn.b.y.toFixed(1)} < ${perColumn.a.y.toFixed(1)})`)
  const idleTone = await tone(page, { x: 0, y: 150, width: VIEWPORT.width, height: pill.y - 160 })

  if (process.env.ATLAS !== '0') {
    const nFrames = library[0].n_frames
    for (const frame of atlasFrames) {
      await page.evaluate(([id, f]) => window.__vidsearch.focusFrame(id, f, 1400), [atlasFilm, frame])
      await settle()
      const inset = 0.1, cols = 96, rows = 12
      const { band, points } = await page.evaluate(([id, f, c, r, i]) => window.__vidsearch.sliceGrid(id, f, c, r, i), [atlasFilm, frame, cols, rows, inset])
      const result = await compareRegion(page, { film: atlasFilm, frame, nFrames, points, cols, rows, band: [0.5 - band / 2 + inset * band, band * (1 - 2 * inset)] })
      atlasVerdict(`slice ${atlasFilm}`, frame, result, `atlas-slice-${frame}`)
    }
    await page.evaluate(() => window.__vidsearch.showOverview())
    await settle()
  }

  await page.mouse.move(960, 300)
  await page.mouse.down()
  await page.mouse.move(960, 500, { steps: 5 })
  await page.mouse.up()
  const afterDrag = await view()
  check(afterDrag.y > overview.y + 0.5, `dragging the plane down moves forward in time (focus y ${overview.y.toFixed(2)} -> ${afterDrag.y.toFixed(2)})`)
  // Back to the overview first: on a short plane the drag alone reaches its far end.
  await page.evaluate(() => window.__vidsearch.showOverview())
  await settle()
  await page.mouse.wheel(0, 300)
  await settle()
  const afterWheel = await view()
  check(afterWheel.y > overview.y + 1, `wheel pans forward in time (focus y ${overview.y.toFixed(2)} -> ${afterWheel.y.toFixed(2)})`)
  await page.keyboard.down('Control')
  await page.mouse.wheel(0, -100)
  await page.keyboard.up('Control')
  await settle()
  const afterZoom = await view()
  check(afterZoom.zoom > afterWheel.zoom * 1.5, `ctrl+wheel zooms (${afterWheel.zoom.toFixed(0)} -> ${afterZoom.zoom.toFixed(0)} px per unit)`)
  await page.evaluate(() => window.__vidsearch.showOverview())
  await page.mouse.move(5, 5)
  await settle()

  const statusText = () => page.textContent('.pill-status')
  const searchFor = async (q) => {
    await page.fill(INPUT, q)
    await page.press(INPUT, 'Enter')
    await page.waitForSelector('.pill-count, .pill-error', { timeout: 30000 })
    await page.waitForTimeout(1600)
    return statusText()
  }

  let failedCalls = 0
  await page.route('**/api/search**', (r) => { failedCalls++; return r.fulfill({ status: 500, body: 'boom' }) })
  await page.fill(INPUT, QUERY)
  await page.press(INPUT, 'Enter')
  await page.waitForTimeout(2500)
  const waking = await page.textContent('.pill-status')
  check(/waking up…/.test(waking) && await page.isVisible('.pill .spinner'), `a search still retrying after 2 s shows "waking up…" beside the spinner: "${waking}"`)
  await page.waitForSelector('.pill-error', { timeout: 30000 })
  const errStatus = await statusText()
  check(/HTTP 500/.test(errStatus) && failedCalls === 3, `a search failing with 5xx retries twice (${failedCalls} requests), then shows its error inside the pill: "${errStatus}"`)
  await page.unroute('**/api/search**')

  let coldCalls = 0
  await page.route('**/api/search**', (r) => (++coldCalls === 1 ? r.fulfill({ status: 503, body: 'cold' }) : r.continue()))
  const coldStatus = await searchFor(QUERY)
  check(/^\d+$/.test(coldStatus) && coldCalls === 2, `a search whose first attempt gets 503 retries and succeeds: "${coldStatus}" after ${coldCalls} requests`)
  await page.unroute('**/api/search**')

  await page.route('**/api/search?q=a%20dog*', async (r) => { await new Promise((ok) => setTimeout(ok, 3000)); await r.continue().catch(() => {}) })
  await page.fill(INPUT, 'a dog')
  await page.press(INPUT, 'Enter')
  check(await page.isVisible('.pill .spinner'), 'loading shows a spinner inside the pill')
  const queryCount = await searchFor(QUERY)
  const queryTop = await page.evaluate(async (q) => (await (await fetch(`/api/search?q=${encodeURIComponent(q)}`)).json()).results[0], QUERY)
  await page.keyboard.press('ArrowDown')
  await page.waitForSelector('.card-label')
  const queryLabel = await page.textContent('.card-label')
  await page.waitForTimeout(3500)
  await page.unroute('**/api/search?q=a%20dog*')
  const later = { label: await page.textContent('.card-label').catch(() => null), count: await statusText() }
  check(later.label === queryLabel && queryLabel === `${queryTop.title} · ${clock(queryTop.start)}` && later.count === `1/${queryCount}`,
    `slow superseded "a dog" search ignored: card stays "${later.label}", count "${later.count}" (${QUERY} top: ${queryTop.title} · ${clock(queryTop.start)})`)
  check(/^\d+$/.test(queryCount), `search "${QUERY}" shows its result count in the pill: "${queryCount}"`)
  await page.mouse.move(5, 5)
  await page.mouse.move(8, 8)
  await page.waitForSelector('.hover', { state: 'detached' })
  await shot(page, '5-query')
  // ArrowDown flew to the top result if it sat near the overview's edge.
  await page.evaluate(() => window.__vidsearch.showOverview())
  await settle()

  const cardsCount = await searchFor(CARDS_QUERY)
  const results = await page.evaluate(async (q) => (await (await fetch(`/api/search?q=${encodeURIComponent(q)}`)).json()).results, CARDS_QUERY)
  check(cardsCount === String(results.length), `search "${CARDS_QUERY}" count in pill: "${cardsCount}" of ${results.length}`)
  await page.waitForFunction(() => window.__vidsearch.card(0)?.opacity === 1, null, { timeout: 5000 })
  await page.waitForTimeout(1200)
  await shot(page, '4-cards')
  const cards = await page.evaluate((n) => Array.from({ length: n }, (_, r) => window.__vidsearch.card(r)), results.length)
  const lifted = cards.filter(Boolean)
  check(lifted.length >= Math.min(results.length, 10) && lifted.every((c) => c.box), `top results lifted as cards on screen: ${lifted.length} of ${results.length}`)
  const resultsTone = await tone(page, { x: 0, y: 150, width: VIEWPORT.width, height: pill.y - 160 })
  check(resultsTone.lum < idleTone.lum * 0.45 && resultsTone.lilac > idleTone.lilac + 2000,
    `results dim the archive (mean luma ${idleTone.lum} -> ${resultsTone.lum}) and show lilac heat (${idleTone.lilac} -> ${resultsTone.lilac} lilac px)`)
  const view0 = await view()
  check(Math.abs(view0.y - overview.y) < 1e-6 && Math.abs(view0.dist - overview.dist) < 1e-6, 'camera stays at the overview after a search')

  if (process.env.ATLAS !== '0') {
    await page.evaluate(() => window.__vidsearch.rawCards(true))
    for (const rank of CARD_ATLAS_RANKS) {
      const m = results[rank]
      const nFrames = library.find((f) => f.id === m.film_id).n_frames
      await page.evaluate(([id, f]) => window.__vidsearch.focusFrame(id, f, 150), [m.film_id, m.frame])
      await settle()
      const inset = 0.1, cols = 96, rows = 72
      const points = await page.evaluate(([r, c, rr, i]) => window.__vidsearch.cardGrid(r, c, rr, i), [rank, cols, rows, inset])
      const result = await compareRegion(page, { film: m.film_id, frame: m.frame, nFrames, points, cols, rows, band: [inset, 1 - 2 * inset] })
      atlasVerdict(`lifted card rank ${rank} ${m.film_id}`, m.frame, result, `atlas-card-${rank}`)
    }
    await page.evaluate(() => window.__vidsearch.rawCards(false))
    await page.evaluate(() => window.__vidsearch.showOverview())
    await settle()
  }

  await page.evaluate(() => document.addEventListener('seeked', (e) => { window.__seekedTo = e.target.currentTime }, true))
  const top = results[0]
  const card0 = await page.evaluate(() => window.__vidsearch.card(0))
  await page.route('**/api/video/**', async (r) => { await new Promise((ok) => setTimeout(ok, 1200)); await r.continue().catch(() => {}) })
  await page.evaluate(() => { window.__stills = [] })
  await page.mouse.move(card0.center.x - 3, card0.center.y - 3)
  await page.mouse.move(card0.center.x, card0.center.y)
  await page.waitForSelector('.card video')
  const stillShown = await page.isVisible('.card-still img') && await page.isVisible('.card-still .spinner')
  const stillSrc = await page.getAttribute('.card-still img', 'src').catch(() => null)
  await shot(page, '3-hover-still')
  await page.waitForFunction(() => { const v = document.querySelector('.card video'); return v && v.readyState >= 2 && !v.seeking }, null, { timeout: 15000 })
  await page.waitForTimeout(800)
  await page.unroute('**/api/video/**')
  const stills = await page.evaluate(() => window.__stills)
  const hidden = stills.find((s) => s.ev === 'hidden')
  check(stillShown && stillSrc === frameUrl(top.film_id, top.frame) && stills[0]?.ev === 'shown',
    `hover card shows the frame still ${stillSrc} and a spinner before the video plays`)
  check(!!hidden && hidden.video.readyState >= 3 && !hidden.video.seeking && !(await page.$('.card-still')),
    `frame still removed once the video has seeked and can play (${JSON.stringify(hidden?.video)})`)
  const video = await page.evaluate(() => {
    const v = document.querySelector('.card video')
    return { seekedTo: window.__seekedTo, currentTime: v.currentTime, paused: v.paused, error: v.error?.code ?? null }
  })
  console.log('hover video:', JSON.stringify(video))
  check(Math.abs(video.seekedTo - top.start) < 1 && Math.abs(video.currentTime - top.start) < 1 && video.error === null && !video.paused,
    `hover video seeked to ${video.seekedTo?.toFixed(2)}, currentTime ${video.currentTime.toFixed(2)} while playing, both within 1 s of result start ${top.start}`)
  const hoverLabel = await page.textContent('.card-label')
  check(hoverLabel === `${top.title} · ${clock(top.start)}`, `hovering the lifted card shows its result: "${hoverLabel}"`)
  check(await page.evaluate(() => getComputedStyle(document.querySelector('.stage')).cursor) === 'pointer', 'pointer cursor over a result card')
  await shot(page, '3-hover')

  const placement = await page.evaluate(() => {
    const rect = (sel) => { const r = document.querySelector(sel).getBoundingClientRect(); return { left: r.left, top: r.top, right: r.right, bottom: r.bottom } }
    return { card: rect('.card'), ring: rect('.connector-ring'), line: rect('.connector'), lifted: window.__vidsearch.card(0).box }
  })
  const { card, ring, line, lifted: box } = placement
  const ringAt = { x: (ring.left + ring.right) / 2, y: (ring.top + ring.bottom) / 2 }
  const anchors = [
    { x: (box.left + box.right) / 2, y: box.top }, { x: box.right, y: (box.top + box.bottom) / 2 },
    { x: box.left, y: (box.top + box.bottom) / 2 }, { x: (box.left + box.right) / 2, y: box.bottom },
  ]
  const overlaps = card.left < box.right && card.right > box.left && card.top < box.bottom && card.bottom > box.top
  const touches = (a, b) => a.left <= b.right + 2 && a.right >= b.left - 2 && a.top <= b.bottom + 2 && a.bottom >= b.top - 2
  check(!overlaps, `hover card does not cover the lifted card (card ${JSON.stringify(card)}, lifted ${JSON.stringify(box)})`)
  check(anchors.some((a) => Math.hypot(a.x - ringAt.x, a.y - ringAt.y) < 3) && touches(line, card),
    `dotted connector runs from the lifted card's edge (ring at ${ringAt.x.toFixed(0)},${ringAt.y.toFixed(0)}) to the hover card`)
  await page.mouse.move(card0.center.x + 8, card0.center.y + 4, { steps: 3 })
  await page.waitForTimeout(300)
  const cardAfter = await page.evaluate(() => document.querySelector('.card').getBoundingClientRect().toJSON())
  check(Math.abs(cardAfter.left - card.left) < 0.5 && Math.abs(cardAfter.top - card.top) < 0.5, 'hover card stays put while the cursor moves within the result (not glued to the cursor)')

  await page.mouse.move(VIEWPORT.width / 2, 60)
  await page.waitForTimeout(200)
  check(await page.evaluate(() => getComputedStyle(document.querySelector('.stage')).cursor) === 'crosshair', 'crosshair cursor over empty canvas')
  await page.mouse.move(5, 5)
  const beforeStep = await view()
  await page.keyboard.press('ArrowDown')
  await page.waitForTimeout(300)
  const step1 = { count: await statusText(), label: await page.textContent('.card-label'), view: await view() }
  check(step1.count === `1/${results.length}` && step1.label === `${top.title} · ${clock(top.start)}`, `ArrowDown shows result 1's card as if hovered: ${step1.count}, "${step1.label}"`)
  check(Math.abs(step1.view.x - beforeStep.x) < 1e-6 && Math.abs(step1.view.y - beforeStep.y) < 1e-6, 'stepping to an on-screen result does not move the camera')
  await page.keyboard.press('ArrowDown')
  await page.waitForTimeout(300)
  check(await statusText() === `2/${results.length}`, `second ArrowDown advances to ${await statusText()}`)

  const far = library.find((f) => f.id !== results[2].film_id) ?? library[0]
  await page.evaluate(([id]) => window.__vidsearch.focusFrame(id, 0, 600), [far.id])
  await settle()
  const travel = await page.evaluate(() => new Promise((done) => {
    const trace = []
    const tick = (now) => {
      trace.push({ ...window.__vidsearch.view(), now })
      if (trace.length < 5 || (trace.length < 150 && !window.__vidsearch.settled())) requestAnimationFrame(tick); else done(trace)
    }
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown' }))
    requestAnimationFrame(tick)
  }))
  const moved = Math.hypot(travel.at(-1).x - travel[0].x, travel.at(-1).y - travel[0].y)
  const speeds = travel.slice(2).map((v, i) => Math.hypot(v.x - travel[i + 1].x, v.y - travel[i + 1].y) / Math.max(1, v.now - travel[i + 1].now))
  const jumps = speeds.filter((sp, i) => i > 0 && sp > speeds[i - 1] * 1.5 + 1e-4).length
  console.log(`travel to off-screen result 3: ${travel.length} frames over ${(travel.at(-1).now - travel[0].now).toFixed(0)} ms, ${moved.toFixed(1)} units, peak speed ${(Math.max(...speeds) * 1000).toFixed(1)} units/s`)
  check(moved > 1 && jumps === 0, `stepping to an off-screen result flies there smoothly (${moved.toFixed(1)} units, ${jumps} speed spikes)`)

  await page.evaluate(() => window.__vidsearch.showOverview())
  await settle()
  const f = await measureFps(page)
  console.log(`${library.length} films, wheel-panning 3 s (headless, ${renderer}): ${f.fps.toFixed(1)} fps, frame interval p50 ${f.p50} ms, p95 ${f.p95} ms, max ${f.max} ms`)

  check(f.fps > 55, `${library.length} films hold 60 fps while panning (${f.fps.toFixed(1)})`)

  const slow = await browser.newPage({ viewport: VIEWPORT })
  slow.on('pageerror', (e) => failures.push(`pageerror (slow library): ${e.message}`))
  await slow.addInitScript(instrument)
  await slow.route('**/api/library', async (r) => { await new Promise((ok) => setTimeout(ok, 4000)); await r.continue().catch(() => {}) })
  await slow.goto(BASE)
  await slow.waitForTimeout(500)
  const early = { visible: await slow.isVisible('.loader'), line: await slow.textContent('.loader-line'), disabled: await slow.isDisabled(INPUT), placeholder: await slow.getAttribute(INPUT, 'placeholder') }
  check(early.visible && early.line === 'loading the archive' && early.disabled && early.placeholder === WAITING_PLACEHOLDER,
    `while /api/library is slow the loader shows "${early.line}" and the pill is disabled ("${early.placeholder}")`)
  await slow.waitForTimeout(3000)
  const later3 = await slow.textContent('.loader-line')
  await shot(slow, '1-loading')
  check(later3 === 'waking up the server', `after 3 s the loader says "${later3}"`)
  await ready(slow)
  const slowLines = await loaderLines(slow)
  check(slowLines[0] === 'loading the archive' && slowLines[1] === 'waking up the server' && /^loading frames/.test(slowLines[2] ?? '') && await slow.textContent('.masthead-stats') === libraryStats(library),
    `slow library ends ready: ${slowLines.join(' | ')}`)
  await slow.close()

  const flaky = await browser.newPage({ viewport: VIEWPORT })
  flaky.on('pageerror', (e) => failures.push(`pageerror (flaky library): ${e.message}`))
  await flaky.addInitScript(instrument)
  let flakyCalls = 0
  await flaky.route('**/api/library', (r) => (++flakyCalls <= 2 ? r.fulfill({ status: 503, body: 'cold' }) : r.continue()))
  const flakyStart = Date.now()
  await flaky.goto(BASE)
  await ready(flaky)
  const flakyLines = await loaderLines(flaky)
  check(flakyCalls === 3 && !flakyLines.some((l) => /did not answer/.test(l)) && await flaky.textContent('.masthead-stats') === libraryStats(library),
    `library failing twice with 503 retries on its own and ends ready after ${((Date.now() - flakyStart) / 1000).toFixed(1)} s (${flakyCalls} requests)`)
  await flaky.close()

  const cold = await browser.newPage({ viewport: VIEWPORT })
  cold.on('pageerror', (e) => failures.push(`pageerror (cold library): ${e.message}`))
  await cold.clock.install()
  let coldLibraryCalls = 0
  await cold.route('**/api/library', (r) => { coldLibraryCalls++; return r.fulfill({ status: 503, body: 'cold' }) })
  await cold.goto(BASE)
  await cold.clock.fastForward(10_500)
  await cold.waitForTimeout(400)
  const stage3 = await cold.textContent('.loader-line')
  check(/^still starting/.test(stage3), `after 10 s the loader says "${stage3}"`)
  for (let i = 0; i < 20 && !(await cold.isVisible('.loader-retry')); i++) {
    await cold.clock.fastForward(4000)
    await cold.waitForTimeout(100)
  }
  const gaveUp = { line: await cold.textContent('.loader-line'), detail: await cold.textContent('.loader-detail').catch(() => null), disabled: await cold.isDisabled(INPUT) }
  check(gaveUp.line === 'the archive did not answer' && /HTTP 503/.test(gaveUp.detail) && gaveUp.disabled && coldLibraryCalls >= 8,
    `after ~60 s of 503s (${coldLibraryCalls} requests) the loader gives up with a retry button: "${gaveUp.line}" / "${gaveUp.detail}"`)
  await cold.unroute('**/api/library')
  await cold.click('.loader-retry')
  await ready(cold)
  check(await cold.textContent('.masthead-stats') === libraryStats(library) && !(await cold.isDisabled(INPUT)), 'retry after the hard error loads the archive')
  await cold.close()

  const broken = await browser.newPage({ viewport: VIEWPORT })
  // r3f reports errors its error boundaries caught through window.reportError, so each failed load is also a page error.
  const atlasErrors = []
  broken.on('pageerror', (e) => (/^Could not load .*\/atlas\//.test(e.message) ? atlasErrors.push(e.message) : failures.push(`pageerror (failing atlas): ${e.message}`)))
  const [dead, flakyFilm] = [library[0].id, library[1].id]
  const atlasCalls = {}
  await broken.route('**/media/*/atlas/*', (r) => {
    const film = new URL(r.request().url()).pathname.split('/')[2]
    atlasCalls[film] = (atlasCalls[film] ?? 0) + 1
    return film === dead || (film === flakyFilm && atlasCalls[film] === 1) ? r.fulfill({ status: 503, body: 'cold' }) : r.continue()
  })
  await broken.goto(BASE)
  await ready(broken, 30000)
  await broken.waitForTimeout(1200)
  const lumaNear = async (id) => { const p = await broken.evaluate((f) => window.__vidsearch.screenOf(f, 20), id); return (await tone(broken, { x: p.x - 12, y: p.y - 8, width: 24, height: 16 })).lum }
  const brokenState = { dead: await lumaNear(dead), flaky: await lumaNear(flakyFilm), enabled: !(await broken.isDisabled(INPUT)) }
  check(atlasCalls[dead] === 4 && atlasCalls[flakyFilm] === 2 && atlasErrors.length === 5 && brokenState.enabled && brokenState.flaky > 15 && brokenState.dead < 10,
    `atlas page failing once is retried and draws (luma ${brokenState.flaky}); one failing 4 times is skipped (luma ${brokenState.dead}) and the archive still opens (${JSON.stringify(atlasCalls)})`)
  await broken.close()

  const page3 = await browser.newPage({ viewport: VIEWPORT })
  page3.on('pageerror', (e) => failures.push(`pageerror (up to 3 films): ${e.message}`))
  await page3.route('**/api/library', async (r) => {
    const body = await (await r.fetch()).json()
    await r.fulfill({ json: { ...body, films: body.films.slice(0, 3) } })
  })
  await page3.goto(BASE)
  await page3.waitForFunction(() => window.__vidsearch?.settled())
  await ready(page3)
  await page3.waitForTimeout(2500)
  await shot(page3, '6-overview-small')
  const pill3 = await page3.locator('form.pill').boundingBox()
  const films3 = await page3.evaluate(async () => (await (await fetch('/api/library')).json()).films)
  const ends3 = await page3.evaluate((films) => films.map((f) => [window.__vidsearch.screenOf(f.id, 0), window.__vidsearch.screenOf(f.id, f.n_frames - 1)]), films3)
  const inView3 = (p) => p && p.x > 0 && p.x < VIEWPORT.width && p.y > 0 && p.y < pill3.y
  check(films3.length === Math.min(3, library.length) && ends3.flat().every(inView3), `${films3.length}-film library (the first films) fits the overview above the pill`)
  const near3 = await page3.evaluate((films) => films.map((f) => window.__vidsearch.screenOf(f.id, 20)), films3)
  const lum3 = []
  for (const p of near3) lum3.push((await tone(page3, { x: p.x - 12, y: p.y - 8, width: 24, height: 16 })).lum)
  check(lum3.every((l) => l > 15), `every film in the ${films3.length}-film library draws (luma near frame 20: ${lum3.join(', ')}; background ~5)`)
}

/**
 * 30 s of the library's first video (looped if shorter) re-encoded at a constant 6 Mbit/s, so it is three 8 MiB parts,
 * and a sparse file just over the 2 GB limit.
 */
async function uploadFixtures() {
  const clip = join(outDir, 'upload-clip.mp4')
  if (!existsSync(clip)) {
    const source = join(outDir, 'upload-source')
    const video = await fetch(new URL(LIBRARY.films[0].video_url, BASE))
    if (!video.ok) throw new Error(`GET ${LIBRARY.films[0].video_url}: HTTP ${video.status}`)
    writeFileSync(source, Buffer.from(await video.arrayBuffer()))
    const r = spawnSync('ffmpeg', ['-y', '-loglevel', 'error', '-stream_loop', '-1', '-i', source, '-t', '30', '-an', '-c:v', 'libx264', '-preset', 'veryfast',
      '-b:v', '6M', '-minrate', '6M', '-maxrate', '6M', '-bufsize', '12M', '-x264-params', 'nal-hrd=cbr', clip], { stdio: 'inherit' })
    if (r.status !== 0) throw new Error(`ffmpeg failed making ${clip}`)
  }
  const oversize = join(outDir, 'upload-oversize.mp4')
  const fd = openSync(oversize, 'w')
  ftruncateSync(fd, 2 * 1024 ** 3 + 1024 ** 2)
  closeSync(fd)
  return { clip, oversize }
}

async function uploadChecks() {
  const mode = (await (await fetch(new URL('/api/library', BASE))).json()).uploads?.mode
  console.log(`uploads mode from /api/library: ${mode}`)
  if (mode === 'device') return deviceChecks()
  if (mode === 'account') return accountChecks()
  check(false, `the API offers uploads (mode "${mode}"); start it with VS_ANON_UPLOADS=1 or VS_DEV_AUTH=1`)
}

/** A fresh browser profile on the loaded archive. `devUser` picks the dev-auth user ('' = signed out). */
async function openPage(contexts, name, { devUser, prepare, reducedMotion } = {}) {
  const ctx = await browser.newContext({ viewport: VIEWPORT, reducedMotion })
  contexts.push(ctx)
  if (devUser !== undefined) await ctx.addInitScript((u) => localStorage.setItem('vidsearch.dev-user', u), devUser)
  const page = await ctx.newPage()
  page.on('pageerror', (e) => failures.push(`pageerror (${name}): ${e.message}`))
  page.on('console', (m) => { if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) console.log(`console.error (${name}):`, m.text()) })
  if (prepare) await prepare(page)
  await page.goto(BASE)
  await page.waitForFunction(() => window.__vidsearch?.settled())
  await ready(page)
  return page
}

/** One API call from the page with `bearer` (null = anonymous); { status, body }. */
function apiAs(page, bearer, path, { method = 'GET', json } = {}) {
  return page.evaluate(async ([b, p, m, j]) => {
    const headers = { ...(b ? { Authorization: `Bearer ${b}` } : {}), ...(j ? { 'Content-Type': 'application/json' } : {}) }
    const r = await fetch(p, { method: m, headers, body: j ? JSON.stringify(j) : undefined })
    return { status: r.status, body: await r.json().catch(() => null) }
  }, [bearer, path, method, json ?? null])
}

async function uiSearch(page, q) {
  const res = page.waitForResponse((r) => r.url().includes('/api/search?'))
  await page.fill(INPUT, q)
  await page.press(INPUT, 'Enter')
  const r = await res
  return { auth: (await r.request().allHeaders()).authorization ?? null, body: await r.json() }
}

async function clickIfShown(page, selector) {
  const el = await page.$(selector)
  if (el) await el.click()
}

async function deviceToken(page) {
  const id = await page.evaluate(() => localStorage.getItem('vidsearch.device'))
  return id && `device:${id}`
}

/**
 * The upload journey both modes share, for the identity `me` ({ bearer, openPanel(page, 'upload' | 'my uploads'), note,
 * afterUpload?(page, filmId) }) on `page`, whose upload panel is already open on a fresh quota. `openOther()` returns
 * another identity's page (reduced motion) and bearer, opened after the film exists. Returns the film id, deleted again by the end.
 */
async function uploadJourney(page, me, openOther, { clip, oversize, run }) {
  const clipBytes = statSync(clip).size
  await page.waitForSelector(`${METER}[aria-label$="left today"]`)
  const quota0 = await quotaOf(page)
  check(quota0.label === '3 of 3 uploads left today' && quota0.dots === 3 && quota0.used === 0, `the panel head's quota meter: "${quota0.label}", ${quota0.used} of ${quota0.dots} dots filled`)
  check(await page.getAttribute('input[name=title]', 'placeholder') === 'defaults to the file name' && await page.isDisabled('.upload-submit'),
    'with no source chosen the title says it defaults to the file name and upload is disabled')
  await shots(page, '1-upload-tab')

  let startCalls = 0
  page.on('request', (r) => { if (r.method() === 'POST' && new URL(r.url()).pathname === '/api/uploads') startCalls++ })
  await page.setInputFiles('.drop input[type=file]', oversize)
  const refusal = await page.textContent('.upload-error').catch(() => null)
  check(/^this file is 2\.0 GB; the limit is 2\.0 GB$/.test(refusal ?? '') && await page.isDisabled('.upload-submit') && startCalls === 0,
    `a file over 2 GB is refused before any request: "${refusal}"`)

  const parts = {}
  let inFlight = 0, peak = 0
  await page.route('**/api/uploads/*/parts/*', async (r) => {
    const n = r.request().url().split('/').pop()
    parts[n] = (parts[n] ?? 0) + 1
    peak = Math.max(peak, ++inFlight)
    await new Promise((ok) => setTimeout(ok, 1500))
    inFlight--
    if (n === '1' && parts[n] === 1) return r.fulfill({ status: 500, body: 'injected failure' })
    await r.continue().catch(() => {})
  })
  await page.setInputFiles('.drop input[type=file]', clip)
  const autoTitle = { value: await page.inputValue('input[name=title]'), placeholder: await page.getAttribute('input[name=title]', 'placeholder') }
  check(autoTitle.value === '' && autoTitle.placeholder === 'upload-clip' && !(await page.$('.upload-error')) && !(await page.isDisabled('.upload-submit')),
    `choosing the clip clears the refusal, enables upload and offers its file name as the default title ("${autoTitle.placeholder}")`)
  const title = `upload clip ${run}`
  await page.fill('input[name=title]', title)
  const viewBefore = await page.evaluate(() => window.__vidsearch.view())
  await page.evaluate(() => {
    window.__progress = []
    window.__progressTimer = setInterval(() => {
      const bar = document.querySelector('.journey .bar')?.getAttribute('aria-valuenow')
      const chip = document.querySelector('.chip .percent')?.textContent
      window.__progress.push({ t: Math.round(performance.now()), bar: bar == null ? null : +bar, chip: chip ? parseInt(chip, 10) : null })
    }, 100)
  })
  await page.click('.upload-submit')
  await page.waitForSelector('.journey .bar')
  const current = '.stages li[data-state=current]'
  await page.waitForFunction((sel) => /^[1-9][\d.]* MB of/.test(document.querySelector(`${sel} .stage-count`)?.textContent ?? ''), current, { timeout: 15000 }).catch(() => {})
  await page.waitForTimeout(300)
  const sending = {
    step: await page.textContent(`${current} .stage-name`).catch(() => null),
    bytes: await page.textContent(`${current} .stage-count`).catch(() => null),
    chip: await page.textContent('.chip-text').catch(() => null),
  }
  await shots(page, '2-uploading')
  check(sending.step === 'uploading' && (sending.bytes ?? '').includes(` of ${(clipBytes / 1024 ** 2).toFixed(1)} MB`) && /^uploading \d+%$/.test(sending.chip ?? ''),
    `while sending, the "${sending.step}" step reads "${sending.bytes}", chip "${sending.chip}"`)
  await page.waitForFunction(() => document.querySelector('.stages li')?.dataset.state === 'done' || !!document.querySelector('.journey-done'), null, { timeout: 60000 })
  check(JSON.stringify(parts) === JSON.stringify({ 0: 1, 1: 2, 2: 1 }) && peak === 3,
    `three 8 MiB parts of ${clipBytes} bytes sent 3 at a time (peak ${peak}); part 1 failed once with 500 and was retried (${JSON.stringify(parts)})`)
  await page.unroute('**/api/uploads/*/parts/*')

  await page.click('.panel-close')
  await page.waitForTimeout(POLL)
  const chipClosed = await page.textContent('.chip-text').catch(() => null)
  check(!!chipClosed && !(await page.$('.panel')), `with the panel closed the job keeps running behind a chip: "${chipClosed}"`)
  await page.click('.chip')
  await page.waitForSelector('.panel.upload .journey')

  const seen = new Set()
  let shotTaken = false, shotAtProgress = false, marks = null
  const deadline = Date.now() + 10 * 60_000
  while (Date.now() < deadline && !(await page.$('.journey-done, .panel.upload .upload-error'))) {
    const cur = await page.$$eval('.stages li', (lis) => lis.map((li) => ({
      name: li.querySelector('.stage-name').textContent, state: li.dataset.state, count: li.querySelector('.stage-count')?.textContent ?? null,
      check: !!li.querySelector('.check'), spinner: !!li.querySelector('.spinner'),
    }))).catch(() => [])
    const at = cur.find((s) => s.state === 'current')
    if (at) {
      seen.add(at.name)
      if (!marks && cur.some((s) => s.state === 'done') && cur.some((s) => s.state === 'todo')) marks = cur
      if (at.count && !shotAtProgress && (!shotTaken || !at.count.startsWith('0 '))) {
        await shots(page, '3-processing')
        shotTaken = true
        shotAtProgress = !at.count.startsWith('0 ')
        console.log(`processing shot at ${at.name} ${at.count}`)
      }
    }
    await page.waitForTimeout(250)
  }
  if (!shotTaken) await shots(page, '3-processing')
  await page.waitForTimeout(400)
  const progress = await page.evaluate(() => { clearInterval(window.__progressTimer); return window.__progress })
  const dwell = progress.filter((p) => p.bar !== null).reduce((m, p, i, a) => (i && a[i - 1].bar === p.bar ? m : [...m, { bar: p.bar, t: p.t }]), [])
  console.log(`bar values over time: ${dwell.map((d, i) => `${d.bar}@${((d.t - dwell[0].t) / 1000).toFixed(1)}s`).join(' ')}`)
  const bars = progress.map((p) => p.bar).filter((v) => v !== null)
  const chips = progress.map((p) => p.chip).filter((v) => v !== null)
  const backwards = (xs) => xs.filter((v, i) => i > 0 && v < xs[i - 1]).length
  const regressions = progress.flatMap((p, i) => (i && p.chip !== null && progress[i - 1].chip !== null && p.chip < progress[i - 1].chip ? [`${progress[i - 1].chip}->${p.chip}@${p.t - progress[0].t}ms`] : []))
  if (regressions.length) console.log(`chip went back: ${regressions.join(', ')}; samples ${JSON.stringify(progress.map((p) => [p.t - progress[0].t, p.bar, p.chip]))}`)
  const distinct = (xs) => [...new Set(xs)]
  console.log(`overall progress over ${progress.length} samples: bar ${distinct(bars).join(' ')} | chip ${distinct(chips).join(' ')}`)
  check(bars.length > 20 && backwards(bars) === 0 && bars.at(-1) === 100 && distinct(bars).length >= 4 && backwards(chips) === 0 && distinct(chips).length >= 4,
    `overall progress only moves forward through the whole upload: bar ${bars[0]} -> ${bars.at(-1)} over ${distinct(bars).length} values, chip ${chips[0]}% -> ${chips.at(-1)}% (${backwards(bars)} + ${backwards(chips)} steps back)`)
  const marksOk = !!marks && marks.every((s) => (s.state === 'done' ? s.check && !s.spinner : s.state === 'current' ? s.spinner && !s.check : !s.check && !s.spinner))
  check(marksOk && marks.some((s) => s.state === 'done') && marks.some((s) => s.state === 'todo'),
    `while processing, done steps carry a check, the current one a spinner, upcoming ones neither (${marks?.map((s) => `${s.name}:${s.state}`).join(', ')})`)
  const doneText = await page.textContent('.job-done').catch(() => null)
  console.log('stages seen while polling:', [...seen].join(' → '))
  // A warm model embeds a 30 s clip between two 1.5 s polls, so which stages the panel catches live varies.
  check(doneText === 'in the archive' && !(await page.$('.panel.upload .stages')) && await page.isVisible('.journey-done button:text-is("view in archive")') && seen.size >= 2,
    `the job's stages advance live (${seen.size} seen), then collapse to one line: "${doneText}" with "view in archive"`)
  check(!(await page.$('.chip')), 'the status chip goes away once nothing is running')

  const jobs = (await apiAs(page, me.bearer, '/api/jobs')).body
  const job = jobs.jobs.find((j) => j.title === title)
  const filmId = job?.film_id
  await page.waitForFunction((id) => !!window.__vidsearch.ownerMark(id), filmId, { timeout: 15000 }).catch(() => {})
  await page.waitForFunction(() => window.__vidsearch.settled(), null, { timeout: 15000 })
  const viewDone = await page.evaluate(() => window.__vidsearch.view())
  check(viewDone.dist < viewBefore.dist * 0.6, `when the job is done the camera eases toward the new strip on its own (dist ${viewBefore.dist.toFixed(1)} -> ${viewDone.dist.toFixed(1)})`)
  await shots(page, '4-done')

  const toStrip = async (what, click) => {
    await page.evaluate(() => window.__vidsearch.showOverview())
    await page.waitForTimeout(100)
    await page.waitForFunction(() => window.__vidsearch.settled(), null, { timeout: 15000 })
    const before = await page.evaluate(() => window.__vidsearch.view())
    await click()
    const closing = !!(await page.$('.panel.closing'))
    await page.waitForSelector('.panel', { state: 'detached', timeout: 2000 })
    await page.waitForTimeout(100)
    await page.waitForFunction(() => window.__vidsearch.settled(), null, { timeout: 15000 })
    await page.waitForTimeout(400)
    const after = await page.evaluate(() => window.__vidsearch.view())
    const n = (await apiAs(page, me.bearer, '/api/library')).body.films.find((f) => f.id === filmId)?.n_frames ?? 0
    const mid = await page.evaluate(([id, n]) => window.__vidsearch.screenOf(id, Math.floor(n / 2)), [filmId, n])
    check(closing && after.dist < before.dist * 0.6 && !!mid && Math.abs(mid.x - VIEWPORT.width / 2) < 120 && Math.abs(mid.y - VIEWPORT.height / 2) < 160,
      `${what} fades the panel out and eases the camera to the strip (dist ${before.dist.toFixed(1)} -> ${after.dist.toFixed(1)}; strip middle at ${mid && `${mid.x.toFixed(0)},${mid.y.toFixed(0)}`})`)
  }
  await toStrip('"view in archive"', () => page.click('.journey-done button:text-is("view in archive")'))
  const mark = await page.evaluate((id) => window.__vidsearch.ownerMark(id), filmId)
  const markTone = mark && await tone(page, { x: mark.left, y: mark.top, width: Math.max(2, mark.right - mark.left), height: Math.max(2, mark.bottom - mark.top) })
  await shots(page, 'new-strip')
  const lib = (await apiAs(page, me.bearer, '/api/library')).body.films
  const mine = lib.find((f) => f.id === filmId)
  check(!!mine?.owner_is_you && lib.filter((f) => f.owner_is_you).length === 1, `the new film ${filmId} is in ${me.bearer}'s library, marked owner_is_you`)
  check(!!mark && markTone.lilac > 0.5 * (mark.right - mark.left) * (mark.bottom - mark.top), `its strip carries the lilac ownership mark (${markTone?.lilac} lilac px in ${JSON.stringify(mark && { w: Math.round(mark.right - mark.left), h: Math.round(mark.bottom - mark.top) })})`)
  const others = await page.evaluate((ids) => ids.map((id) => window.__vidsearch.ownerMark(id)), lib.filter((f) => !f.owner_is_you).map((f) => f.id))
  check(others.every((m) => m === null), 'public films carry no ownership mark')

  const probe = (await apiAs(page, me.bearer, `/api/search?q=${encodeURIComponent(QUERY)}&k=3&films=${filmId}`)).body.results
  const query = probe[0]?.caption ?? QUERY
  console.log(`searching for a moment of the clip: "${query}"`)
  const found = await uiSearch(page, query)
  const hit = found.body.results.find((r) => r.film_id === filmId)
  check(found.auth === `Bearer ${me.bearer}` && !!hit && found.body.heatmap.films[filmId] !== undefined,
    `the uploader's search finds a moment in the upload (rank ${hit?.rank}, ${hit && clock(hit.t)}), sending its bearer token`)
  if (me.afterUpload) await me.afterUpload(page, filmId)

  const other = await openOther()
  const lib2 = (await apiAs(other.page, other.bearer, '/api/library')).body.films
  const anon = (await apiAs(other.page, null, '/api/library')).body.films
  const found2 = await uiSearch(other.page, query)
  const job2 = await apiAs(other.page, other.bearer, `/api/jobs/${job.id}`)
  check(!lib2.some((f) => f.id === filmId) && !anon.some((f) => f.id === filmId) && !(await other.page.evaluate((id) => window.__vidsearch.screenOf(id, 0), filmId)),
    `another identity (${other.bearer}) and anonymous visitors don't get the film in the library or on the plane`)
  check(found2.auth === `Bearer ${other.bearer}` && found2.body.results.length > 0 && !found2.body.results.some((r) => r.film_id === filmId) && found2.body.heatmap.films[filmId] === undefined && job2.status === 403,
    `another identity's search for the same words returns ${found2.body.results.length} public results and none from the upload; its job is 403 to them`)
  await me.openPanel(other.page, 'my uploads')
  const still = await other.page.evaluate(() => {
    const p = document.querySelector('.panel')
    return { animations: p.getAnimations().length, name: getComputedStyle(p).animationName }
  })
  await other.page.click('.panel-close')
  const goneAtOnce = !(await other.page.$('.panel'))
  check(still.animations === 0 && still.name === 'none' && goneAtOnce, `with prefers-reduced-motion the panel opens and closes without animating (${JSON.stringify(still)}, gone at once: ${goneAtOnce})`)

  await page.fill(INPUT, '')
  await page.press(INPUT, 'Enter')
  await me.openPanel(page, 'upload')
  const opening = await page.evaluate(() => document.querySelector('.panel').getAnimations().map((a) => a.animationName))
  check(opening.includes('panel-in'), `opening the panel plays its fade-and-rise (${JSON.stringify(opening)})`)
  await clickIfShown(page, 'button:text-is("upload another")')  // after a reload the panel opens on the form
  await page.fill('input[name=url]', 'not a link')
  await page.fill('input[name=title]', 'bad link')
  await page.click('.upload-submit')
  await page.waitForSelector('.panel.upload .upload-error')
  const urlError = await page.textContent('.panel.upload .upload-error')
  check(urlError === 'url must be an http(s) link to a video file', `a URL upload of a non-http string shows the API's error: "${urlError}"`)
  await page.click('.upload-submit:text-is("try again")')
  await page.waitForSelector(`${METER}[aria-label^="2 of 3"]`, { timeout: 5000 }).catch(() => {})
  const quota1 = await quotaOf(page)
  check(quota1.label === '2 of 3 uploads left today' && quota1.used === 1, `the meter follows /api/jobs: "${quota1.label}", ${quota1.used} dot filled`)

  const box = () => page.evaluate(() => { const r = document.querySelector('.panel').getBoundingClientRect(); return { top: r.top, left: r.left, right: r.right } })
  await page.waitForTimeout(600)
  const boxUpload = await box()
  const heights = page.evaluate(() => new Promise((done) => {
    const hs = []
    const tick = () => { hs.push(document.querySelector('.panel-body').getBoundingClientRect().height); if (hs.length < 40) requestAnimationFrame(tick); else done(hs) }
    requestAnimationFrame(tick)
  }))
  await page.click('.tab:text-is("my uploads")')
  const hs = await heights
  const boxMine = await box()
  const [h0, h1] = [hs[0], hs.at(-1)]
  const between = hs.filter((h) => h < Math.max(h0, h1) - 2 && h > Math.min(h0, h1) + 2).length
  check(Math.abs(boxUpload.top - boxMine.top) < 0.5 && Math.abs(boxUpload.left - boxMine.left) < 0.5 && Math.abs(boxUpload.right - boxMine.right) < 0.5,
    `switching tabs keeps the panel in place (top/left ${boxUpload.top.toFixed(1)},${boxUpload.left.toFixed(1)} -> ${boxMine.top.toFixed(1)},${boxMine.left.toFixed(1)})`)
  check(Math.abs(h0 - h1) > 40 && between >= 3, `and eases its height (${h0.toFixed(0)} -> ${h1.toFixed(0)} px through ${between} frames in between)`)

  const row = `.mine-list li[data-film="${filmId}"]`
  await page.waitForSelector(row)
  await page.waitForSelector(`${row} .thumb img[data-loaded]`, { timeout: 10000 }).catch(() => {})
  const film = lib.find((f) => f.id === filmId)
  const runtime = `${Math.floor(Math.round(film.duration) / 60)}:${String(Math.round(film.duration) % 60).padStart(2, '0')}`
  const rowInfo = await page.$eval(row, (li) => {
    const img = li.querySelector('.thumb img')
    return { title: li.querySelector('.row-title').textContent, meta: li.querySelector('.row-meta').textContent, thumb: img && { w: img.naturalWidth, src: new URL(img.src).pathname, done: img.complete } }
  })
  check(rowInfo.title === title && rowInfo.meta.startsWith(`${runtime} · `) && /· (just now|\d+ min ago)$/.test(rowInfo.meta),
    `my uploads lists "${rowInfo.title}" with its duration and a relative time: "${rowInfo.meta}"`)
  check(!!rowInfo.thumb?.done && rowInfo.thumb.w > 0 && rowInfo.thumb.src.startsWith(`/media/${filmId}/frames/`), `its thumbnail is a frame still that loaded (${rowInfo.thumb?.src}, naturalWidth ${rowInfo.thumb?.w})`)
  const foot = await page.$eval('.panel', (p) => { const n = p.querySelector('.panel-foot .browser-note'); return n && { text: n.textContent, title: n.getAttribute('title') } })
  check(me.note ? foot?.text === 'saved to this browser' && foot.title === me.note : foot === null,
    me.note ? `the footer says "${foot?.text}" with the full note in its tooltip ("${foot?.title}")` : 'account mode has no browser note')
  const caps = await page.evaluate(() => [...document.querySelectorAll('.panel *, .masthead *, .account-bar *')]
    .filter((el) => { const s = getComputedStyle(el); return s.textTransform === 'uppercase' || s.fontVariantCaps !== 'normal' }).map((el) => el.className))
  check(caps.length === 0, `no text in the masthead, corner or panel is set in capitals (${JSON.stringify(caps)})`)
  await page.hover(row)
  await shots(page, '5-my-uploads')

  await toStrip('clicking the ready row', () => page.click(`${row} .row-open`))

  await me.openPanel(page, 'my uploads')
  let deletes = 0
  const onDelete = (r) => { if (r.method() === 'DELETE' && new URL(r.url()).pathname.startsWith('/api/films/')) deletes++ }
  page.on('request', onDelete)
  await page.hover(row)
  await page.click(`${row} .row-delete`)
  const confirm = await page.textContent(`${row} .row-confirm`)
  await page.click(`${row} .confirm-no`)
  const kept = !!(await page.$(`${row} .row-delete`)) && deletes === 0
  await page.hover(row)
  await page.click(`${row} .row-delete`)
  await page.click(`${row} .confirm-yes`)
  const leaving = await page.waitForSelector(`${row}[data-leaving]`, { timeout: 10000 }).then(() => true, () => false)
  await page.waitForSelector(row, { state: 'detached', timeout: 10000 })
  page.off('request', onDelete)
  check(confirm === 'delete?yes·no' && kept, `delete asks inline first ("${confirm}": "delete?", "yes", "·", "no"); "no" keeps the row and sends nothing`)
  await page.waitForFunction((id) => !window.__vidsearch.screenOf(id, 0), filmId, { timeout: 15000 }).catch(() => {})
  const libAfter = (await apiAs(page, me.bearer, '/api/library')).body.films
  const gone = { leaving, deletes, inLibrary: libAfter.some((f) => f.id === filmId), onPlane: !!(await page.evaluate((id) => window.__vidsearch.screenOf(id, 0), filmId)) }
  check(gone.leaving && gone.deletes === 1 && !gone.inLibrary && !gone.onPlane,
    `confirming deletes the film from the API library and its strip from the plane, and the row collapses away (${JSON.stringify(gone)})`)

  await page.waitForSelector('.mine-empty', { timeout: 10000 }).catch(() => {})
  const empty = await page.textContent('.mine-empty p').catch(() => null)
  const emptyButton = await page.textContent('.mine-empty button').catch(() => null)
  await shots(page, '6-empty')
  await page.click('.mine-empty button')
  const switched = await page.$eval('.tab[aria-selected=true]', (t) => t.textContent).catch(() => null)
  check(empty === 'nothing uploaded yet' && emptyButton === `upload a ${IDENTITY.noun}` && switched === 'upload' && !!(await page.$('.panel.upload .upload-form')),
    `the empty list says "${empty}" and its "${emptyButton}" button switches to the ${switched} tab`)
  return filmId
}

async function accountChecks() {
  const run = Date.now().toString(36)
  const me = `tester-${run}`, other = `other-${run}`
  const fixtures = { ...(await uploadFixtures()), run }
  const contexts = []
  const menu = async (page, item) => {
    if (!(await page.$('.menu'))) await page.click('.account-avatar')
    await page.click(`.menu button:text-is("${item}")`)
  }

  const out = await openPage(contexts, 'signed out', { devUser: '' })
  await out.waitForTimeout(800)
  await shot(out, '1-signed-out')
  check(await out.textContent('.account-signin') === 'sign in' && !(await out.$('.account-avatar')) && !(await out.$('.panel')) && !(await out.$('.chip')),
    'signed out: top-right shows "sign in", no avatar, no upload')
  await out.click('.account-signin')
  await out.click('.account-avatar')
  check(await out.textContent('.menu-who') === DEV_USER, `"sign in" with dev auth signs in as VITE_DEV_AUTH (${await out.textContent('.menu-who')})`)

  const page = await openPage(contexts, me, { devUser: me })
  await page.click('.account-avatar')
  const items = await page.$$eval('.menu button', (bs) => bs.map((b) => b.textContent))
  check(JSON.stringify(items) === JSON.stringify(['upload', 'my uploads', 'sign out']), `signed in: the avatar menu lists ${JSON.stringify(items)}`)
  await menu(page, 'upload')
  await uploadJourney(page, { bearer: `dev:${me}`, openPanel: menu, note: null },
    async () => ({ page: await openPage(contexts, other, { devUser: other, reducedMotion: 'reduce' }), bearer: `dev:${other}` }), fixtures)

  let jobsCalls = 0
  const expired = await openPage(contexts, 'expired', { devUser: me, prepare: (p) => p.route('**/api/jobs', (r) => { if (r.request().headers().authorization) jobsCalls++; return r.fulfill({ status: 401, json: { detail: 'invalid token: expired' } }) }) })
  await expired.waitForSelector('.account-notice', { timeout: 10000 }).catch(() => {})
  const notice = await expired.textContent('.account-notice').catch(() => null)
  check(notice === 'your session expired; sign in again' && jobsCalls === 2 && await expired.isVisible('.account-signin'),
    `a 401 that survives one fresh-token retry (${jobsCalls} requests) signs the user out with "${notice}"`)

  for (const ctx of contexts) await ctx.close()
}

async function deviceChecks() {
  const run = Date.now().toString(36)
  const fixtures = { ...(await uploadFixtures()), run }
  const contexts = []
  const corner = (page, item) => page.click(`.account-bar button:text-is("${item}")`)
  const openPanel = async (page, item) => {
    if (!(await page.$(`.panel.${item === 'upload' ? 'upload' : 'mine'}`))) await corner(page, item)
  }

  const page = await openPage(contexts, 'browser A')
  await page.waitForTimeout(800)
  await shot(page, '1-corner')
  const links = await page.$$eval('.account-bar button', (bs) => bs.map((b) => b.textContent))
  check(JSON.stringify(links) === JSON.stringify(['upload', 'my uploads']) && !(await page.$('.account-avatar')) && !(await page.$('.account-signin')) && !(await page.$('.panel')),
    `without sign-in the top-right offers ${JSON.stringify(links)}: no avatar, no sign in`)
  check(await deviceToken(page) === null, 'merely visiting mints no browser id')
  const jobsAuth = page.waitForRequest((r) => new URL(r.url()).pathname === '/api/jobs')
  await corner(page, 'upload')
  const bearer = await deviceToken(page)
  const sent = (await (await jobsAuth).allHeaders()).authorization
  check(DEVICE_TOKEN.test(bearer ?? '') && sent === `Bearer ${bearer}`, `opening upload mints this browser's id and sends it: ${sent}`)

  const openOther = async () => {
    const other = await openPage(contexts, 'browser B', { reducedMotion: 'reduce' })
    await corner(other, 'my uploads')
    await other.waitForSelector('.panel.mine .mine-empty', { timeout: 10000 }).catch(() => {})
    const empty = await other.textContent('.panel.mine .mine-empty p').catch(() => null)
    const otherBearer = await deviceToken(other)
    check(DEVICE_TOKEN.test(otherBearer ?? '') && otherBearer !== bearer && empty === 'nothing uploaded yet',
      `a second browser (fresh storage) gets its own id and an empty "my uploads": "${empty}"`)
    await other.click('.panel-close')
    return { page: other, bearer: otherBearer }
  }
  const afterUpload = async (p, filmId) => {
    const asked = []
    // Finished requests only: StrictMode's mount, unmount, mount aborts the dev server's first library request.
    const onRequest = (r) => { if (new URL(r.url()).pathname === '/api/library') asked.push(r.headers().authorization ?? null) }
    p.on('requestfinished', onRequest)
    await p.reload()
    await p.waitForFunction(() => window.__vidsearch?.settled())
    await ready(p)
    p.off('requestfinished', onRequest)
    const strip = await p.evaluate((id) => !!window.__vidsearch.screenOf(id, 0) && !!window.__vidsearch.ownerMark(id), filmId)
    check(strip && JSON.stringify(asked) === JSON.stringify([null, `Bearer ${bearer}`]),
      `after a reload this browser gets its strip back: the library is asked anonymously, then as the stored id (${JSON.stringify(asked)})`)
  }
  await uploadJourney(page, { bearer, openPanel, note: DEVICE_NOTE, afterUpload }, openOther, fixtures)

  // Cancel mid-upload gives the day's upload back.
  const used = async () => (await apiAs(page, bearer, '/api/jobs')).body.uploads_today
  await openPanel(page, 'upload')
  await clickIfShown(page, '.upload-submit:text-is("try again")')
  await page.waitForSelector(`${METER}[aria-label$="left today"]`)
  const quotaBefore = (await quotaOf(page)).label
  await page.route('**/api/uploads/*/parts/*', () => {})  // parts hang until the cancel aborts them
  await page.setInputFiles('.drop input[type=file]', fixtures.clip)
  await page.fill('input[name=title]', `cancel me ${run}`)
  await page.click('.upload-submit')
  await page.waitForSelector('.journey-cancel')
  await page.waitForTimeout(500)
  const usedDuring = await used()
  await shot(page, '7-cancelling')
  await page.click('.tab:text-is("my uploads")')
  const activeRow = await page.waitForSelector('.mine-list li:first-child[data-kind=active]', { timeout: 5000 })
    .then((li) => li.evaluate((el) => ({ title: el.querySelector('.row-title').textContent, meta: el.querySelector('.row-meta').textContent, bar: el.querySelector('.bar')?.getAttribute('aria-valuenow') })), () => null)
  check(activeRow?.title === `cancel me ${run}` && activeRow.meta === 'uploading' && activeRow.bar !== undefined && activeRow.bar !== null,
    `a running upload sits at the top of my uploads with its step and a progress bar (${JSON.stringify(activeRow)})`)
  await page.click('.tab:text-is("upload")')
  await page.click('.journey-cancel')
  await page.waitForSelector('.panel.upload .upload-error:text-is("upload cancelled")', { timeout: 10000 })
  await page.unroute('**/api/uploads/*/parts/*')
  const usedAfter = await used()
  const cancelled = (await apiAs(page, bearer, '/api/jobs')).body.jobs.find((j) => j.title === `cancel me ${run}`)
  await page.click('.upload-submit:text-is("try again")')
  await page.waitForFunction(([sel, q]) => document.querySelector(sel)?.getAttribute('aria-label') === q, [METER, quotaBefore], { timeout: 5000 }).catch(() => {})
  const quotaAfter = (await quotaOf(page)).label
  check(usedDuring === usedAfter + 1 && cancelled?.state === 'failed' && cancelled?.error === 'cancelled' && quotaAfter === quotaBefore,
    `cancel mid-upload frees the slot: ${usedDuring} used while sending, ${usedAfter} after; the panel is back to "${quotaAfter}"`)

  await page.fill('input[name=url]', 'http://127.0.0.1/never.mp4')
  await page.fill('input[name=title]', `fails ${run}`)
  await page.click('.upload-submit')
  await page.waitForSelector('.panel.upload .journey .upload-error', { timeout: 60000 })
  await page.click('.tab:text-is("my uploads")')
  const failedRow = '.mine-list li[data-kind=failed]'
  await page.waitForSelector(failedRow, { timeout: 10000 })
  const failed = await page.$eval(failedRow, (li) => ({ title: li.querySelector('.row-title').textContent, reason: li.querySelector('.row-error').textContent, color: getComputedStyle(li.querySelector('.row-error')).color }))
  const cancelledShown = await page.$$eval('.mine-list .row-title', (ts) => ts.some((t) => t.textContent.startsWith('cancel me')))
  await page.waitForTimeout(500)
  await shot(page, '9-failed-row')
  await page.click(`${failedRow} .row-dismiss`)
  await page.waitForSelector(failedRow, { state: 'detached', timeout: 5000 })
  await page.click('.panel-close')
  await corner(page, 'my uploads')
  await page.waitForSelector('.mine-empty', { timeout: 5000 }).catch(() => {})
  check(failed.title === `fails ${run}` && /private address/.test(failed.reason) && failed.color === 'rgb(217, 138, 138)' && !cancelledShown && !!(await page.$('.mine-empty')),
    `a failed upload shows its reason in muted red ("${failed.reason}") until dismissed, and stays dismissed; the cancelled one is not listed`)
  await page.click('.mine-empty button')
  await clickIfShown(page, 'button:text-is("upload another")')

  // Fill this network's daily uploads from other browser ids, then the panel shows the per-IP refusal.
  const fillers = []
  let full = null
  for (let i = 0; i < 40 && !full; i++) {
    const id = `device:${Array.from({ length: 32 }, () => '0123456789abcdef'[Math.floor(Math.random() * 16)]).join('')}`
    const r = await apiAs(page, id, '/api/uploads', { method: 'POST', json: { title: 'filler', filename: 'f.mp4', size: 10 } })
    if (r.status === 200) fillers.push([id, r.body.upload_id])
    else full = r
  }
  console.log(`filled the per-IP quota with ${fillers.length} uploads: ${full && `${full.status} ${full.body?.detail}`}`)
  await page.setInputFiles('.drop input[type=file]', fixtures.clip)
  await page.fill('input[name=title]', `over the cap ${run}`)
  await page.click('.upload-submit')
  await page.waitForSelector('.panel.upload .upload-error', { timeout: 10000 })
  const capText = await page.textContent('.panel.upload .upload-error')
  await shot(page, '8-ip-cap')
  check(full?.status === 429 && /^daily limit of \d+ uploads from this network reached; try again after 00:00 UTC$/.test(capText),
    `with this network's uploads used up the panel says: "${capText}"`)
  for (const [id, upload] of fillers) await apiAs(page, id, `/api/uploads/${upload}`, { method: 'DELETE' })

  for (const ctx of contexts) await ctx.close()
}

/** A production build (`vite build`, served by `vite preview`) talking to the API at VITE_API_URL: no dev hooks here. */
async function prodChecks() {
  const ctx = await browser.newContext({ viewport: VIEWPORT })
  const page = await ctx.newPage()
  page.on('pageerror', (e) => failures.push(`pageerror (prod): ${e.message}`))
  const api = new Set()
  page.on('request', (r) => { if (r.url().includes('/api/')) api.add(new URL(r.url()).origin) })
  await page.goto(BASE)
  await page.waitForSelector('.account-bar', { timeout: 60000 })
  const links = await page.$$eval('.account-bar button', (bs) => bs.map((b) => b.textContent))
  check(JSON.stringify(links) === JSON.stringify(['upload', 'my uploads']) && !(await page.$('.account-signin')),
    `prod build: the corner offers ${JSON.stringify(links)} and no sign in`)
  const jobs = page.waitForRequest((r) => new URL(r.url()).pathname === '/api/jobs')
  await page.click('.account-bar button:text-is("upload")')
  const sent = (await (await jobs).allHeaders()).authorization
  await page.waitForSelector(`${METER}[aria-label$="left today"]`, { timeout: 15000 }).catch(() => {})
  const quota = await page.getAttribute(METER, 'aria-label').catch(() => null)
  const bearer = await deviceToken(page)
  check(DEVICE_TOKEN.test(bearer ?? '') && sent === `Bearer ${bearer}` && quota === '3 of 3 uploads left today',
    `prod build: upload mints a browser id the API accepts (${sent}): "${quota}"`)
  await page.click('.account-bar button:text-is("my uploads")')
  const note = await page.getAttribute('.panel-foot .browser-note', 'title').catch(() => null)
  check(note === DEVICE_NOTE, `prod build: the panel footer notes "${note}"`)
  await shot(page, 'prod-device')
  console.log(`API origins called: ${[...api].join(', ')}`)
  await ctx.close()
}

/** Any collection's API: what the page calls the archive and its items, and hover seek on films not sampled at 1 fps. */
async function collectionChecks() {
  const ctx = await browser.newContext({ viewport: VIEWPORT })
  const page = await ctx.newPage()
  page.on('pageerror', (e) => failures.push(`pageerror (collection): ${e.message}`))
  await page.addInitScript(instrument)
  const preload = (await (await fetch(BASE)).text()).match(/<title>(.*?)<\/title>/)?.[1]
  await page.route('**/api/library', async (r) => { await new Promise((ok) => setTimeout(ok, 1500)); await r.continue().catch(() => {}) })
  await page.goto(BASE)
  await page.waitForSelector('.masthead-title')
  const before = { title: await page.textContent('.masthead-title'), doc: await page.title() }
  check(before.title === preload && before.doc === preload, `before /api/library answers the masthead and tab show the page's own title "${preload}": "${before.title}", "${before.doc}"`)
  await page.waitForFunction(() => window.__vidsearch?.settled())
  await ready(page)
  await page.unroute('**/api/library')
  const library = await page.evaluate(async () => (await (await fetch('/api/library')).json()).films)
  console.log('films:', library.map((f) => `${f.id} (${f.n_frames} at ${f.sample_fps} fps)`).join(', '))
  const masthead = { title: await page.textContent('.masthead-title'), stats: await page.textContent('.masthead-stats'), doc: await page.title() }
  check(masthead.title === IDENTITY.title && masthead.doc === IDENTITY.title, `then they show the API's title: "${masthead.title}", tab "${masthead.doc}"`)
  check(masthead.stats === libraryStats(library), `stats count ${IDENTITY.noun_plural}: "${masthead.stats}" (expected "${libraryStats(library)}")`)
  await shot(page, 'collection-overview')

  await page.click('.account-bar button:text-is("my uploads")')
  await page.waitForSelector('.panel.mine .mine-empty', { timeout: 10000 }).catch(() => {})
  const empty = { line: await page.textContent('.mine-empty p').catch(() => null), button: await page.textContent('.mine-empty button').catch(() => null) }
  await shots(page, 'collection-empty')
  await page.click('.mine-empty button')
  const placeholder = await page.getAttribute('.panel.upload input[name=url]', 'placeholder')
  check(empty.line === 'nothing uploaded yet' && empty.button === `upload a ${IDENTITY.noun}` && placeholder === `https://…/${IDENTITY.noun}.mp4`,
    `upload copy uses "${IDENTITY.noun}": "${empty.line}", "${empty.button}", link placeholder "${placeholder}"`)
  await page.click('.panel-close')

  const settle = async () => {
    await page.waitForTimeout(100)
    await page.waitForFunction(() => window.__vidsearch.settled(), null, { timeout: 15000 })
    await page.waitForTimeout(400)
  }
  await page.evaluate(() => document.addEventListener('seeked', (e) => { window.__seekedTo = e.target.currentTime }, true))
  const seekOf = async () => {
    await page.waitForFunction(() => { const v = document.querySelector('.card video'); return v && v.readyState >= 2 && !v.seeking && window.__seekedTo !== undefined }, null, { timeout: 15000 })
    return page.evaluate(() => ({ seekedTo: window.__seekedTo, label: document.querySelector('.card-label').textContent, still: document.querySelector('.card-still img')?.getAttribute('src') ?? null }))
  }
  const odd = library.filter((f) => f.sample_fps !== 1)
  check(odd.length > 0, `the library has films sampled at other than 1 fps: ${odd.map((f) => `${f.id} @ ${f.sample_fps}`).join(', ')}`)
  for (const film of odd) {
    const frame = Math.floor(film.n_frames * 0.6)
    const at = frame / film.sample_fps
    await page.mouse.move(5, 5)
    await page.evaluate(() => { window.__seekedTo = undefined })
    await page.evaluate(([id, f]) => window.__vidsearch.focusFrame(id, f, 150), [film.id, frame])
    await settle()
    const xy = await page.evaluate(([id, f]) => window.__vidsearch.screenOf(id, f), [film.id, frame])
    await page.route('**/api/video/**', async (r) => { await new Promise((ok) => setTimeout(ok, 800)); await r.continue().catch(() => {}) })
    await page.mouse.move(xy.x - 2, xy.y)
    await page.mouse.move(xy.x, xy.y)
    await page.waitForSelector('.card video')
    const still = await page.getAttribute('.card-still img', 'src').catch(() => null)
    await page.unroute('**/api/video/**')
    const seek = await seekOf()
    await shot(page, `collection-hover-${film.id}`)
    check(Math.abs(seek.seekedTo - at) < 0.3 && seek.label === `${film.title} · ${clock(at)}` && still === frameUrl(film.id, frame),
      `hovering ${film.id} frame ${frame} at ${film.sample_fps} fps seeks to ${seek.seekedTo?.toFixed(2)} s (expected ${at}), labels "${seek.label}", still ${still}`)
  }

  await page.evaluate(() => window.__vidsearch.showOverview())
  await settle()
  await page.mouse.move(5, 5)
  const { body } = await uiSearch(page, 'a car on the street')
  const top = body.results[0]
  const fps = library.find((f) => f.id === top.film_id).sample_fps
  check(Math.abs(top.t - top.frame / fps) < 1e-3 && top.start <= top.t && top.t < top.end && top.end - top.start <= 3.0 + 1e-6,
    `search times are seconds at ${fps} fps: t ${top.t} = frame ${top.frame} / ${fps}, moment ${top.start}-${top.end}`)
  await page.waitForTimeout(1500)
  await page.evaluate(() => { window.__seekedTo = undefined })
  // ArrowDown rather than pointing at the lifted card: in a library of one short clip the cards rise past the top of the view.
  await page.keyboard.press('ArrowDown')
  await page.waitForSelector('.card video')
  const seek = await seekOf()
  await shot(page, 'collection-result')
  check(Math.abs(seek.seekedTo - top.start) < 0.3 && seek.label === `${top.title} · ${clock(top.start)}`,
    `stepping to the top result seeks to ${seek.seekedTo?.toFixed(2)} s (result start ${top.start}), labels "${seek.label}"; caption: ${JSON.stringify(top.caption)}`)

  if (process.env.SERVED_PAGE) {
    const served = await ctx.newPage()
    served.on('pageerror', (e) => failures.push(`pageerror (served page): ${e.message}`))
    await served.route('**/api/library', async (r) => { await new Promise((ok) => setTimeout(ok, 1500)); await r.continue().catch(() => {}) })
    await served.goto(process.env.SERVED_PAGE)
    await served.waitForSelector('.masthead-title')
    const early = { title: await served.textContent('.masthead-title'), doc: await served.title() }
    await served.waitForSelector('.masthead-stats', { timeout: 20000 })
    const later = { title: await served.textContent('.masthead-title'), doc: await served.title() }
    check([early.title, early.doc, later.title, later.doc].every((t) => t === IDENTITY.title),
      `the page the API serves names "${IDENTITY.title}" before and after /api/library: ${JSON.stringify({ early, later })}`)
  }
  await ctx.close()
}
