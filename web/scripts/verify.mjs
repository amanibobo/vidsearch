// Drives the running dev server in headless Chromium at 1920x1080: screenshots, the archive title and loaders (slow,
// failing and cold API; atlas progress), atlas pixel checks on plane slices and lifted cards, hover-card placement, frame
// still and video seek, search states and retries, camera travel, fps, and the overview at a 3-film library. Then
// accounts and uploads: sign-in UI, quota, client-side refusals, a chunked upload with an injected part failure, job
// stages, the new private strip and camera move, search and privacy across dev users, URL errors, delete, expired sessions.
// Usage: node scripts/verify.mjs <outDir> <label>   (env BASE_URL, default http://localhost:5199; CHROMIUM to override the binary;
// ONLY=archive or ONLY=account runs one half). The account half needs the API on VS_DEV_AUTH=1 and the dev server on
// VITE_DEV_AUTH=tester, and makes its 30 s upload clip from ../videos/Cops.mp4 with ffmpeg.
import { chromium } from 'playwright-core'
import { spawnSync } from 'node:child_process'
import { closeSync, existsSync, ftruncateSync, mkdirSync, openSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

const [outDir = '.', label = 'run'] = process.argv.slice(2)
mkdirSync(outDir, { recursive: true })
const BASE = process.env.BASE_URL ?? 'http://localhost:5199'
const VIEWPORT = { width: 1920, height: 1080 }
const ATLAS_FILM = 'one-week'
const ATLAS_FRAMES = [0, 700, 1335, 1336]
const CARD_ATLAS_RANKS = [0, 1, 2]
const INPUT = 'input[aria-label="Search"]'
const ARCHIVE_TITLE = process.env.VITE_ARCHIVE_TITLE || 'The Keaton Archive'
const WAITING_PLACEHOLDER = 'waiting for the archive…'
const ONLY = process.env.ONLY ?? ''
const DEV_USER = 'tester'
const SOURCE_FILM = new URL('../../videos/Cops.mp4', import.meta.url).pathname
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
  const frames = films.reduce((n, f) => n + f.n_frames, 0)
  return `${films.length} films · ${Math.floor(minutes / 60)} h ${minutes % 60} min · ${frames.toLocaleString('en-US')} frames`
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
  const png = await page.screenshot({ clip: rect })
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
  if (ONLY !== 'account') await archiveChecks()
  if (ONLY !== 'archive') await accountChecks()
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
  check(masthead.title === ARCHIVE_TITLE && masthead.doc === ARCHIVE_TITLE, `archive title "${masthead.title}", document title "${masthead.doc}"`)
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
  const perColumn = await page.evaluate((id) => { const a = window.__vidsearch.screenOf(id, 0), b = window.__vidsearch.screenOf(id, 1); return { a, b } }, ATLAS_FILM)
  check(perColumn.b.y < perColumn.a.y, `time runs away from the viewer: frame 1 is above frame 0 on screen (${perColumn.b.y.toFixed(1)} < ${perColumn.a.y.toFixed(1)})`)
  const idleTone = await tone(page, { x: 0, y: 150, width: VIEWPORT.width, height: pill.y - 160 })

  if (process.env.ATLAS !== '0') {
    const nFrames = library.find((f) => f.id === ATLAS_FILM).n_frames
    for (const frame of ATLAS_FRAMES) {
      await page.evaluate(([id, f]) => window.__vidsearch.focusFrame(id, f, 1400), [ATLAS_FILM, frame])
      await settle()
      const inset = 0.1, cols = 96, rows = 12
      const { band, points } = await page.evaluate(([id, f, c, r, i]) => window.__vidsearch.sliceGrid(id, f, c, r, i), [ATLAS_FILM, frame, cols, rows, inset])
      const result = await compareRegion(page, { film: ATLAS_FILM, frame, nFrames, points, cols, rows, band: [0.5 - band / 2 + inset * band, band * (1 - 2 * inset)] })
      atlasVerdict(`slice ${ATLAS_FILM}`, frame, result, `atlas-slice-${frame}`)
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
  await page.mouse.wheel(0, 300)
  await settle()
  const afterWheel = await view()
  check(afterWheel.y > afterDrag.y + 1, `wheel pans forward in time (focus y ${afterDrag.y.toFixed(2)} -> ${afterWheel.y.toFixed(2)})`)
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
  await page.fill(INPUT, 'a train')
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
  const coldStatus = await searchFor('a train')
  check(/^\d+$/.test(coldStatus) && coldCalls === 2, `a search whose first attempt gets 503 retries and succeeds: "${coldStatus}" after ${coldCalls} requests`)
  await page.unroute('**/api/search**')

  await page.route('**/api/search?q=a%20dog*', async (r) => { await new Promise((ok) => setTimeout(ok, 3000)); await r.continue().catch(() => {}) })
  await page.fill(INPUT, 'a dog')
  await page.press(INPUT, 'Enter')
  check(await page.isVisible('.pill .spinner'), 'loading shows a spinner inside the pill')
  const trainCount = await searchFor('a train')
  const trainTop = await page.evaluate(async () => (await (await fetch('/api/search?q=a%20train')).json()).results[0])
  await page.keyboard.press('ArrowDown')
  await page.waitForSelector('.card-label')
  const trainLabel = await page.textContent('.card-label')
  await page.waitForTimeout(3500)
  await page.unroute('**/api/search?q=a%20dog*')
  const later = { label: await page.textContent('.card-label').catch(() => null), count: await statusText() }
  check(later.label === trainLabel && trainLabel === `${trainTop.title} · ${clock(trainTop.start)}` && later.count === `1/${trainCount}`,
    `slow superseded "a dog" search ignored: card stays "${later.label}", count "${later.count}" (a train top: ${trainTop.title} · ${clock(trainTop.start)})`)
  check(/^\d+$/.test(trainCount), `search "a train" shows its result count in the pill: "${trainCount}"`)
  await page.mouse.move(5, 5)
  await page.mouse.move(8, 8)
  await page.waitForSelector('.hover', { state: 'detached' })
  await shot(page, '5-a-train')

  const chasedCount = await searchFor('being chased')
  const results = await page.evaluate(async () => (await (await fetch('/api/search?q=being%20chased')).json()).results)
  check(chasedCount === String(results.length), `search "being chased" count in pill: "${chasedCount}" of ${results.length}`)
  await page.waitForFunction(() => window.__vidsearch.card(0)?.opacity === 1, null, { timeout: 5000 })
  await page.waitForTimeout(1200)
  await shot(page, '4-being-chased')
  const cards = await page.evaluate((n) => Array.from({ length: n }, (_, r) => window.__vidsearch.card(r)), results.length)
  const lifted = cards.filter(Boolean)
  check(lifted.length >= Math.min(results.length, 10) && lifted.every((c) => c.box), `top results lifted as cards on screen: ${lifted.length} of ${results.length}`)
  const chasedTone = await tone(page, { x: 0, y: 150, width: VIEWPORT.width, height: pill.y - 160 })
  check(chasedTone.lum < idleTone.lum * 0.45 && chasedTone.lilac > idleTone.lilac + 2000,
    `results dim the archive (mean luma ${idleTone.lum} -> ${chasedTone.lum}) and show lilac heat (${idleTone.lilac} -> ${chasedTone.lilac} lilac px)`)
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
  page3.on('pageerror', (e) => failures.push(`pageerror (3 films): ${e.message}`))
  await page3.route('**/api/library', async (r) => {
    const body = await (await r.fetch()).json()
    await r.fulfill({ json: { ...body, films: body.films.slice(0, 3) } })
  })
  await page3.goto(BASE)
  await page3.waitForFunction(() => window.__vidsearch?.settled())
  await ready(page3)
  await page3.waitForTimeout(2500)
  await shot(page3, '6-overview-3-films')
  const pill3 = await page3.locator('form.pill').boundingBox()
  const films3 = await page3.evaluate(async () => (await (await fetch('/api/library')).json()).films)
  const ends3 = await page3.evaluate((films) => films.map((f) => [window.__vidsearch.screenOf(f.id, 0), window.__vidsearch.screenOf(f.id, f.n_frames - 1)]), films3)
  const inView3 = (p) => p && p.x > 0 && p.x < VIEWPORT.width && p.y > 0 && p.y < pill3.y
  check(films3.length === 3 && ends3.flat().every(inView3), `3-film library (first 3 films) fits the overview above the pill`)
  const near3 = await page3.evaluate((films) => films.map((f) => window.__vidsearch.screenOf(f.id, 20)), films3)
  const lum3 = []
  for (const p of near3) lum3.push((await tone(page3, { x: p.x - 12, y: p.y - 8, width: 24, height: 16 })).lum)
  check(lum3.every((l) => l > 15), `every film in the 3-film library draws (luma near frame 20: ${lum3.join(', ')}; background ~5)`)
}

/** A 30 s clip re-encoded at 6 Mbit/s, so it is three 8 MiB parts, and a sparse file just over the 2 GB limit. */
function uploadFixtures() {
  const clip = join(outDir, 'upload-clip.mp4')
  if (!existsSync(clip)) {
    const r = spawnSync('ffmpeg', ['-y', '-loglevel', 'error', '-ss', '600', '-t', '30', '-i', SOURCE_FILM, '-c:v', 'libx264', '-preset', 'veryfast',
      '-b:v', '6M', '-maxrate', '6M', '-bufsize', '12M', '-c:a', 'aac', clip], { stdio: 'inherit' })
    if (r.status !== 0) throw new Error(`ffmpeg failed making ${clip}`)
  }
  const oversize = join(outDir, 'upload-oversize.mp4')
  const fd = openSync(oversize, 'w')
  ftruncateSync(fd, 2 * 1024 ** 3 + 1024 ** 2)
  closeSync(fd)
  return { clip, oversize }
}

async function accountChecks() {
  const run = Date.now().toString(36)
  const me = `tester-${run}`, other = `other-${run}`
  const { clip, oversize } = uploadFixtures()
  const clipBytes = statSync(clip).size
  const contexts = []

  /** A fresh browser profile signed in as `user` through dev auth ('' = signed out), on the loaded archive. */
  const open = async (user, prepare) => {
    const ctx = await browser.newContext({ viewport: VIEWPORT })
    contexts.push(ctx)
    await ctx.addInitScript((u) => localStorage.setItem('vidsearch.dev-user', u), user)
    const page = await ctx.newPage()
    page.on('pageerror', (e) => failures.push(`pageerror (${user || 'signed out'}): ${e.message}`))
    page.on('console', (m) => { if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) console.log(`console.error (${user || 'signed out'}):`, m.text()) })
    if (prepare) await prepare(page)
    await page.goto(BASE)
    await page.waitForFunction(() => window.__vidsearch?.settled())
    await ready(page)
    return page
  }
  const apiAs = (page, user, path) => page.evaluate(async ([u, p]) => {
    const r = await fetch(p, { headers: u ? { Authorization: `Bearer dev:${u}` } : {} })
    return { status: r.status, body: await r.json().catch(() => null) }
  }, [user, path])
  const menu = async (page, item) => {
    if (!(await page.$('.menu'))) await page.click('.account-avatar')
    await page.click(`.menu button:text-is("${item}")`)
  }
  const uiSearch = async (page, q) => {
    const res = page.waitForResponse((r) => r.url().includes('/api/search?'))
    await page.fill(INPUT, q)
    await page.press(INPUT, 'Enter')
    const r = await res
    return { auth: (await r.request().allHeaders()).authorization ?? null, body: await r.json() }
  }

  const out = await open('')
  await out.waitForTimeout(800)
  await shot(out, '1-signed-out')
  check(await out.textContent('.account-signin') === 'sign in' && !(await out.$('.account-avatar')) && !(await out.$('.panel')) && !(await out.$('.chip')),
    'signed out: top-right shows "sign in", no avatar, no upload')
  await out.click('.account-signin')
  await out.click('.account-avatar')
  check(await out.textContent('.menu-who') === DEV_USER, `"sign in" with dev auth signs in as VITE_DEV_AUTH (${await out.textContent('.menu-who')})`)

  const page = await open(me)
  await page.click('.account-avatar')
  const items = await page.$$eval('.menu button', (bs) => bs.map((b) => b.textContent))
  check(JSON.stringify(items) === JSON.stringify(['upload', 'my uploads', 'sign out']), `signed in: the avatar menu lists ${JSON.stringify(items)}`)
  await menu(page, 'upload')
  await page.waitForSelector('.quota:text-matches("of 3")')
  const quota0 = await page.textContent('.quota')
  check(quota0 === '3 of 3 uploads left today', `upload panel shows "${quota0}"`)
  await shot(page, '2-upload-panel')

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
  const autoTitle = await page.inputValue('input[name=title]')
  check(autoTitle === 'upload-clip' && !(await page.$('.upload-error')) && !(await page.isDisabled('.upload-submit')), `choosing the clip clears the refusal and titles it from the file name ("${autoTitle}")`)
  const title = `Cops clip ${run}`
  await page.fill('input[name=title]', title)
  const viewBefore = await page.evaluate(() => window.__vidsearch.view())
  await page.click('.upload-submit')
  await page.waitForSelector('.transfer-bar')
  await page.waitForFunction(() => /^[1-9][\d.]* MB of/.test(document.querySelector('.transfer-bytes')?.textContent ?? ''), null, { timeout: 15000 }).catch(() => {})
  await page.waitForTimeout(300)
  const sending = { bytes: await page.textContent('.transfer-bytes').catch(() => null), chip: await page.textContent('.chip-text').catch(() => null) }
  await shot(page, '3-uploading')
  check((sending.bytes ?? '').includes(` of ${(clipBytes / 1024 ** 2).toFixed(1)} MB`) && /^uploading \d+%$/.test(sending.chip ?? ''), `while sending: "${sending.bytes}", chip "${sending.chip}"`)
  await page.waitForSelector('.stages', { timeout: 60000 })
  check(JSON.stringify(parts) === JSON.stringify({ 0: 1, 1: 2, 2: 1 }) && peak === 3,
    `three 8 MiB parts of ${clipBytes} bytes sent 3 at a time (peak ${peak}); part 1 failed once with 500 and was retried (${JSON.stringify(parts)})`)
  await page.unroute('**/api/uploads/*/parts/*')

  await page.click('.panel-close')
  await page.waitForTimeout(POLL)
  const chipClosed = await page.textContent('.chip-text').catch(() => null)
  check(!!chipClosed && !(await page.$('.panel')), `with the panel closed the job keeps running behind a chip: "${chipClosed}"`)
  await page.click('.chip')
  await page.waitForSelector('.panel.upload .stages')

  const seen = new Set()
  let shotTaken = false, shotAtProgress = false
  const deadline = Date.now() + 10 * 60_000
  while (Date.now() < deadline && !(await page.$('.job-done, .panel.upload .upload-error'))) {
    const cur = await page.$eval('.stages li[data-state=current]', (li) => ({ name: li.querySelector('.stage-name').textContent, count: li.querySelector('.stage-count')?.textContent ?? null })).catch(() => null)
    if (cur) {
      seen.add(cur.name)
      if (cur.count && !shotAtProgress && (!shotTaken || !cur.count.startsWith('0 '))) {
        await shot(page, '4-processing')
        shotTaken = true
        shotAtProgress = !cur.count.startsWith('0 ')
        console.log(`processing shot at ${cur.name} ${cur.count}`)
      }
    }
    await page.waitForTimeout(250)
  }
  if (!shotTaken) await shot(page, '4-processing')
  const doneText = await page.textContent('.job-done').catch(() => null)
  const finalStages = await page.$$eval('.stages li', (lis) => lis.map((li) => `${li.querySelector('.stage-name').textContent}:${li.dataset.state}`))
  console.log('stages seen while polling:', [...seen].join(' → '))
  check(!!doneText && finalStages.every((s) => s.endsWith(':done')) && seen.size >= 3 && seen.has('embedding'),
    `the job's stages advance to done: "${doneText}" (${finalStages.join(', ')})`)
  check(!(await page.$('.chip')), 'the status chip goes away once nothing is running')

  const jobs = (await apiAs(page, me, '/api/jobs')).body
  const job = jobs.jobs.find((j) => j.title === title)
  const filmId = job?.film_id
  await page.waitForFunction((id) => !!window.__vidsearch.ownerMark(id), filmId, { timeout: 15000 }).catch(() => {})
  await page.click('.panel-close')
  await page.waitForTimeout(100)
  await page.waitForFunction(() => window.__vidsearch.settled(), null, { timeout: 15000 })
  await page.waitForTimeout(600)
  const viewAfter = await page.evaluate(() => window.__vidsearch.view())
  const mark = await page.evaluate((id) => window.__vidsearch.ownerMark(id), filmId)
  const markTone = mark && await tone(page, { x: mark.left, y: mark.top, width: Math.max(2, mark.right - mark.left), height: Math.max(2, mark.bottom - mark.top) })
  const stripMid = await page.evaluate(([id, n]) => window.__vidsearch.screenOf(id, Math.floor(n / 2)), [filmId, 30])
  await shot(page, '5-new-strip')
  const lib = (await apiAs(page, me, '/api/library')).body.films
  const mine = lib.find((f) => f.id === filmId)
  check(!!mine?.owner_is_you && lib.filter((f) => f.owner_is_you).length === 1, `the new film ${filmId} is in ${me}'s library, marked owner_is_you`)
  check(!!mark && markTone.lilac > 0.5 * (mark.right - mark.left) * (mark.bottom - mark.top), `its strip carries the lilac ownership mark (${markTone?.lilac} lilac px in ${JSON.stringify(mark && { w: Math.round(mark.right - mark.left), h: Math.round(mark.bottom - mark.top) })})`)
  const others = await page.evaluate((ids) => ids.map((id) => window.__vidsearch.ownerMark(id)), lib.filter((f) => !f.owner_is_you).map((f) => f.id))
  check(others.every((m) => m === null), 'public films carry no ownership mark')
  check(viewAfter.dist < viewBefore.dist * 0.6 && !!stripMid && Math.abs(stripMid.x - VIEWPORT.width / 2) < 120 && Math.abs(stripMid.y - VIEWPORT.height / 2) < 160,
    `the camera eased to the new strip (dist ${viewBefore.dist.toFixed(1)} -> ${viewAfter.dist.toFixed(1)}; strip middle on screen at ${stripMid && `${stripMid.x.toFixed(0)},${stripMid.y.toFixed(0)}`})`)

  const probe = (await apiAs(page, me, `/api/search?q=${encodeURIComponent('a man')}&k=3&films=${filmId}`)).body.results
  const query = probe[0]?.caption ?? 'a man'
  console.log(`searching for a moment of the clip: "${query}"`)
  const found = await uiSearch(page, query)
  const hit = found.body.results.find((r) => r.film_id === filmId)
  check(found.auth === `Bearer dev:${me}` && !!hit && found.body.heatmap.films[filmId] !== undefined,
    `${me}'s search finds a moment in the upload (rank ${hit?.rank}, ${hit && clock(hit.t)}), sending its bearer token`)

  const page2 = await open(other)
  const lib2 = (await apiAs(page2, other, '/api/library')).body.films
  const anon = (await apiAs(page2, null, '/api/library')).body.films
  const found2 = await uiSearch(page2, query)
  const job2 = await apiAs(page2, other, `/api/jobs/${job.id}`)
  check(!lib2.some((f) => f.id === filmId) && !anon.some((f) => f.id === filmId) && !(await page2.evaluate((id) => window.__vidsearch.screenOf(id, 0), filmId)),
    `another user (${other}) and anonymous visitors don't get the film in the library or on the plane`)
  check(found2.auth === `Bearer dev:${other}` && found2.body.results.length > 0 && !found2.body.results.some((r) => r.film_id === filmId) && found2.body.heatmap.films[filmId] === undefined && job2.status === 403,
    `another user's search for the same words returns ${found2.body.results.length} public results and none from the upload; its job is 403 to them`)

  await page.fill(INPUT, '')
  await page.press(INPUT, 'Enter')
  await menu(page, 'upload')
  await page.click('.upload-submit:text-is("upload another")')
  await page.fill('input[name=url]', 'not a link')
  await page.fill('input[name=title]', 'bad link')
  await page.click('.upload-submit')
  await page.waitForSelector('.panel.upload .upload-error')
  const urlError = await page.textContent('.panel.upload .upload-error')
  check(urlError === 'url must be an http(s) link to a video file', `a URL upload of a non-http string shows the API's error: "${urlError}"`)
  await page.click('.upload-submit:text-is("try again")')
  const quota1 = await page.textContent('.quota')
  check(quota1 === '2 of 3 uploads left today', `the remaining count follows /api/jobs: "${quota1}"`)

  await menu(page, 'my uploads')
  const row = `.mine-list li[data-film="${filmId}"]`
  await page.waitForSelector(row)
  const rowText = { title: await page.textContent(`${row} .mine-title`), meta: await page.textContent(`${row} .mine-meta`) }
  check(rowText.title === title && /^ready · /.test(rowText.meta), `my uploads lists "${rowText.title}" as "${rowText.meta}"`)
  await shot(page, '6-my-uploads')
  await page.click(`${row} .mine-delete`)
  await page.click(`${row} .danger`)
  await page.waitForSelector(`${row} .mine-meta:text-matches("^deleted")`, { timeout: 15000 })
  await page.waitForFunction((id) => !window.__vidsearch.screenOf(id, 0), filmId, { timeout: 15000 }).catch(() => {})
  const libAfter = (await apiAs(page, me, '/api/library')).body.films
  check(!libAfter.some((f) => f.id === filmId) && !(await page.evaluate((id) => window.__vidsearch.screenOf(id, 0), filmId)) && !(await page.$(`${row} .mine-delete`)),
    'delete (after confirming) removes the film from the API library and its strip from the plane')

  let jobsCalls = 0
  const expired = await open(me, (p) => p.route('**/api/jobs', (r) => { if (r.request().headers().authorization) jobsCalls++; return r.fulfill({ status: 401, json: { detail: 'invalid token: expired' } }) }))
  await expired.waitForSelector('.account-notice', { timeout: 10000 }).catch(() => {})
  const notice = await expired.textContent('.account-notice').catch(() => null)
  check(notice === 'your session expired; sign in again' && jobsCalls === 2 && await expired.isVisible('.account-signin'),
    `a 401 that survives one fresh-token retry (${jobsCalls} requests) signs the user out with "${notice}"`)

  for (const ctx of contexts) await ctx.close()
}