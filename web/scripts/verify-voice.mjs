// Voice search end to end in headless Chromium with a fake microphone that plays <wav>: live partial text in the pill,
// the final transcript, the automatic search and its results; then Esc, the mic hidden when the API has no voice,
// a 429 from the token route, a denied microphone and a missing one (the last three never reach ElevenLabs).
// Usage: node scripts/verify-voice.mjs <wav> <expected text> <outDir> [prefix]   (env BASE_URL, default http://localhost:5199;
// CHROMIUM to override the binary). The API needs ELEVENLABS_API_KEY; the real path opens two ElevenLabs sessions.
import { chromium } from 'playwright-core'
import { existsSync, readdirSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

const [wav, expected, outDir = '.', prefix = 'v10'] = process.argv.slice(2)
if (!wav || !expected) throw new Error('usage: verify-voice.mjs <wav> <expected text> <outDir> [prefix]')
const BASE = process.env.BASE_URL ?? 'http://localhost:5199'
const VIEWPORT = { width: 1920, height: 1080 }
const MIC = 'button.pill-mic'
const INPUT = 'input[aria-label="Search"]'

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
const shot = (page, name) => page.screenshot({ path: join(outDir, `${prefix}-${name}.png`) })
const words = (s) => s.toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').split(/\s+/).filter(Boolean)
/** Word-level similarity: 1 - edit distance / longer length. */
function similarity(a, b) {
  const x = words(a), y = words(b)
  const d = Array.from({ length: x.length + 1 }, (_, i) => [i, ...Array(y.length).fill(0)])
  for (let j = 1; j <= y.length; j++) d[0][j] = j
  for (let i = 1; i <= x.length; i++) for (let j = 1; j <= y.length; j++) {
    d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (x[i - 1] === y[j - 1] ? 0 : 1))
  }
  return 1 - d[x.length][y.length] / Math.max(x.length, y.length, 1)
}

const fakeMic = ['--use-fake-device-for-media-stream', `--use-file-for-fake-audio-capture=${wav}%noloop`]
const executablePath = findChromium()
const sessions = []

async function open(browser) {
  const page = await browser.newPage({ viewport: VIEWPORT })
  page.on('websocket', (ws) => {
    if (!ws.url().includes('elevenlabs.io')) return
    const s = { opened: Date.now(), closed: null, sent: 0, received: [] }
    sessions.push(s)
    ws.on('framesent', () => { s.sent++ })
    ws.on('framereceived', (f) => { try { s.received.push(JSON.parse(f.payload).message_type) } catch {} })
    ws.on('close', () => { s.closed = Date.now() })
  })
  page.on('pageerror', (e) => console.log(`page error: ${e.message}`))
  await page.goto(BASE)
  await page.waitForSelector('.loader', { state: 'detached', timeout: 60000 })
  return page
}
const micClass = (page) => page.$eval(MIC, (b) => b.className)

const granted = await chromium.launch({ executablePath, headless: true, args: ['--use-fake-ui-for-media-stream', ...fakeMic] })
try {
  // The real path: speak, watch partial text, let VAD end it, see results.
  const page = await open(granted)
  check(await page.isVisible(MIC), 'mic button shown when the API offers voice')
  check(await page.getAttribute(MIC, 'aria-label') === 'Search by voice', 'mic labelled for screen readers')
  await shot(page, '1-mic-idle')
  const searched = page.waitForRequest((r) => r.url().includes('/api/search?'), { timeout: 30000 })
  const levels = []
  const sampler = setInterval(() => page.$eval(MIC, (b) => Number(getComputedStyle(b).getPropertyValue('--level')) || 0).then((l) => levels.push(l)).catch(() => {}), 100)
  await page.click(MIC)
  await page.waitForFunction(() => document.querySelector('.pill-transcript .partial')?.textContent?.trim(), null, { timeout: 15000 })
  const partial = await page.$eval('.pill-transcript', (el) => el.textContent)
  check(/listening/.test(await micClass(page)), `listening state with partial text "${partial}"`)
  await shot(page, '2-listening')
  const query = new URL((await searched).url()).searchParams.get('q')
  clearInterval(sampler)
  const sim = similarity(query, expected)
  check(sim >= 0.85, `final transcript "${query}" matches "${expected}" (word similarity ${sim.toFixed(2)})`)
  check(Math.max(...levels) > 0.05, `level ring follows the microphone (peak --level ${Math.max(...levels).toFixed(2)})`)
  await page.waitForSelector('.pill-count', { timeout: 30000 })
  const count = Number(await page.textContent('.pill-count'))
  check(count > 0, `the search ran automatically and returned ${count} results`)
  check(await page.inputValue(INPUT) === query, 'the transcript stays in the field, editable')
  check(!/listening|connecting|finishing/.test(await micClass(page)), 'mic back to idle after the search')
  await page.waitForTimeout(1500)
  await shot(page, '3-results')
  await page.close()

  // Esc while listening stops without searching.
  const esc = await open(granted)
  await esc.fill(INPUT, 'typed first')
  let escSearched = false
  esc.on('request', (r) => { if (r.url().includes('/api/search?')) escSearched = true })
  await esc.click(MIC)
  await esc.waitForFunction((sel) => document.querySelector(sel)?.classList.contains('listening'), MIC, { timeout: 15000 })
  await esc.keyboard.press('Escape')
  await esc.waitForFunction((sel) => !/listening|connecting|finishing/.test(document.querySelector(sel)?.className ?? ''), MIC, { timeout: 5000 })
  const left = await esc.inputValue(INPUT)
  check(!escSearched, `Esc stops listening without searching (field: "${left}")`)
  check(!(await esc.isVisible('.pill-transcript')), 'Esc removes the live transcript')
  await esc.close()

  // A 429 from the token route shows the limit in the pill, with the API's reason in its tooltip.
  const limited = await open(granted)
  const detail = 'voice search limit of 20 an hour from this network reached; try again later or type your search'
  await limited.route('**/api/voice/token', (r) => r.fulfill({ status: 429, contentType: 'application/json', body: JSON.stringify({ detail }) }))
  await limited.click(MIC)
  await limited.waitForSelector('.pill-error', { timeout: 10000 })
  check(await limited.textContent('.pill-error') === 'voice limit reached', '429 shows "voice limit reached" in the pill')
  check(await limited.getAttribute('.pill-error', 'title') === detail, '429 tooltip carries the API detail')
  check(/error/.test(await micClass(limited)), 'mic shows the error state')
  await shot(limited, '4-limit')
  await limited.fill(INPUT, 'a truck')
  check(!(await limited.isVisible('.pill-error')), 'typing clears the voice error')
  await limited.close()

  // No microphone at all.
  const nomic = await granted.newPage({ viewport: VIEWPORT })
  await nomic.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException('Requested device not found', 'NotFoundError'))
  })
  await nomic.goto(BASE)
  await nomic.waitForSelector('.loader', { state: 'detached', timeout: 60000 })
  await nomic.click(MIC)
  await nomic.waitForSelector('.pill-error', { timeout: 10000 })
  check(await nomic.textContent('.pill-error') === 'no microphone found', 'no microphone shows "no microphone found"')
  await nomic.close()

  // The mic is hidden when the API says voice is unavailable.
  const off = await granted.newPage({ viewport: VIEWPORT })
  await off.route('**/api/library', async (r) => {
    const res = await r.fetch()
    await r.fulfill({ response: res, json: { ...(await res.json()), voice: { available: false } } })
  })
  await off.goto(BASE)
  await off.waitForSelector('.loader', { state: 'detached', timeout: 60000 })
  check(!(await off.isVisible(MIC)), 'mic hidden when the API reports voice unavailable')
  await off.close()
} finally {
  await granted.close()
}

// Denied: no auto-accepting fake UI, and the microphone permission set to denied for this origin.
const denied = await chromium.launch({ executablePath, headless: true, args: fakeMic })
try {
  const cdp = await denied.newBrowserCDPSession()
  await cdp.send('Browser.setPermission', { permission: { name: 'microphone' }, setting: 'denied', origin: new URL(BASE).origin })
  const page = await open(denied)
  let tokenAsked = false
  page.on('request', (r) => { if (r.url().includes('/api/voice/token')) tokenAsked = true })
  await page.click(MIC)
  await page.waitForSelector('.pill-error', { timeout: 10000 })
  check(await page.textContent('.pill-error') === 'microphone blocked', 'denied permission shows "microphone blocked"')
  check(!tokenAsked, 'denied permission never mints a token')
  await shot(page, '5-denied')
} finally {
  await denied.close()
}

const secs = sessions.reduce((n, s) => n + ((s.closed ?? Date.now()) - s.opened) / 1000, 0)
console.log(`ElevenLabs sessions: ${sessions.length}, ${secs.toFixed(1)} s open; messages: ${sessions.map((s) => s.received.join(',')).join(' | ')}`)
console.log(failures.length ? `\n${failures.length} FAILED` : '\nall passed')
process.exit(failures.length ? 1 : 0)
