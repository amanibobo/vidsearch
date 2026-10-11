import { useEffect, useRef, useState, type DragEvent, type FormEvent, type ReactNode } from 'react'
import { deleteFilm, STAGES, type Film, type FilmId, type Job, type Jobs, type UploadsMode } from './api'
import { useAccount, type Session, type Viewer } from './auth'
import { checkFile, formatBytes, titleFrom, uploadActive, type Upload, type Uploads } from './uploads'

type Panel = 'none' | 'menu' | 'upload' | 'mine'

/** What the top-right corner offers, from the API's upload mode and this build's sign-in. */
type Corner =
  | { kind: 'hidden' }
  | { kind: 'sign-in'; notice: string | null }
  | { kind: 'signed-in'; viewer: Viewer }
  /** Uploads tied to this browser: `upload` and `my uploads` right in the corner, no account. */
  | { kind: 'device' }

function cornerFor(mode: UploadsMode, session: Session): Corner {
  if (mode === 'device') return { kind: 'device' }
  if (mode === 'off') return { kind: 'hidden' }
  if (session.status === 'signed-in') return { kind: 'signed-in', viewer: session.viewer }
  if (session.status === 'signed-out') return { kind: 'sign-in', notice: session.notice }
  return { kind: 'hidden' }  // still checking, or this build has no sign-in configured
}

/**
 * Top-right corner. Account mode: `sign in` when signed out; signed in, an initial with a menu. Device mode: `upload`
 * and `my uploads` links. Either way the upload panels and a progress chip while an upload runs.
 */
export function AccountCorner({ mode, uploads, films, onLibraryChanged, onDeviceNeeded }: {
  mode: UploadsMode
  uploads: Uploads
  films: readonly Film[]
  onLibraryChanged: () => void
  /** Device mode: make sure this browser has its id before a panel asks the API about its uploads. */
  onDeviceNeeded: () => void
}) {
  const { session, signIn, signOut } = useAccount()
  const [panel, setPanel] = useState<Panel>('none')
  const root = useRef<HTMLDivElement>(null)
  const corner = cornerFor(mode, session)

  useEffect(() => {
    if (panel !== 'menu') return
    const close = (e: PointerEvent) => { if (!root.current?.querySelector('.account-bar')?.contains(e.target as Node)) setPanel('none') }
    window.addEventListener('pointerdown', close)
    return () => window.removeEventListener('pointerdown', close)
  }, [panel])

  if (corner.kind === 'hidden') return null
  if (corner.kind === 'sign-in') {
    return (
      <div className="account">
        <div className="account-bar">
          <button type="button" className="account-link account-signin" onClick={signIn}>sign in</button>
        </div>
        {corner.notice && <p className="account-notice">{corner.notice}</p>}
      </div>
    )
  }

  const open = (p: Panel) => {
    if (corner.kind === 'device' && p !== 'none') onDeviceNeeded()
    setPanel(p)
  }
  const toggle = (p: Panel) => open(panel === p ? 'none' : p)
  const active = uploads.list.find(uploadActive)
  return (
    <div className="account" ref={root}>
      <div className={`account-bar${corner.kind === 'device' ? ' device' : ''}`}>
        {active && <StatusChip upload={active} onClick={() => open('upload')} />}
        {corner.kind === 'device' ? (
          <>
            <button type="button" className="account-link" aria-expanded={panel === 'upload'} onClick={() => toggle('upload')}>upload</button>
            <button type="button" className="account-link" aria-expanded={panel === 'mine'} onClick={() => toggle('mine')}>my uploads</button>
          </>
        ) : (
          <>
            <button type="button" className="account-avatar" aria-label={`account: ${corner.viewer.name}`} aria-expanded={panel === 'menu'} onClick={() => toggle('menu')}>
              <Avatar viewer={corner.viewer} />
            </button>
            {panel === 'menu' && (
              <div className="menu" role="menu">
                <p className="menu-who">{corner.viewer.name}</p>
                <button type="button" role="menuitem" onClick={() => open('upload')}>upload</button>
                <button type="button" role="menuitem" onClick={() => open('mine')}>my uploads</button>
                <button type="button" role="menuitem" onClick={signOut}>sign out</button>
              </div>
            )}
          </>
        )}
      </div>
      {panel === 'upload' && <UploadPanel uploads={uploads} onClose={() => open('none')} />}
      {panel === 'mine' && <MyUploads uploads={uploads} films={films} browserOnly={corner.kind === 'device'} onClose={() => open('none')} onLibraryChanged={onLibraryChanged} />}
    </div>
  )
}

function Avatar({ viewer }: { viewer: Viewer }) {
  const [broken, setBroken] = useState(false)
  if (viewer.picture && !broken) return <img src={viewer.picture} alt="" referrerPolicy="no-referrer" onError={() => setBroken(true)} />
  return <span>{(viewer.name.trim()[0] ?? '?').toUpperCase()}</span>
}

const percent = (done: number, total: number) => `${Math.floor((100 * done) / Math.max(1, total))}%`

function chipText(u: Upload): string {
  if (u.phase === 'sending') return u.size ? `uploading ${percent(u.sent, u.size)}` : 'sending the link'
  if (u.phase === 'server' && u.job.state === 'processing' && u.job.stage) {
    return u.job.total > 1 ? `${u.job.stage} ${percent(u.job.done, u.job.total)}` : u.job.stage
  }
  return 'queued'
}

function StatusChip({ upload, onClick }: { upload: Upload; onClick: () => void }) {
  return (
    <button type="button" className="chip" onClick={onClick} title={upload.title}>
      <span className="spinner" aria-hidden="true" />
      <span className="chip-text">{chipText(upload)}</span>
    </button>
  )
}

function PanelFrame({ title, className, onClose, children }: { title: string; className: string; onClose: () => void; children: ReactNode }) {
  return (
    <section className={`panel ${className}`} aria-label={title}>
      <header className="panel-head">
        <h2>{title}</h2>
        <button type="button" className="panel-close" aria-label="Close" onClick={onClose}>
          <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" /></svg>
        </button>
      </header>
      {children}
    </section>
  )
}

function quotaLine(jobs: Jobs | null): { text: string; left: number } {
  if (!jobs) return { text: 'checking today’s uploads…', left: 1 }
  const left = Math.max(0, jobs.limits.uploadsPerDay - jobs.uploadsToday)
  return { text: left ? `${left} of ${jobs.limits.uploadsPerDay} uploads left today` : 'no uploads left today; more after 00:00 UTC', left }
}

function UploadPanel({ uploads, onClose }: { uploads: Uploads; onClose: () => void }) {
  // The panel follows the newest upload until the user asks for a new one.
  const [viewing, setViewing] = useState<string | null>(() => uploads.list[0]?.key ?? null)
  const shown = uploads.list.find((u) => u.key === viewing)
  return (
    <PanelFrame title={shown ? shown.title : 'upload a film'} className="upload" onClose={onClose}>
      {shown ? <UploadProgress upload={shown} onCancel={() => uploads.cancel(shown.key)} onAnother={() => setViewing(null)} />
        : <UploadForm uploads={uploads} onStarted={setViewing} />}
    </PanelFrame>
  )
}

type Source = { kind: 'none' } | { kind: 'file'; file: File } | { kind: 'url'; url: string }

function UploadForm({ uploads, onStarted }: { uploads: Uploads; onStarted: (key: string) => void }) {
  const [source, setSource] = useState<Source>({ kind: 'none' })
  const [title, setTitle] = useState('')
  const [titleEdited, setTitleEdited] = useState(false)
  const [refusal, setRefusal] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const picker = useRef<HTMLInputElement>(null)
  const quota = quotaLine(uploads.jobs)
  const limits = uploads.jobs?.limits

  const choose = (next: Source, name: string) => {
    setSource(next)
    setRefusal(next.kind === 'file' && limits ? checkFile(next.file, limits) : null)
    if (!titleEdited) setTitle(titleFrom(name))
  }
  const pickFile = (file: File | undefined) => { if (file) choose({ kind: 'file', file }, file.name) }
  const onDrop = (e: DragEvent) => { e.preventDefault(); setDragging(false); pickFile(e.dataTransfer.files[0]) }

  const submit = (e: FormEvent) => {
    e.preventDefault()
    const name = title.trim()
    if (source.kind === 'none' || !name || refusal || !quota.left) return
    onStarted(source.kind === 'file' ? uploads.startFile(source.file, name) : uploads.startUrl(source.url.trim(), name))
  }

  const file = source.kind === 'file' ? source.file : null
  const ready = source.kind !== 'none' && !!title.trim() && !refusal && quota.left > 0 && !!uploads.jobs
  return (
    <form className="upload-form" onSubmit={submit}>
      <div
        className={['drop', dragging && 'over', file && 'chosen'].filter(Boolean).join(' ')}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        {file ? (
          <p className="drop-file"><span className="drop-name">{file.name}</span> <span className="drop-size">{formatBytes(file.size)}</span></p>
        ) : (
          <p className="drop-line">drop a video here</p>
        )}
        <button type="button" className="text-button" onClick={() => picker.current?.click()}>{file ? 'choose another' : 'or choose a file'}</button>
        <p className="drop-hint">up to {limits ? formatBytes(limits.maxBytes) : '2 GB'} and {limits?.maxMinutes ?? 60} min</p>
        <input ref={picker} type="file" accept="video/*" hidden onChange={(e) => { pickFile(e.target.files?.[0]); e.target.value = '' }} />
      </div>
      <label className="field">
        <span>or paste a direct video link</span>
        <input
          name="url"
          type="text"
          inputMode="url"
          placeholder="https://…/film.mp4"
          spellCheck={false}
          value={source.kind === 'url' ? source.url : ''}
          onChange={(e) => (e.target.value ? choose({ kind: 'url', url: e.target.value }, e.target.value) : setSource({ kind: 'none' }))}
        />
      </label>
      <label className="field">
        <span>title</span>
        <input name="title" type="text" maxLength={200} value={title} onChange={(e) => { setTitle(e.target.value); setTitleEdited(true) }} />
      </label>
      {refusal && <p className="upload-error" role="alert">{refusal}</p>}
      <div className="upload-foot">
        <span className="quota">{quota.text}</span>
        <button type="submit" className="upload-submit" disabled={!ready}>upload</button>
      </div>
    </form>
  )
}

function stagesFor(job: Job): readonly string[] {
  return job.source === 'url' ? STAGES : STAGES.filter((s) => s !== 'downloading')
}

function UploadProgress({ upload, onCancel, onAnother }: { upload: Upload; onCancel: () => void; onAnother: () => void }) {
  if (upload.phase === 'sending') {
    const url = upload.size === 0
    return (
      <div className="transfer">
        <div className="transfer-bar" aria-hidden="true"><span style={{ transform: `scaleX(${url ? 0 : upload.sent / upload.size})` }} /></div>
        <div className="transfer-line">
          <span className="transfer-bytes">{url ? 'sending the link…' : `${formatBytes(upload.sent)} of ${formatBytes(upload.size)}${upload.bytesPerSec ? ` · ${formatBytes(upload.bytesPerSec)}/s` : ''}`}</span>
          <button type="button" className="text-button transfer-cancel" onClick={onCancel}>cancel</button>
        </div>
      </div>
    )
  }
  if (upload.phase === 'refused') {
    return (
      <div className="transfer">
        <p className="upload-error" role="alert">{upload.message}</p>
        <div className="upload-foot"><span /><button type="button" className="upload-submit" onClick={onAnother}>try again</button></div>
      </div>
    )
  }
  const { job } = upload
  const order = stagesFor(job)
  const at = job.stage ? order.indexOf(job.stage) : -1
  const stateOf = (i: number) =>
    job.state === 'done' ? 'done' : job.state === 'queued' ? 'todo' : i < at ? 'done' : i === at ? (job.state === 'failed' ? 'failed' : 'current') : 'todo'
  return (
    <div className="job">
      <ol className="stages">
        {order.map((stage, i) => {
          const st = stateOf(i)
          return (
            <li key={stage} data-state={st}>
              <span className="stage-mark" aria-hidden="true">{st === 'current' ? <span className="spinner" /> : null}</span>
              <span className="stage-name">{stage}</span>
              {st === 'current' && job.total > 1 && <span className="stage-count">{job.done.toLocaleString('en-US')} / {job.total.toLocaleString('en-US')}</span>}
            </li>
          )
        })}
      </ol>
      {job.state === 'queued' && <p className="job-note">queued, waiting for a worker</p>}
      {job.state === 'done' && <p className="job-note job-done">ready: it is in the archive{job.error ? ` (${job.error})` : ''}</p>}
      {job.state === 'failed' && <p className="upload-error" role="alert">{job.error ?? 'processing failed'}</p>}
      {!(job.state === 'queued' || job.state === 'processing') && (
        <div className="upload-foot"><span /><button type="button" className="upload-submit" onClick={onAnother}>upload another</button></div>
      )}
    </div>
  )
}

const when = (t: number) => new Date(t * 1000).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false })

function jobLabel(job: Job, inLibrary: boolean): string {
  if (job.state === 'done') return inLibrary ? 'ready' : 'deleted'
  if (job.state === 'failed') return job.error === 'cancelled' ? 'cancelled' : `failed: ${job.error ?? 'unknown error'}`
  if (job.state === 'uploading') return 'upload not finished'
  if (job.state === 'queued') return 'queued'
  return job.stage ?? 'processing'
}

function MyUploads({ uploads, films, browserOnly, onClose, onLibraryChanged }: { uploads: Uploads; films: readonly Film[]; browserOnly: boolean; onClose: () => void; onLibraryChanged: () => void }) {
  const [confirming, setConfirming] = useState<FilmId | null>(null)
  const [deleting, setDeleting] = useState<FilmId | null>(null)
  const [error, setError] = useState<string | null>(null)
  const { refreshJobs } = uploads
  useEffect(() => { refreshJobs() }, [refreshJobs])
  const mine = new Set(films.filter((f) => f.ownerIsYou).map((f) => f.id))

  const remove = (id: FilmId) => {
    setDeleting(id)
    setError(null)
    deleteFilm(id)
      .then(() => { onLibraryChanged(); refreshJobs() })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => { setDeleting(null); setConfirming(null) })
  }

  const jobs = uploads.jobs?.jobs
  return (
    <PanelFrame title="my uploads" className="mine" onClose={onClose}>
      {browserOnly && <p className="mine-note">uploads are tied to this browser; clearing site data loses access</p>}
      {!jobs ? <p className="job-note">loading…</p> : jobs.length === 0 ? <p className="job-note">nothing uploaded yet</p> : (
        <ul className="mine-list">
          {jobs.map((job) => {
            const live = mine.has(job.filmId)
            return (
              <li key={job.id} data-film={job.filmId} data-state={job.error === 'cancelled' ? 'cancelled' : job.state}>
                <div className="mine-main">
                  <span className="mine-title">{job.title}</span>
                  <span className="mine-meta">{jobLabel(job, live)} · {when(job.created)}</span>
                </div>
                {job.state === 'done' && live && (confirming === job.filmId ? (
                  <span className="mine-confirm">
                    <button type="button" className="text-button danger" disabled={deleting === job.filmId} onClick={() => remove(job.filmId)}>{deleting === job.filmId ? 'deleting…' : 'delete'}</button>
                    <button type="button" className="text-button" onClick={() => setConfirming(null)}>keep</button>
                  </span>
                ) : (
                  <button type="button" className="text-button mine-delete" onClick={() => setConfirming(job.filmId)}>delete</button>
                ))}
              </li>
            )
          })}
        </ul>
      )}
      {error && <p className="upload-error" role="alert">{error}</p>}
      {uploads.jobs && <p className="quota mine-quota">{quotaLine(uploads.jobs).text}</p>}
    </PanelFrame>
  )
}
