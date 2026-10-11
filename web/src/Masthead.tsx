import type { Film } from './api'

export const ARCHIVE_TITLE = String(import.meta.env.VITE_ARCHIVE_TITLE || 'The Keaton Archive')

/** `8 films · 2 h 46 min · 9,951 frames`: film count, total running time, total sampled frames. */
export function archiveStats(films: readonly Film[]): string {
  const minutes = Math.round(films.reduce((s, f) => s + f.duration, 0) / 60)
  const h = Math.floor(minutes / 60), m = minutes % 60
  const runtime = h && m ? `${h} h ${m} min` : h ? `${h} h` : `${m} min`
  const frames = films.reduce((n, f) => n + f.nFrames, 0).toLocaleString('en-US')
  return `${films.length} ${films.length === 1 ? 'film' : 'films'} · ${runtime} · ${frames} frames`
}

export function Masthead({ films }: { films: readonly Film[] | null }) {
  return (
    <header className="masthead">
      <h1 className="masthead-title">{ARCHIVE_TITLE}</h1>
      {films && <p className="masthead-stats">{archiveStats(films)}</p>}
    </header>
  )
}
