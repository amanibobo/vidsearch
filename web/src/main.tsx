import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './App'
import { AccountProvider } from './auth'
import { ARCHIVE_TITLE } from './Masthead'
import './styles.css'

document.title = ARCHIVE_TITLE

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AccountProvider>
      <App />
    </AccountProvider>
  </StrictMode>,
)
