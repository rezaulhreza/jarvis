import { useState, useRef, useEffect } from 'react'
import { LogOut } from 'lucide-react'
import { useAuth } from '../../contexts/AuthContext'

export function UserMenu() {
  const { user, authEnabled, logout } = useAuth()
  const [open, setOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

  // Close on click outside
  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  // Don't render if auth is not enabled
  if (!authEnabled || !user) return null

  const initials = user.name
    ? user.name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)
    : user.email[0].toUpperCase()

  return (
    <div ref={menuRef} className="relative border-t border-border/10 p-3">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2.5 px-2 py-1.5 rounded-lg hover:bg-surface-2 transition-colors"
      >
        {user.avatar_url ? (
          <img src={user.avatar_url} alt="" className="w-7 h-7 rounded-full" />
        ) : (
          <div className="w-7 h-7 rounded-full bg-primary/20 flex items-center justify-center text-xs font-medium text-primary">
            {initials}
          </div>
        )}
        <div className="flex-1 text-left min-w-0">
          <div className="text-sm font-medium truncate">{user.name || user.email}</div>
          {user.name && (
            <div className="text-[11px] text-text-muted truncate">{user.email}</div>
          )}
        </div>
      </button>

      {open && (
        <div className="absolute bottom-full left-3 right-3 mb-1 bg-surface border border-border rounded-lg shadow-lg overflow-hidden">
          <button
            onClick={async () => { setOpen(false); await logout(); window.location.href = '/login' }}
            className="w-full flex items-center gap-2 px-3 py-2 text-sm text-text-muted hover:bg-surface-2 hover:text-error transition-colors"
          >
            <LogOut size={14} />
            Sign out
          </button>
        </div>
      )}
    </div>
  )
}
