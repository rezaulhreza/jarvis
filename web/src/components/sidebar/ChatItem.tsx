import { useState, useRef, useEffect } from 'react'
import { MessageSquare, MoreHorizontal, Pencil, Trash2, Check, X } from 'lucide-react'
import { cn } from '../../lib/utils'

interface Chat {
  id: string
  title: string
  created_at: string
  updated_at: string
}

interface ChatItemProps {
  chat: Chat
  isActive: boolean
  onSelect: () => void
  onDelete: () => void
  onRename: (title: string) => void
}

export function ChatItem({ chat, isActive, onSelect, onDelete, onRename }: ChatItemProps) {
  const [editing, setEditing] = useState(false)
  const [editTitle, setEditTitle] = useState(chat.title)
  const [menuOpen, setMenuOpen] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus()
      inputRef.current.select()
    }
  }, [editing])

  // Close menu on outside click
  useEffect(() => {
    if (!menuOpen) return
    const handleClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [menuOpen])

  const handleSave = () => {
    const trimmed = editTitle.trim()
    if (trimmed && trimmed !== chat.title) {
      onRename(trimmed)
    }
    setEditing(false)
  }

  const handleCancel = () => {
    setEditTitle(chat.title)
    setEditing(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSave()
    if (e.key === 'Escape') handleCancel()
  }

  if (editing) {
    return (
      <div className="flex items-center gap-1 px-2 py-1.5 rounded-lg bg-surface-2">
        <input
          ref={inputRef}
          value={editTitle}
          onChange={(e) => setEditTitle(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={handleSave}
          className="flex-1 text-sm bg-transparent border-none outline-none text-text min-w-0"
        />
        <button onClick={handleSave} className="p-0.5 text-success hover:text-success/80">
          <Check size={14} />
        </button>
        <button onClick={handleCancel} className="p-0.5 text-text-muted hover:text-error">
          <X size={14} />
        </button>
      </div>
    )
  }

  return (
    <div className="relative group">
      <button
        onClick={onSelect}
        className={cn(
          'w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-left text-sm transition-colors',
          isActive
            ? 'bg-primary/10 text-text'
            : 'text-text-muted hover:bg-surface-2 hover:text-text'
        )}
      >
        <MessageSquare size={14} className="flex-shrink-0 opacity-50" />
        <span className="flex-1 truncate pr-6">{chat.title || 'New chat'}</span>
      </button>

      {/* Always-visible ellipsis menu trigger */}
      <div className="absolute right-1.5 top-1/2 -translate-y-1/2" ref={menuRef}>
        <button
          onClick={(e) => { e.stopPropagation(); setMenuOpen(!menuOpen) }}
          className={cn(
            'p-1 rounded-md transition-colors',
            menuOpen
              ? 'text-text bg-surface-2'
              : 'text-text-muted/40 hover:text-text-muted opacity-0 group-hover:opacity-100 focus:opacity-100',
            isActive && 'opacity-100 text-text-muted/60'
          )}
        >
          <MoreHorizontal size={14} />
        </button>

        {/* Dropdown menu */}
        {menuOpen && (
          <div className="absolute right-0 top-full mt-1 w-32 py-1 rounded-lg bg-surface-2 border border-border/30 shadow-lg z-50">
            <button
              onClick={(e) => {
                e.stopPropagation()
                setMenuOpen(false)
                setEditTitle(chat.title)
                setEditing(true)
              }}
              className="w-full flex items-center gap-2 px-3 py-1.5 text-sm text-text-muted hover:text-text hover:bg-surface/50 transition-colors"
            >
              <Pencil size={12} />
              <span>Rename</span>
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation()
                setMenuOpen(false)
                onDelete()
              }}
              className="w-full flex items-center gap-2 px-3 py-1.5 text-sm text-text-muted hover:text-error hover:bg-error/10 transition-colors"
            >
              <Trash2 size={12} />
              <span>Delete</span>
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
