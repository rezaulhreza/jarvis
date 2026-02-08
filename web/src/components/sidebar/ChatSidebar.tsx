import { useState, useEffect, useCallback } from 'react'
import { Plus, Search, PanelLeftClose, PanelLeft, Trash2 } from 'lucide-react'
import { ChatItem } from './ChatItem'
import { UserMenu } from './UserMenu'
import { cn, apiFetch } from '../../lib/utils'

interface Chat {
  id: string
  title: string
  created_at: string
  updated_at: string
}

interface ChatGroup {
  label: string
  chats: Chat[]
}

function groupChats(chats: Chat[]): ChatGroup[] {
  const now = new Date()
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const yesterday = new Date(today.getTime() - 86400000)
  const lastWeek = new Date(today.getTime() - 7 * 86400000)
  const lastMonth = new Date(today.getTime() - 30 * 86400000)

  const groups: ChatGroup[] = [
    { label: 'Today', chats: [] },
    { label: 'Yesterday', chats: [] },
    { label: 'Last 7 days', chats: [] },
    { label: 'Last 30 days', chats: [] },
    { label: 'Older', chats: [] },
  ]

  for (const chat of chats) {
    const date = new Date(chat.updated_at || chat.created_at)
    if (date >= today) {
      groups[0].chats.push(chat)
    } else if (date >= yesterday) {
      groups[1].chats.push(chat)
    } else if (date >= lastWeek) {
      groups[2].chats.push(chat)
    } else if (date >= lastMonth) {
      groups[3].chats.push(chat)
    } else {
      groups[4].chats.push(chat)
    }
  }

  return groups.filter(g => g.chats.length > 0)
}

interface ChatSidebarProps {
  activeChatId: string | null
  onSelectChat: (chatId: string) => void
  onNewChat: () => void
  isOpen: boolean
  onToggle: () => void
}

export function ChatSidebar({ activeChatId, onSelectChat, onNewChat, isOpen, onToggle }: ChatSidebarProps) {
  const [chats, setChats] = useState<Chat[]>([])
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(true)
  const [showClearConfirm, setShowClearConfirm] = useState(false)

  const fetchChats = useCallback(async () => {
    try {
      const res = await fetch('/api/chats')
      if (res.ok) {
        const data = await res.json()
        setChats(data.chats || [])
      }
    } catch {
      // ignore
    }
    setLoading(false)
  }, [])

  useEffect(() => {
    fetchChats()
  }, [fetchChats])

  // Refresh chat list when active chat changes
  useEffect(() => {
    if (activeChatId) {
      fetchChats()
    }
  }, [activeChatId, fetchChats])

  const handleDelete = async (chatId: string) => {
    try {
      const res = await apiFetch(`/api/chats/${chatId}`, { method: 'DELETE' })
      if (res.ok) {
        setChats(prev => prev.filter(c => c.id !== chatId))
        if (activeChatId === chatId) {
          onNewChat()
        }
      }
    } catch {
      // ignore
    }
  }

  const handleRename = async (chatId: string, newTitle: string) => {
    try {
      const res = await apiFetch(`/api/chats/${chatId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: newTitle }),
      })
      if (res.ok) {
        setChats(prev => prev.map(c => c.id === chatId ? { ...c, title: newTitle } : c))
      }
    } catch {
      // ignore
    }
  }

  const handleClearAll = async () => {
    try {
      const res = await apiFetch('/api/chats', { method: 'DELETE' })
      if (res.ok) {
        setChats([])
        setShowClearConfirm(false)
        onNewChat()
      }
    } catch {
      // ignore
    }
  }

  const filtered = search
    ? chats.filter(c => c.title.toLowerCase().includes(search.toLowerCase()))
    : chats

  const groups = groupChats(filtered)

  // Toggle button when sidebar is closed
  if (!isOpen) {
    return (
      <button
        onClick={onToggle}
        className="fixed top-3 left-3 z-30 p-2 rounded-lg bg-surface/80 backdrop-blur border border-border/20 text-text-muted hover:text-text transition-colors"
        title="Open sidebar"
      >
        <PanelLeft size={18} />
      </button>
    )
  }

  return (
    <>
      {/* Mobile overlay backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-30 md:hidden"
        onClick={onToggle}
      />

      <div className={cn(
        'h-full flex flex-col bg-surface/95 backdrop-blur-xl border-r border-border/20',
        'fixed inset-y-0 left-0 z-40 w-[280px]',
        'md:relative md:z-auto'
      )}>
        {/* Top bar */}
        <div className="flex items-center justify-between p-3 border-b border-border/10">
          <button
            onClick={onToggle}
            className="p-1.5 rounded-lg text-text-muted hover:text-text hover:bg-surface-2 transition-colors"
            title="Close sidebar"
          >
            <PanelLeftClose size={18} />
          </button>
          <button
            onClick={onNewChat}
            className="p-1.5 rounded-lg text-text-muted hover:text-text hover:bg-surface-2 transition-colors"
            title="New chat"
          >
            <Plus size={18} />
          </button>
        </div>

        {/* Search */}
        <div className="px-3 py-2">
          <div className="relative">
            <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted/50" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search chats..."
              className="w-full pl-8 pr-3 py-1.5 text-sm rounded-lg border border-border/30 bg-background/50 text-text placeholder:text-text-muted/40 focus:outline-none focus:ring-1 focus:ring-primary/30"
            />
          </div>
        </div>

        {/* Chat list */}
        <div className="flex-1 overflow-y-auto px-2 py-1">
          {loading ? (
            <div className="text-center text-text-muted/50 text-sm py-8">Loading...</div>
          ) : groups.length === 0 ? (
            <div className="text-center text-text-muted/50 text-sm py-8">
              {search ? 'No chats found' : 'No conversations yet'}
            </div>
          ) : (
            groups.map(group => (
              <div key={group.label} className="mb-2">
                <div className="px-2 py-1.5 text-[11px] font-medium text-text-muted/60 uppercase tracking-wider">
                  {group.label}
                </div>
                {group.chats.map(chat => (
                  <ChatItem
                    key={chat.id}
                    chat={chat}
                    isActive={chat.id === activeChatId}
                    onSelect={() => onSelectChat(chat.id)}
                    onDelete={() => handleDelete(chat.id)}
                    onRename={(title) => handleRename(chat.id, title)}
                  />
                ))}
              </div>
            ))
          )}
        </div>

        {/* Clear all chats */}
        {chats.length > 0 && (
          <div className="px-3 py-2 border-t border-border/10">
            {showClearConfirm ? (
              <div className="flex items-center gap-2">
                <span className="text-xs text-text-muted flex-1">Delete all chats?</span>
                <button
                  onClick={handleClearAll}
                  className="px-2 py-1 text-xs rounded bg-error/20 text-error hover:bg-error/30 transition-colors"
                >
                  Delete
                </button>
                <button
                  onClick={() => setShowClearConfirm(false)}
                  className="px-2 py-1 text-xs rounded bg-surface-2 text-text-muted hover:text-text transition-colors"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                onClick={() => setShowClearConfirm(true)}
                className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-sm text-text-muted hover:text-error hover:bg-error/10 transition-colors"
              >
                <Trash2 size={14} />
                <span>Clear all chats</span>
              </button>
            )}
          </div>
        )}

        {/* User menu at bottom */}
        <UserMenu />
      </div>
    </>
  )
}
