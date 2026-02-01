import { useState, useEffect, useCallback } from 'react'
import { cn } from '../lib/utils'
import {
  MessageSquare,
  Plus,
  Search,
  Pencil,
  Trash2,
  Check,
  X,
  ChevronLeft,
  Loader2,
} from 'lucide-react'

interface Chat {
  id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
  preview?: string
}

interface ChatSidebarProps {
  isOpen: boolean
  onToggle: () => void
  currentChatId: string | null
  onSelectChat: (chatId: string) => void
  onNewChat: () => void
}

export function ChatSidebar({
  isOpen,
  onToggle,
  currentChatId,
  onSelectChat,
  onNewChat,
}: ChatSidebarProps) {
  const [chats, setChats] = useState<Chat[]>([])
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(true)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editTitle, setEditTitle] = useState('')

  // Fetch chats
  const fetchChats = useCallback(async () => {
    try {
      const params = new URLSearchParams()
      if (search) params.set('search', search)

      const res = await fetch(`/api/chats?${params}`)
      const data = await res.json()
      setChats(data.chats || [])
    } catch (e) {
      console.error('Failed to fetch chats:', e)
    } finally {
      setLoading(false)
    }
  }, [search])

  useEffect(() => {
    fetchChats()
  }, [fetchChats])

  // Refresh chats periodically when sidebar is open
  useEffect(() => {
    if (!isOpen) return
    const interval = setInterval(fetchChats, 5000)
    return () => clearInterval(interval)
  }, [isOpen, fetchChats])

  const handleDelete = async (chatId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirm('Delete this chat?')) return

    try {
      await fetch(`/api/chats/${chatId}`, { method: 'DELETE' })
      setChats(chats.filter(c => c.id !== chatId))
      if (currentChatId === chatId) {
        onNewChat()
      }
    } catch (e) {
      console.error('Failed to delete chat:', e)
    }
  }

  const handleEdit = (chat: Chat, e: React.MouseEvent) => {
    e.stopPropagation()
    setEditingId(chat.id)
    setEditTitle(chat.title)
  }

  const handleSaveEdit = async (chatId: string) => {
    try {
      await fetch(`/api/chats/${chatId}?title=${encodeURIComponent(editTitle)}`, {
        method: 'PATCH'
      })
      setChats(chats.map(c =>
        c.id === chatId ? { ...c, title: editTitle } : c
      ))
    } catch (e) {
      console.error('Failed to update chat:', e)
    }
    setEditingId(null)
  }

  const handleCancelEdit = () => {
    setEditingId(null)
    setEditTitle('')
  }

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const days = Math.floor(diff / (1000 * 60 * 60 * 24))

    if (days === 0) return 'Today'
    if (days === 1) return 'Yesterday'
    if (days < 7) return `${days} days ago`
    return date.toLocaleDateString()
  }

  return (
    <>
      {/* Overlay for mobile */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={onToggle}
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          'fixed left-0 top-0 h-full bg-[#0a0a0f] border-r border-[#2a2a3a] z-50 transition-transform duration-300 flex flex-col',
          'w-72',
          isOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        {/* Header */}
        <div className="p-4 border-b border-[#2a2a3a] flex items-center justify-between">
          <h2 className="font-semibold text-white">Chat History</h2>
          <button
            onClick={onToggle}
            className="p-2 rounded-lg hover:bg-[#1a1a24] text-[#71717a] hover:text-white"
          >
            <ChevronLeft size={18} />
          </button>
        </div>

        {/* New Chat Button */}
        <div className="p-3">
          <button
            onClick={() => {
              onNewChat()
              if (window.innerWidth < 1024) onToggle()
            }}
            className="w-full flex items-center gap-2 px-4 py-3 rounded-lg bg-blue-500 hover:bg-blue-600 text-white font-medium transition-colors"
          >
            <Plus size={18} />
            New Chat
          </button>
        </div>

        {/* Search */}
        <div className="px-3 pb-3">
          <div className="relative">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#51515a]" />
            <input
              type="text"
              placeholder="Search chats..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-9 pr-4 py-2 bg-[#1a1a24] border border-[#2a2a3a] rounded-lg text-white text-sm placeholder-[#51515a] focus:outline-none focus:border-blue-500"
            />
          </div>
        </div>

        {/* Chat List */}
        <div className="flex-1 overflow-y-auto px-3 pb-3">
          {loading ? (
            <div className="flex items-center justify-center py-8 text-[#51515a]">
              <Loader2 size={20} className="animate-spin" />
            </div>
          ) : chats.length === 0 ? (
            <div className="text-center py-8 text-[#51515a] text-sm">
              {search ? 'No chats found' : 'No chats yet'}
            </div>
          ) : (
            <div className="space-y-1">
              {chats.map((chat) => (
                <div
                  key={chat.id}
                  onClick={() => {
                    onSelectChat(chat.id)
                    if (window.innerWidth < 1024) onToggle()
                  }}
                  className={cn(
                    'group p-3 rounded-lg cursor-pointer transition-colors',
                    currentChatId === chat.id
                      ? 'bg-blue-500/20 border border-blue-500/30'
                      : 'hover:bg-[#1a1a24] border border-transparent'
                  )}
                >
                  {editingId === chat.id ? (
                    <div className="flex items-center gap-2">
                      <input
                        type="text"
                        value={editTitle}
                        onChange={(e) => setEditTitle(e.target.value)}
                        onClick={(e) => e.stopPropagation()}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') handleSaveEdit(chat.id)
                          if (e.key === 'Escape') handleCancelEdit()
                        }}
                        className="flex-1 px-2 py-1 bg-[#0a0a0f] border border-[#2a2a3a] rounded text-white text-sm focus:outline-none focus:border-blue-500"
                        autoFocus
                      />
                      <button
                        onClick={(e) => { e.stopPropagation(); handleSaveEdit(chat.id) }}
                        className="p-1 text-green-400 hover:text-green-300"
                      >
                        <Check size={14} />
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); handleCancelEdit() }}
                        className="p-1 text-red-400 hover:text-red-300"
                      >
                        <X size={14} />
                      </button>
                    </div>
                  ) : (
                    <>
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex items-center gap-2 min-w-0">
                          <MessageSquare size={14} className="text-[#51515a] shrink-0" />
                          <span className="text-white text-sm font-medium truncate">
                            {chat.title}
                          </span>
                        </div>
                        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
                          <button
                            onClick={(e) => handleEdit(chat, e)}
                            className="p-1 text-[#51515a] hover:text-white rounded"
                          >
                            <Pencil size={12} />
                          </button>
                          <button
                            onClick={(e) => handleDelete(chat.id, e)}
                            className="p-1 text-[#51515a] hover:text-red-400 rounded"
                          >
                            <Trash2 size={12} />
                          </button>
                        </div>
                      </div>
                      <div className="mt-1 flex items-center gap-2 text-xs text-[#51515a]">
                        <span>{formatDate(chat.updated_at)}</span>
                        <span>·</span>
                        <span>{chat.message_count} messages</span>
                      </div>
                      {chat.preview && (
                        <p className="mt-1 text-xs text-[#71717a] truncate">
                          {chat.preview}
                        </p>
                      )}
                    </>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </aside>

      {/* Toggle button when closed */}
      {!isOpen && (
        <button
          onClick={onToggle}
          className="fixed left-4 top-4 z-30 p-2 rounded-lg bg-[#1a1a24] border border-[#2a2a3a] text-[#71717a] hover:text-white hover:bg-[#2a2a3a] transition-colors"
          title="Open chat history"
        >
          <MessageSquare size={20} />
        </button>
      )}
    </>
  )
}
