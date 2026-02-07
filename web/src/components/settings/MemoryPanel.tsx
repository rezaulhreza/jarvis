import { useState, useEffect, useCallback } from 'react'
import { cn } from '../../lib/utils'
import {
  Trash2,
  Pencil,
  Plus,
  Brain,
  Save,
  X,
  Loader2,
  AlertCircle,
  Search,
} from 'lucide-react'

interface Memory {
  id: number
  category: string
  fact: string
  date: string | null
  section: string
}

const CATEGORIES = [
  'Job',
  'Preference',
  'Tech',
  'Location',
  'Project',
  'Identity',
  'Education',
  'Other',
] as const

type Category = (typeof CATEGORIES)[number]

const CATEGORY_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  Job: { bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/30' },
  'Current Employment': { bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/30' },
  'Previous Experience': { bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/30' },
  'Professional Summary': { bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/30' },
  Preference: { bg: 'bg-purple-500/20', text: 'text-purple-400', border: 'border-purple-500/30' },
  Tech: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30' },
  'Technical Skills': { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30' },
  'Primary Stack (Expert Level)': { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30' },
  'Currently Learning': { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30' },
  'Other Technologies': { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30' },
  Frontend: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30' },
  Databases: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30' },
  'DevOps and Infrastructure': { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30' },
  Location: { bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500/30' },
  Project: { bg: 'bg-cyan-500/20', text: 'text-cyan-400', border: 'border-cyan-500/30' },
  Identity: { bg: 'bg-pink-500/20', text: 'text-pink-400', border: 'border-pink-500/30' },
  Education: { bg: 'bg-indigo-500/20', text: 'text-indigo-400', border: 'border-indigo-500/30' },
  Learned: { bg: 'bg-teal-500/20', text: 'text-teal-400', border: 'border-teal-500/30' },
  Other: { bg: 'bg-gray-500/20', text: 'text-gray-400', border: 'border-gray-500/30' },
}

function getCategoryStyle(category: string) {
  return (
    CATEGORY_COLORS[category] ||
    CATEGORY_COLORS['Other']
  )
}

export function MemoryPanel() {
  const [memories, setMemories] = useState<Memory[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [searchQuery, setSearchQuery] = useState('')

  // Add form state
  const [newCategory, setNewCategory] = useState<Category>('Other')
  const [newFact, setNewFact] = useState('')
  const [adding, setAdding] = useState(false)

  // Edit state
  const [editingId, setEditingId] = useState<number | null>(null)
  const [editFact, setEditFact] = useState('')
  const [editCategory, setEditCategory] = useState<string>('')
  const [saving, setSaving] = useState(false)

  // Delete confirmation
  const [confirmDeleteId, setConfirmDeleteId] = useState<number | null>(null)

  const fetchMemories = useCallback(async () => {
    try {
      setLoading(true)
      const res = await fetch('/api/memories')
      if (res.ok) {
        const data = await res.json()
        setMemories(data.memories || [])
      } else {
        setError('Failed to load memories')
      }
    } catch {
      setError('Network error loading memories')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchMemories()
  }, [fetchMemories])

  const handleAdd = async () => {
    if (!newFact.trim()) return
    setAdding(true)
    setError('')
    try {
      const res = await fetch('/api/memories', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ category: newCategory, fact: newFact.trim() }),
      })
      if (res.ok) {
        setNewFact('')
        setNewCategory('Other')
        await fetchMemories()
      } else {
        const data = await res.json()
        setError(data.error || 'Failed to add memory')
      }
    } catch {
      setError('Network error')
    } finally {
      setAdding(false)
    }
  }

  const handleEdit = (memory: Memory) => {
    setEditingId(memory.id)
    setEditFact(memory.fact)
    setEditCategory(memory.category)
    setConfirmDeleteId(null)
  }

  const handleSaveEdit = async () => {
    if (editingId === null || !editFact.trim()) return
    setSaving(true)
    setError('')
    try {
      const res = await fetch(`/api/memories/${editingId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ category: editCategory, fact: editFact.trim() }),
      })
      if (res.ok) {
        setEditingId(null)
        await fetchMemories()
      } else {
        const data = await res.json()
        setError(data.error || 'Failed to update memory')
      }
    } catch {
      setError('Network error')
    } finally {
      setSaving(false)
    }
  }

  const handleCancelEdit = () => {
    setEditingId(null)
    setEditFact('')
    setEditCategory('')
  }

  const handleDelete = async (id: number) => {
    setError('')
    try {
      const res = await fetch(`/api/memories/${id}`, { method: 'DELETE' })
      if (res.ok) {
        setConfirmDeleteId(null)
        await fetchMemories()
      } else {
        const data = await res.json()
        setError(data.error || 'Failed to delete memory')
      }
    } catch {
      setError('Network error')
    }
  }

  const filteredMemories = searchQuery
    ? memories.filter(
        (m) =>
          m.fact.toLowerCase().includes(searchQuery.toLowerCase()) ||
          m.category.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : memories

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-6 h-6 animate-spin text-text-muted" />
      </div>
    )
  }

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Brain size={18} className="text-purple-400" />
        <h3 className="text-sm font-medium text-text">Memories</h3>
        <span className="text-xs px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-400">
          {memories.length}
        </span>
      </div>

      <p className="text-xs text-text-muted">
        Facts Jarvis has learned about you. These are used to personalize responses.
        You can also edit <code className="px-1 py-0.5 bg-surface-2 rounded">~/.jarvis/memory/facts.md</code> directly.
      </p>

      {/* Error display */}
      {error && (
        <div className="p-3 rounded-xl bg-error/20 border border-error/30 text-error text-sm flex items-center gap-2">
          <AlertCircle size={16} />
          {error}
          <button onClick={() => setError('')} className="ml-auto p-1 hover:bg-error/20 rounded">
            <X size={14} />
          </button>
        </div>
      )}

      {/* Add new memory form */}
      <div className="p-4 rounded-xl bg-surface border border-border/20 space-y-3">
        <div className="text-sm font-medium text-text">Add Memory</div>
        <div className="flex gap-2">
          <select
            value={newCategory}
            onChange={(e) => setNewCategory(e.target.value as Category)}
            className={cn(
              'px-3 py-2 rounded-lg text-sm',
              'bg-surface-2 border border-border/30 text-text',
              'focus:outline-none focus:border-purple-500/50',
              'appearance-none cursor-pointer'
            )}
          >
            {CATEGORIES.map((cat) => (
              <option key={cat} value={cat}>
                {cat}
              </option>
            ))}
          </select>
          <input
            type="text"
            value={newFact}
            onChange={(e) => setNewFact(e.target.value)}
            placeholder="Enter a fact..."
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleAdd()
              }
            }}
            className={cn(
              'flex-1 px-3 py-2 rounded-lg text-sm',
              'bg-surface-2 border border-border/30 text-text',
              'placeholder:text-text-muted/40',
              'focus:outline-none focus:border-purple-500/50'
            )}
          />
          <button
            onClick={handleAdd}
            disabled={!newFact.trim() || adding}
            className={cn(
              'flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium',
              'bg-purple-500/20 text-purple-400',
              'hover:bg-purple-500/30 transition-colors',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
          >
            {adding ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Plus size={14} />
            )}
            Add
          </button>
        </div>
      </div>

      {/* Search */}
      {memories.length > 5 && (
        <div className="relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search memories..."
            className={cn(
              'w-full pl-9 pr-3 py-2 rounded-lg text-sm',
              'bg-surface border border-border/20 text-text',
              'placeholder:text-text-muted/40',
              'focus:outline-none focus:border-purple-500/50'
            )}
          />
        </div>
      )}

      {/* Memory list */}
      {filteredMemories.length === 0 ? (
        <div className="py-10 text-center">
          <Brain size={40} className="mx-auto text-text-muted/30 mb-3" />
          <p className="text-sm text-text-muted">
            {searchQuery
              ? 'No memories match your search.'
              : 'No memories yet. Start chatting and I\'ll learn about you.'}
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {filteredMemories.map((memory) => {
            const style = getCategoryStyle(memory.category)
            const isEditing = editingId === memory.id
            const isConfirmingDelete = confirmDeleteId === memory.id

            return (
              <div
                key={memory.id}
                className={cn(
                  'p-3 rounded-xl border transition-all',
                  isEditing
                    ? 'bg-surface border-purple-500/30'
                    : 'bg-surface border-border/20 hover:border-border/40'
                )}
              >
                {isEditing ? (
                  /* Edit mode */
                  <div className="space-y-2">
                    <div className="flex gap-2">
                      <select
                        value={editCategory}
                        onChange={(e) => setEditCategory(e.target.value)}
                        className={cn(
                          'px-2 py-1.5 rounded-lg text-xs',
                          'bg-surface-2 border border-border/30 text-text',
                          'focus:outline-none focus:border-purple-500/50'
                        )}
                      >
                        {CATEGORIES.map((cat) => (
                          <option key={cat} value={cat}>
                            {cat}
                          </option>
                        ))}
                        {/* Include current category if not in standard list */}
                        {!CATEGORIES.includes(editCategory as Category) && (
                          <option value={editCategory}>{editCategory}</option>
                        )}
                      </select>
                      <input
                        type="text"
                        value={editFact}
                        onChange={(e) => setEditFact(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') handleSaveEdit()
                          if (e.key === 'Escape') handleCancelEdit()
                        }}
                        autoFocus
                        className={cn(
                          'flex-1 px-3 py-1.5 rounded-lg text-sm',
                          'bg-surface-2 border border-border/30 text-text',
                          'focus:outline-none focus:border-purple-500/50'
                        )}
                      />
                    </div>
                    <div className="flex justify-end gap-2">
                      <button
                        onClick={handleCancelEdit}
                        className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs text-text-muted hover:text-text hover:bg-surface-2 transition-colors"
                      >
                        <X size={12} />
                        Cancel
                      </button>
                      <button
                        onClick={handleSaveEdit}
                        disabled={saving || !editFact.trim()}
                        className={cn(
                          'flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs',
                          'bg-purple-500/20 text-purple-400 hover:bg-purple-500/30',
                          'disabled:opacity-50 disabled:cursor-not-allowed',
                          'transition-colors'
                        )}
                      >
                        {saving ? (
                          <Loader2 size={12} className="animate-spin" />
                        ) : (
                          <Save size={12} />
                        )}
                        Save
                      </button>
                    </div>
                  </div>
                ) : (
                  /* Display mode */
                  <div className="flex items-start gap-3">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span
                          className={cn(
                            'inline-flex px-2 py-0.5 rounded-full text-xs font-medium border',
                            style.bg,
                            style.text,
                            style.border
                          )}
                        >
                          {memory.category}
                        </span>
                        {memory.date && (
                          <span className="text-xs text-text-muted/60">
                            {memory.date}
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-text leading-relaxed">{memory.fact}</p>
                    </div>
                    <div className="flex items-center gap-1 flex-shrink-0">
                      {isConfirmingDelete ? (
                        <>
                          <button
                            onClick={() => handleDelete(memory.id)}
                            className="px-2 py-1 rounded-lg text-xs bg-error/20 text-error hover:bg-error/30 transition-colors"
                          >
                            Confirm
                          </button>
                          <button
                            onClick={() => setConfirmDeleteId(null)}
                            className="px-2 py-1 rounded-lg text-xs text-text-muted hover:text-text hover:bg-surface-2 transition-colors"
                          >
                            Cancel
                          </button>
                        </>
                      ) : (
                        <>
                          <button
                            onClick={() => handleEdit(memory)}
                            className="p-1.5 rounded-lg text-text-muted hover:text-text hover:bg-surface-2 transition-colors"
                            title="Edit"
                          >
                            <Pencil size={14} />
                          </button>
                          <button
                            onClick={() => {
                              setConfirmDeleteId(memory.id)
                              setEditingId(null)
                            }}
                            className="p-1.5 rounded-lg text-text-muted hover:text-error hover:bg-error/10 transition-colors"
                            title="Delete"
                          >
                            <Trash2 size={14} />
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
