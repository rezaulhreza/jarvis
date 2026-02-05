import { useState, useCallback, useEffect } from 'react'
import { cn } from '../../lib/utils'
import { RotateCcw, Save, FileText, Loader2 } from 'lucide-react'

interface SystemInstructionsProps {
  value: string
  onChange: (value: string) => Promise<void>
  onReset: () => Promise<void>
  isDefault: boolean
}

const MAX_CHARS = 4000

export function SystemInstructions({
  value,
  onChange,
  onReset,
  isDefault,
}: SystemInstructionsProps) {
  const [localValue, setLocalValue] = useState(value)
  const [isSaving, setIsSaving] = useState(false)
  const [hasChanges, setHasChanges] = useState(false)

  // Sync with prop
  useEffect(() => {
    setLocalValue(value)
    setHasChanges(false)
  }, [value])

  const handleChange = useCallback((newValue: string) => {
    const trimmed = newValue.slice(0, MAX_CHARS)
    setLocalValue(trimmed)
    setHasChanges(trimmed !== value)
  }, [value])

  const handleSave = useCallback(async () => {
    if (!hasChanges) return
    setIsSaving(true)
    try {
      await onChange(localValue)
      setHasChanges(false)
    } finally {
      setIsSaving(false)
    }
  }, [localValue, hasChanges, onChange])

  const handleReset = useCallback(async () => {
    setIsSaving(true)
    try {
      await onReset()
      setHasChanges(false)
    } finally {
      setIsSaving(false)
    }
  }, [onReset])

  const charCount = localValue.length
  const charPercentage = (charCount / MAX_CHARS) * 100

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FileText size={18} className="text-cyan-400" />
          <h3 className="text-sm font-medium text-text">System Instructions</h3>
          {isDefault && (
            <span className="px-2 py-0.5 text-xs rounded-full bg-cyan-500/20 text-cyan-400">
              Default
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleReset}
            disabled={isSaving || isDefault}
            className={cn(
              'p-2 rounded-lg transition-colors',
              'text-text-muted hover:text-text hover:bg-surface-2',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
            title="Reset to default"
          >
            <RotateCcw size={16} />
          </button>
          <button
            onClick={handleSave}
            disabled={isSaving || !hasChanges}
            className={cn(
              'flex items-center gap-2 px-3 py-1.5 rounded-lg',
              'text-sm font-medium transition-colors',
              hasChanges
                ? 'bg-cyan-500 text-white hover:bg-cyan-400'
                : 'bg-surface-2 text-text-muted',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
          >
            {isSaving ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Save size={14} />
            )}
            Save
          </button>
        </div>
      </div>

      {/* Description */}
      <p className="text-xs text-text-muted">
        Customize how Jarvis behaves. These instructions are sent with every message.
        You can also edit <code className="px-1 py-0.5 bg-surface-2 rounded">~/.jarvis/soul.md</code> directly.
      </p>

      {/* Textarea */}
      <div className="relative">
        <textarea
          value={localValue}
          onChange={(e) => handleChange(e.target.value)}
          placeholder="Enter custom instructions for Jarvis..."
          rows={12}
          className={cn(
            'w-full p-4 rounded-xl resize-none',
            'bg-surface/50 border border-border/30',
            'text-text text-sm font-mono leading-relaxed',
            'placeholder:text-text-muted/40',
            'focus:outline-none focus:border-cyan-500/50',
            'transition-colors'
          )}
        />

        {/* Character count */}
        <div className="absolute bottom-3 right-3 flex items-center gap-2">
          <div
            className={cn(
              'text-xs',
              charPercentage > 90
                ? 'text-red-400'
                : charPercentage > 75
                  ? 'text-warning'
                  : 'text-text-muted/60'
            )}
          >
            {charCount.toLocaleString()} / {MAX_CHARS.toLocaleString()}
          </div>
        </div>
      </div>

      {/* Progress bar */}
      <div className="h-1 bg-surface-2 rounded-full overflow-hidden">
        <div
          className={cn(
            'h-full rounded-full transition-all duration-300',
            charPercentage > 90
              ? 'bg-red-500'
              : charPercentage > 75
                ? 'bg-warning'
                : 'bg-cyan-500'
          )}
          style={{ width: `${Math.min(charPercentage, 100)}%` }}
        />
      </div>

      {/* Tips */}
      <div className="p-3 rounded-lg bg-surface/30 border border-border/20">
        <p className="text-xs text-text-muted leading-relaxed">
          <strong className="text-text">Tips:</strong> Be specific about tone, format preferences,
          and areas of expertise. You can reference previous context with "Based on our conversation..."
          or set boundaries with "Always/Never...".
        </p>
      </div>
    </div>
  )
}
