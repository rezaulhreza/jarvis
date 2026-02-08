import { useRef, useCallback, useEffect, type KeyboardEvent } from 'react'
import { cn } from '../../lib/utils'
import { Send, Mic, MicOff, Paperclip, Square } from 'lucide-react'
import { FilePreview } from './FilePreview'
import type { UploadedFile } from '../../types'

interface UnifiedInputProps {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  onStop?: () => void
  onVoiceToggle: () => void
  onFilesAdded: (files: FileList | File[]) => void
  files: UploadedFile[]
  onRemoveFile: (id: string) => void
  isRecording: boolean
  isLoading: boolean
  isStreaming?: boolean
  disabled?: boolean
  placeholder?: string
}

export function UnifiedInput({
  value,
  onChange,
  onSend,
  onStop,
  onVoiceToggle,
  onFilesAdded,
  files,
  onRemoveFile,
  isRecording,
  isLoading,
  isStreaming = false,
  disabled = false,
  placeholder = 'Message Jarvis...',
}: UnifiedInputProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea based on content (max 2 visible lines for pill aesthetic)
  useEffect(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      const newHeight = Math.min(textarea.scrollHeight, 80)
      textarea.style.height = `${newHeight}px`
    }
  }, [value])

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        if (value.trim() && !isLoading) {
          onSend()
        }
      }
    },
    [value, isLoading, onSend]
  )

  const handleFileClick = useCallback(() => {
    fileInputRef.current?.click()
  }, [])

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files
      if (files?.length) {
        onFilesAdded(files)
      }
      e.target.value = ''
    },
    [onFilesAdded]
  )

  const hasFiles = files.length > 0
  const canSend = (value.trim() || hasFiles) && !isLoading

  return (
    <div className="bg-background/50 backdrop-blur-sm px-4 pb-4 pt-2">
      {/* File previews */}
      <FilePreview files={files} onRemove={onRemoveFile} />

      <div className="max-w-3xl mx-auto">
        <div
          className={cn(
            'flex items-end gap-2 p-2 rounded-full',
            'glass-card',
            'transition-colors',
            'focus-within:border-cyan-500/30'
          )}
        >
          {/* File upload button */}
          <button
            onClick={handleFileClick}
            disabled={disabled || isLoading}
            className={cn(
              'p-2 rounded-full transition-colors flex-shrink-0',
              'text-text-muted hover:text-text hover:bg-white/5',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
            title="Attach files"
          >
            <Paperclip size={18} />
          </button>

          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*,video/*,audio/*,.pdf,.txt,.md,.json"
            onChange={handleFileChange}
            className="hidden"
          />

          {/* Text input */}
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isRecording ? 'Listening...' : placeholder}
            disabled={disabled || isRecording}
            rows={1}
            className={cn(
              'flex-1 resize-none bg-transparent',
              'text-text placeholder:text-text-muted/50',
              'focus:outline-none',
              'min-h-[36px] max-h-[80px] py-2 px-1',
              'disabled:opacity-50',
              'overflow-y-auto'
            )}
          />

          {/* Voice toggle button */}
          <button
            onClick={onVoiceToggle}
            disabled={disabled || isLoading}
            className={cn(
              'p-2 rounded-full transition-all flex-shrink-0',
              isRecording
                ? 'bg-red-500 text-white shadow-glow-red-sm'
                : 'text-text-muted hover:text-text hover:bg-white/5',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
            title={isRecording ? 'Stop recording' : 'Start recording'}
          >
            {isRecording ? <MicOff size={18} /> : <Mic size={18} />}
          </button>

          {/* Send / Stop button - circular */}
          {(isLoading || isStreaming) ? (
            <button
              onClick={onStop}
              className={cn(
                'w-9 h-9 rounded-full flex items-center justify-center transition-all flex-shrink-0',
                'bg-red-500 text-white hover:bg-red-400 shadow-[0_0_12px_rgba(239,68,68,0.4)]'
              )}
              title="Stop generating"
            >
              <Square size={14} />
            </button>
          ) : (
            <button
              onClick={onSend}
              disabled={!canSend}
              className={cn(
                'w-9 h-9 rounded-full flex items-center justify-center transition-all flex-shrink-0',
                canSend
                  ? 'bg-cyan-500 text-white hover:bg-cyan-400 shadow-glow-cyan'
                  : 'bg-surface-2 text-text-muted',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
              title="Send message"
            >
              <Send size={16} />
            </button>
          )}
        </div>

        {/* Keyboard hint */}
        <div className="flex justify-center mt-2">
          <p className="text-xs text-text-muted/40">
            <kbd className="px-1 py-0.5 rounded bg-surface-2/50 text-text-muted/50 text-[10px]">Enter</kbd> send
            {' '}
            <kbd className="px-1 py-0.5 rounded bg-surface-2/50 text-text-muted/50 text-[10px]">Shift+Enter</kbd> new line
          </p>
        </div>
      </div>
    </div>
  )
}
