import { useRef, useCallback, useEffect, type KeyboardEvent } from 'react'
import { cn } from '../../lib/utils'
import { Send, Mic, MicOff, Paperclip } from 'lucide-react'
import { FilePreview } from './FilePreview'
import type { UploadedFile } from '../../types'

interface UnifiedInputProps {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  onVoiceToggle: () => void
  onFilesAdded: (files: FileList | File[]) => void
  files: UploadedFile[]
  onRemoveFile: (id: string) => void
  isRecording: boolean
  isLoading: boolean
  disabled?: boolean
  placeholder?: string
}

export function UnifiedInput({
  value,
  onChange,
  onSend,
  onVoiceToggle,
  onFilesAdded,
  files,
  onRemoveFile,
  isRecording,
  isLoading,
  disabled = false,
  placeholder = 'Message Jarvis...',
}: UnifiedInputProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea based on content
  useEffect(() => {
    const textarea = textareaRef.current
    if (textarea) {
      // Reset height to auto to get the correct scrollHeight
      textarea.style.height = 'auto'
      // Set height to scrollHeight, capped at max-height
      const newHeight = Math.min(textarea.scrollHeight, 200)
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
      // Reset input so same file can be selected again
      e.target.value = ''
    },
    [onFilesAdded]
  )

  const hasFiles = files.length > 0
  const canSend = (value.trim() || hasFiles) && !isLoading

  return (
    <div className="border-t border-border/30 bg-background/50 backdrop-blur-sm">
      {/* File previews */}
      <FilePreview files={files} onRemove={onRemoveFile} />

      {/* Input row */}
      <div className="p-4">
        <div className="max-w-3xl mx-auto">
          <div
            className={cn(
              'flex items-end gap-2 p-2 rounded-2xl',
              'bg-surface/50 border border-border/30',
              'transition-colors',
              'focus-within:border-cyan-500/50 focus-within:bg-surface/80'
            )}
          >
            {/* File upload button */}
            <button
              onClick={handleFileClick}
              disabled={disabled || isLoading}
              className={cn(
                'p-2.5 rounded-xl transition-colors',
                'text-text-muted hover:text-text hover:bg-surface-2',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
              title="Attach files"
            >
              <Paperclip size={20} />
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
                'min-h-[44px] max-h-[200px] py-2.5 px-1',
                'disabled:opacity-50',
                'overflow-y-auto'
              )}
            />

            {/* Voice toggle button */}
            <button
              onClick={onVoiceToggle}
              disabled={disabled || isLoading}
              className={cn(
                'p-2.5 rounded-xl transition-all',
                isRecording
                  ? 'bg-red-500 text-white shadow-glow-red'
                  : 'text-text-muted hover:text-text hover:bg-surface-2',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
              title={isRecording ? 'Stop recording' : 'Start recording'}
            >
              {isRecording ? <MicOff size={20} /> : <Mic size={20} />}
            </button>

            {/* Send button */}
            <button
              onClick={onSend}
              disabled={!canSend}
              className={cn(
                'p-2.5 rounded-xl transition-all',
                canSend
                  ? 'bg-cyan-500 text-white hover:bg-cyan-400 shadow-glow-cyan'
                  : 'bg-surface-2 text-text-muted',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
              title="Send message"
            >
              <Send size={20} />
            </button>
          </div>

          {/* Keyboard hint */}
          <div className="flex justify-center mt-2">
            <p className="text-xs text-text-muted/50">
              Press <kbd className="px-1.5 py-0.5 rounded bg-surface-2 text-text-muted">Enter</kbd> to send, <kbd className="px-1.5 py-0.5 rounded bg-surface-2 text-text-muted">Shift+Enter</kbd> for new line
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
