import { useCallback, useState, useRef, type DragEvent, type ReactNode } from 'react'
import { cn } from '../../lib/utils'
import { Upload } from 'lucide-react'

interface FileUploadZoneProps {
  children: ReactNode
  onFilesAdded: (files: FileList) => void
  disabled?: boolean
}

export function FileUploadZone({
  children,
  onFilesAdded,
  disabled = false,
}: FileUploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false)
  const dragCounter = useRef(0)

  const handleDragEnter = useCallback(
    (e: DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      if (disabled) return

      dragCounter.current++
      if (e.dataTransfer?.items?.length) {
        setIsDragging(true)
      }
    },
    [disabled]
  )

  const handleDragLeave = useCallback(
    (e: DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      if (disabled) return

      dragCounter.current--
      if (dragCounter.current === 0) {
        setIsDragging(false)
      }
    },
    [disabled]
  )

  const handleDragOver = useCallback(
    (e: DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
    },
    []
  )

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      if (disabled) return

      setIsDragging(false)
      dragCounter.current = 0

      const files = e.dataTransfer?.files
      if (files?.length) {
        onFilesAdded(files)
      }
    },
    [disabled, onFilesAdded]
  )

  // Handle paste events for images
  const handlePaste = useCallback(
    (e: ClipboardEvent) => {
      if (disabled) return

      const items = e.clipboardData?.items
      if (!items) return

      const files: File[] = []
      for (const item of items) {
        if (item.kind === 'file') {
          const file = item.getAsFile()
          if (file) files.push(file)
        }
      }

      if (files.length > 0) {
        const dataTransfer = new DataTransfer()
        files.forEach((f) => dataTransfer.items.add(f))
        onFilesAdded(dataTransfer.files)
      }
    },
    [disabled, onFilesAdded]
  )

  // Add paste listener on mount
  const containerRef = useRef<HTMLDivElement>(null)

  // Use effect-like pattern with ref callback
  const setContainerRef = useCallback(
    (node: HTMLDivElement | null) => {
      // Remove old listener
      if (containerRef.current) {
        document.removeEventListener('paste', handlePaste)
      }

      containerRef.current = node

      // Add new listener
      if (node) {
        document.addEventListener('paste', handlePaste)
      }
    },
    [handlePaste]
  )

  return (
    <div
      ref={setContainerRef}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      className="relative flex-1 flex flex-col min-h-0 overflow-hidden"
    >
      {children}

      {/* Drag overlay */}
      {isDragging && (
        <div
          className={cn(
            'absolute inset-0 z-50',
            'bg-cyan-500/10 backdrop-blur-sm',
            'border-2 border-dashed border-cyan-500/50 rounded-2xl',
            'flex items-center justify-center',
            'transition-all duration-200'
          )}
        >
          <div className="flex flex-col items-center gap-3 text-cyan-400">
            <div className="p-4 rounded-full bg-cyan-500/20">
              <Upload size={32} />
            </div>
            <p className="text-lg font-medium">Drop files here</p>
            <p className="text-sm text-text-muted">
              Images, documents, audio, video
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
