import { cn } from '../../lib/utils'
import type { UploadedFile } from '../../types'
import { X, File, Image, Video, Music, Loader2 } from 'lucide-react'

interface FilePreviewProps {
  files: UploadedFile[]
  onRemove: (id: string) => void
}

export function FilePreview({ files, onRemove }: FilePreviewProps) {
  if (files.length === 0) return null

  return (
    <div className="flex gap-2 flex-wrap px-4 py-2 border-t border-border/30">
      {files.map((file) => (
        <FilePreviewItem key={file.id} file={file} onRemove={onRemove} />
      ))}
    </div>
  )
}

interface FilePreviewItemProps {
  file: UploadedFile
  onRemove: (id: string) => void
}

function FilePreviewItem({ file, onRemove }: FilePreviewItemProps) {
  const isImage = file.type === 'image'
  const isUploading = file.status === 'uploading' || file.status === 'pending'
  const isError = file.status === 'error'

  return (
    <div
      className={cn(
        'relative group rounded-lg overflow-hidden',
        'border transition-colors',
        isError
          ? 'border-red-500/50 bg-red-500/10'
          : isUploading
            ? 'border-cyan-500/30 bg-cyan-500/10'
            : 'border-border/50 bg-surface/50'
      )}
    >
      {/* Preview content */}
      <div className="w-16 h-16 flex items-center justify-center">
        {isImage && file.preview ? (
          <img
            src={file.preview}
            alt={file.name}
            className="w-full h-full object-cover"
          />
        ) : (
          <FileIcon type={file.type} />
        )}

        {/* Upload overlay */}
        {isUploading && (
          <div className="absolute inset-0 bg-background/80 flex items-center justify-center">
            <Loader2 size={20} className="animate-spin text-cyan-500" />
          </div>
        )}
      </div>

      {/* File name tooltip */}
      <div
        className={cn(
          'absolute bottom-0 left-0 right-0 p-1',
          'bg-gradient-to-t from-background/90 to-transparent',
          'text-[10px] text-text-muted truncate text-center'
        )}
      >
        {file.name.slice(0, 12)}
        {file.name.length > 12 && '...'}
      </div>

      {/* Remove button */}
      <button
        onClick={() => onRemove(file.id)}
        className={cn(
          'absolute top-1 right-1 p-1 rounded-full',
          'bg-background/80 text-text-muted',
          'opacity-0 group-hover:opacity-100',
          'transition-opacity hover:text-red-400'
        )}
      >
        <X size={12} />
      </button>

      {/* Error indicator */}
      {isError && (
        <div className="absolute inset-0 bg-red-500/20 flex items-center justify-center">
          <span className="text-red-400 text-xs">Error</span>
        </div>
      )}
    </div>
  )
}

function FileIcon({ type }: { type: string }) {
  const iconProps = { size: 24, className: 'text-text-muted' }

  switch (type) {
    case 'image':
      return <Image {...iconProps} />
    case 'video':
      return <Video {...iconProps} />
    case 'audio':
      return <Music {...iconProps} />
    default:
      return <File {...iconProps} />
  }
}
