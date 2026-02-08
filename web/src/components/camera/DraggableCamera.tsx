import { cn } from '../../lib/utils'
import { useDraggable } from '../../hooks/useDraggable'
import { CameraOff } from 'lucide-react'

interface DraggableCameraProps {
  isActive: boolean
  videoRef: React.RefObject<HTMLVideoElement | null>
  onClose: () => void
}

export function DraggableCamera({ isActive, videoRef, onClose }: DraggableCameraProps) {
  const {
    position,
    size,
    isDragging,
    isResizing,
    dragHandleProps,
    resizeHandleProps,
  } = useDraggable({
    defaultPosition: { x: window.innerWidth - 180, y: window.innerHeight - 200 },
    defaultSize: { width: 160, height: 120 },
    minSize: { width: 120, height: 90 },
    maxSize: { width: 480, height: 360 },
    storageKey: 'jarvis_camera_position',
  })

  if (!isActive) return null

  return (
    <div
      className={cn(
        'fixed z-30 rounded-2xl overflow-hidden',
        'border-2 border-cyan-500/40 shadow-lg shadow-cyan-500/20',
        'select-none',
        isDragging && 'opacity-90',
      )}
      style={{
        left: position.x,
        top: position.y,
        width: size.width,
        height: size.height + 28, // +28 for the handle bar
      }}
    >
      {/* Drag handle bar */}
      <div
        {...dragHandleProps}
        className="h-7 bg-surface/80 backdrop-blur-sm flex items-center justify-between px-2"
      >
        <div className="flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
          <span className="text-[10px] text-cyan-400 font-medium">LIVE</span>
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation()
            onClose()
          }}
          className="w-5 h-5 rounded-full bg-white/10 flex items-center justify-center text-white/60 hover:text-white transition-colors"
        >
          <CameraOff size={10} />
        </button>
      </div>

      {/* Video */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full h-full object-cover"
        style={{
          transform: 'scaleX(-1)',
          height: size.height,
        }}
      />

      {/* Resize handles */}
      {(['top-left', 'top-right', 'bottom-left', 'bottom-right'] as const).map(corner => (
        <div
          key={corner}
          {...resizeHandleProps(corner)}
          className={cn(
            'absolute w-4 h-4 z-10',
            'opacity-0 hover:opacity-100 transition-opacity',
            corner === 'top-left' && 'top-0 left-0 cursor-nw-resize',
            corner === 'top-right' && 'top-0 right-0 cursor-ne-resize',
            corner === 'bottom-left' && 'bottom-0 left-0 cursor-sw-resize',
            corner === 'bottom-right' && 'bottom-0 right-0 cursor-se-resize',
          )}
        >
          <div className={cn(
            'absolute w-2 h-2 rounded-full bg-cyan-400/60',
            corner === 'top-left' && 'top-1 left-1',
            corner === 'top-right' && 'top-1 right-1',
            corner === 'bottom-left' && 'bottom-1 left-1',
            corner === 'bottom-right' && 'bottom-1 right-1',
          )} />
        </div>
      ))}

      {/* Dragging/resizing visual feedback */}
      {(isDragging || isResizing) && (
        <div className="absolute inset-0 border-2 border-cyan-400/60 rounded-2xl pointer-events-none" />
      )}
    </div>
  )
}
