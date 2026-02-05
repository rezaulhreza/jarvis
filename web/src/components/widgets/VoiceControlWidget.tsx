import { cn } from '../../lib/utils'
import { Orb } from '../orb'
import type { OrbState } from '../../types'
import { Volume2, VolumeX } from 'lucide-react'

interface VoiceControlWidgetProps {
  className?: string
  orbState: OrbState
  volume: number
  playbackVolume: number
  onOrbClick: () => void
  wakeWord: string
  wakeWordEnabled: boolean
  isMuted: boolean
  onToggleMute: () => void
}

export function VoiceControlWidget({
  className = '',
  orbState,
  volume,
  playbackVolume,
  onOrbClick,
  wakeWord,
  wakeWordEnabled,
  isMuted,
  onToggleMute,
}: VoiceControlWidgetProps) {
  return (
    <div className={`flex flex-col items-center ${className}`}>
      {/* Orb */}
      <div className="relative">
        <Orb
          state={orbState}
          volume={volume}
          playbackVolume={playbackVolume}
          onClick={onOrbClick}
          size="md"
        />
      </div>

      {/* Status Text */}
      <p className="text-xs text-text-muted mt-4 text-center">
        {wakeWordEnabled ? (
          <>Say "<span className="text-listening">Hey {wakeWord}</span>" to activate</>
        ) : (
          'Click orb to activate'
        )}
      </p>

      {/* Controls */}
      <div className="flex items-center gap-2 mt-4 pt-4 border-t border-border/20 w-full justify-center">
        <button
          onClick={onToggleMute}
          className={cn(
            'flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs transition-all',
            isMuted
              ? 'bg-error/20 text-error'
              : 'bg-surface-2 text-text-muted hover:text-text'
          )}
        >
          {isMuted ? <VolumeX size={14} /> : <Volume2 size={14} />}
          Voice {isMuted ? 'Off' : 'On'}
        </button>
      </div>
    </div>
  )
}
