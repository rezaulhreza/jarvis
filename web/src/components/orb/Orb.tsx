import { cn } from '../../lib/utils'
import type { OrbProps, OrbState } from '../../types'
import { OrbRings } from './OrbRings'

const SIZE_CLASSES = {
  sm: 'w-16 h-16',
  md: 'w-32 h-32',
  lg: 'w-48 h-48',
}

const INNER_SIZE_CLASSES = {
  sm: 'w-12 h-12',
  md: 'w-24 h-24',
  lg: 'w-36 h-36',
}

const IMAGE_SIZE_CLASSES = {
  sm: 'w-10 h-10',
  md: 'w-20 h-20',
  lg: 'w-28 h-28',
}

interface ExtendedOrbProps extends OrbProps {
  as?: 'button' | 'div'
}

export function Orb({
  state,
  volume,
  playbackVolume,
  onClick,
  size = 'lg',
  as: Component = 'button',
}: ExtendedOrbProps) {
  const isIdle = state === 'idle'
  const isListening = state === 'listening'
  const isSpeaking = state === 'speaking'
  const isThinking = state === 'thinking'

  // Dynamic scale based on volume
  const innerScale = isListening
    ? 1 + volume * 0.15
    : isSpeaking
      ? 1 + playbackVolume * 0.1
      : 1

  return (
    <Component
      onClick={onClick}
      className={cn(
        'relative rounded-full flex items-center justify-center',
        'transition-all duration-300 cursor-pointer',
        'focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500/50',
        SIZE_CLASSES[size],
        // Background glow based on state
        isIdle && 'bg-surface-2/80 hover:bg-surface-2',
        isListening && 'bg-purple-500/10 hover:bg-purple-500/20',
        isSpeaking && 'bg-emerald-500/10 hover:bg-emerald-500/20',
        isThinking && 'bg-cyan-500/10',
        // Idle breathing animation
        isIdle && 'animate-breathe'
      )}
    >
      {/* Animated rings */}
      <OrbRings
        state={state}
        volume={volume}
        playbackVolume={playbackVolume}
      />

      {/* Inner orb container */}
      <div
        className={cn(
          'rounded-full flex items-center justify-center',
          'transition-all duration-150',
          INNER_SIZE_CLASSES[size],
          isIdle && 'bg-surface-2',
          isListening && 'bg-purple-500/20',
          isSpeaking && 'bg-emerald-500/20',
          isThinking && 'bg-cyan-500/20 animate-pulse'
        )}
        style={{
          transform: `scale(${innerScale})`,
        }}
      >
        {/* Avatar image */}
        <img
          src="/jarvis.jpeg"
          alt="Jarvis"
          className={cn(
            'rounded-full object-cover',
            'transition-all duration-300',
            IMAGE_SIZE_CLASSES[size],
            isSpeaking && 'scale-105',
            isThinking && 'opacity-80'
          )}
        />
      </div>

      {/* Glow effect overlay */}
      <div
        className={cn(
          'absolute inset-0 rounded-full pointer-events-none',
          'transition-opacity duration-300',
          isListening && volume > 0.02 && 'shadow-glow-purple opacity-100',
          isSpeaking && playbackVolume > 0.1 && 'shadow-glow-green opacity-100',
          isThinking && 'shadow-glow-cyan opacity-50',
          (!isListening || volume <= 0.02) &&
            (!isSpeaking || playbackVolume <= 0.1) &&
            !isThinking &&
            'opacity-0'
        )}
      />
    </Component>
  )
}

// Helper function to determine orb state from component state
export function getOrbState(props: {
  isPlaying: boolean
  isLoading: boolean
  isListening: boolean
  isRecording: boolean
}): OrbState {
  const { isPlaying, isLoading, isListening, isRecording } = props

  if (isPlaying) return 'speaking'
  if (isLoading) return 'thinking'
  if (isListening || isRecording) return 'listening'
  return 'idle'
}

export { OrbRings }
