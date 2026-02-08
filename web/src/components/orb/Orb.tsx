import { cn } from '../../lib/utils'
import type { OrbProps } from '../../types'
import { OrbRings } from './OrbRings'

const SIZE_CLASSES = {
  sm: 'w-16 h-16',
  md: 'w-32 h-32',
  lg: 'w-48 h-48',
  xl: 'w-72 h-72',
}

const INNER_SIZE_CLASSES = {
  sm: 'w-12 h-12',
  md: 'w-24 h-24',
  lg: 'w-36 h-36',
  xl: 'w-56 h-56',
}

const IMAGE_SIZE_CLASSES = {
  sm: 'w-10 h-10',
  md: 'w-20 h-20',
  lg: 'w-28 h-28',
  xl: 'w-44 h-44',
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

  // Iridescent gradient backgrounds per state
  const getGradientStyle = () => {
    if (isListening) {
      return {
        background: 'radial-gradient(circle at 40% 40%, rgba(168,85,247,0.3), rgba(139,92,246,0.15), rgba(168,85,247,0.05))',
      }
    }
    if (isSpeaking) {
      return {
        background: 'radial-gradient(circle at 40% 40%, rgba(34,197,94,0.3), rgba(16,185,129,0.15), rgba(34,197,94,0.05))',
      }
    }
    if (isThinking) {
      return {
        background: 'radial-gradient(circle at 40% 40%, rgba(6,182,212,0.3), rgba(14,165,233,0.15), rgba(6,182,212,0.05))',
      }
    }
    // idle
    return {
      background: 'radial-gradient(circle at 40% 40%, rgba(14,165,233,0.1), rgba(168,85,247,0.05), transparent)',
    }
  }

  return (
    <Component
      onClick={onClick}
      className={cn(
        'relative rounded-full flex items-center justify-center',
        'transition-all duration-300 cursor-pointer',
        'focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500/50',
        SIZE_CLASSES[size],
        // Iridescent animation
        !isIdle && 'animate-orb-iridescent',
        // Idle breathing animation
        isIdle && 'animate-breathe'
      )}
      style={getGradientStyle()}
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

      {/* Glass shine overlay */}
      <div
        className="absolute inset-0 rounded-full pointer-events-none overflow-hidden"
      >
        <div
          className="absolute inset-0 animate-glass-shine opacity-30"
          style={{
            background: 'linear-gradient(135deg, transparent 30%, rgba(255,255,255,0.15) 45%, transparent 60%)',
            width: '200%',
            height: '200%',
          }}
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

export { OrbRings }
