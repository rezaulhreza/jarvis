import { cn } from '../../lib/utils'
import type { OrbState } from '../../types'

interface OrbRingsProps {
  state: OrbState
  volume: number
  playbackVolume: number
}

export function OrbRings({ state, volume, playbackVolume }: OrbRingsProps) {
  const isListening = state === 'listening'
  const isSpeaking = state === 'speaking'
  const isThinking = state === 'thinking'

  // Calculate scale based on state and volume
  const getScale = () => {
    if (isSpeaking) return 1 + playbackVolume * 0.3
    if (isListening) return 1 + volume * 0.5
    return 1
  }

  return (
    <>
      {/* Outer glow ring */}
      <div
        className={cn(
          'absolute inset-0 rounded-full transition-all duration-100',
          'pointer-events-none',
          isListening && 'border-2 border-purple-500/50',
          isSpeaking && 'border-2 border-emerald-500/50',
          isThinking && 'border-2 border-cyan-500/30 animate-spin-slow',
          !isListening && !isSpeaking && !isThinking && 'border border-white/5'
        )}
        style={{
          transform: `scale(${getScale()})`,
          opacity: volume > 0.01 || playbackVolume > 0.1 ? 1 : 0.5,
        }}
      />

      {/* Middle glow ring */}
      <div
        className={cn(
          'absolute inset-2 rounded-full transition-all duration-75',
          'pointer-events-none',
          isListening && 'border border-purple-400/40',
          isSpeaking && 'border border-emerald-400/40',
          isThinking && 'border border-cyan-400/20 animate-pulse'
        )}
        style={{
          transform: `scale(${1 + (getScale() - 1) * 0.7})`,
        }}
      />

      {/* Inner subtle ring */}
      <div
        className={cn(
          'absolute inset-4 rounded-full transition-all duration-50',
          'pointer-events-none',
          isListening && volume > 0.02 && 'border border-purple-300/30',
          isSpeaking && playbackVolume > 0.1 && 'border border-emerald-300/30'
        )}
      />

      {/* Iridescent conic gradient halo ring - visible during listening/speaking */}
      {(isListening || isSpeaking) && (
        <div
          className="absolute -inset-3 rounded-full pointer-events-none animate-spin-slow"
          style={{
            background: 'conic-gradient(from 0deg, rgba(168,85,247,0.3), rgba(6,182,212,0.3), rgba(34,197,94,0.3), rgba(168,85,247,0.3))',
            filter: 'blur(8px)',
            opacity: isListening ? Math.min(volume * 5, 0.8) : Math.min(playbackVolume * 3, 0.8),
          }}
        />
      )}

      {/* Thinking spinner overlay */}
      {isThinking && (
        <div className="absolute inset-0 rounded-full overflow-hidden pointer-events-none">
          <div className="absolute inset-0 bg-gradient-conic from-cyan-500/20 via-transparent to-cyan-500/20 animate-spin-slow" />
        </div>
      )}
    </>
  )
}
