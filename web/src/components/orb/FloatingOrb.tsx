import { cn } from '../../lib/utils'
import type { OrbState } from '../../types'

interface FloatingOrbProps {
  state: OrbState
  volume: number
  isStreaming: boolean
  onClick: () => void
}

export function FloatingOrb({ state, volume, isStreaming, onClick }: FloatingOrbProps) {
  const isIdle = state === 'idle'
  const isListening = state === 'listening'
  const isSpeaking = state === 'speaking'
  const isThinking = state === 'thinking'
  const isActive = !isIdle || isStreaming

  // Pulse scale based on volume/activity
  const pulseScale = isListening
    ? 1 + volume * 0.12
    : isSpeaking
      ? 1 + volume * 0.08
      : 1

  return (
    <button
      onClick={onClick}
      className="fixed bottom-24 right-4 z-30 md:hidden group"
      title="Voice mode"
    >
      {/* Outer pulse rings - Iron Man arc reactor style */}
      {isActive && (
        <>
          <div
            className={cn(
              'absolute inset-0 -m-3 rounded-full animate-ping',
              isListening && 'border border-purple-500/30',
              isSpeaking && 'border border-emerald-500/30',
              (isThinking || isStreaming) && 'border border-cyan-500/30',
            )}
            style={{ animationDuration: '2s' }}
          />
          <div
            className={cn(
              'absolute inset-0 -m-6 rounded-full animate-ping',
              isListening && 'border border-purple-500/15',
              isSpeaking && 'border border-emerald-500/15',
              (isThinking || isStreaming) && 'border border-cyan-500/15',
            )}
            style={{ animationDuration: '3s' }}
          />
        </>
      )}

      {/* Spinning ring - arc reactor outer ring */}
      <div
        className={cn(
          'absolute inset-0 -m-1.5 rounded-full transition-opacity duration-500',
          isActive ? 'opacity-100' : 'opacity-0',
        )}
      >
        <div
          className={cn(
            'w-full h-full rounded-full',
            isActive && 'animate-spin-slow',
          )}
          style={{
            background: isListening
              ? 'conic-gradient(from 0deg, transparent, rgba(168,85,247,0.4), transparent 40%)'
              : isSpeaking
                ? 'conic-gradient(from 0deg, transparent, rgba(34,197,94,0.4), transparent 40%)'
                : 'conic-gradient(from 0deg, transparent, rgba(14,165,233,0.4), transparent 40%)',
          }}
        />
      </div>

      {/* Main orb body */}
      <div
        className={cn(
          'relative w-14 h-14 rounded-full flex items-center justify-center',
          'transition-all duration-300',
          'shadow-lg',
          // Background
          isIdle && !isStreaming && 'bg-surface-2 border border-border/30',
          isListening && 'bg-purple-500/20 border border-purple-500/40',
          isSpeaking && 'bg-emerald-500/20 border border-emerald-500/40',
          (isThinking || isStreaming) && 'bg-cyan-500/20 border border-cyan-500/40',
          // Glows
          isListening && 'shadow-[0_0_20px_rgba(168,85,247,0.4)]',
          isSpeaking && 'shadow-[0_0_20px_rgba(34,197,94,0.4)]',
          (isThinking || isStreaming) && 'shadow-[0_0_20px_rgba(14,165,233,0.4)]',
          // Idle
          isIdle && !isStreaming && 'animate-breathe group-hover:shadow-[0_0_15px_rgba(14,165,233,0.3)]',
        )}
        style={{ transform: `scale(${pulseScale})` }}
      >
        {/* Inner glow ring */}
        <div
          className={cn(
            'absolute inset-1 rounded-full',
            'transition-all duration-300',
            isListening && 'bg-purple-500/10 shadow-[inset_0_0_10px_rgba(168,85,247,0.2)]',
            isSpeaking && 'bg-emerald-500/10 shadow-[inset_0_0_10px_rgba(34,197,94,0.2)]',
            (isThinking || isStreaming) && 'bg-cyan-500/10 shadow-[inset_0_0_10px_rgba(14,165,233,0.2)]',
          )}
        />

        {/* Avatar */}
        <img
          src="/jarvis.jpeg"
          alt="Jarvis"
          className={cn(
            'relative w-9 h-9 rounded-full object-cover',
            'transition-all duration-300',
            isThinking && 'opacity-70 animate-pulse',
            isStreaming && !isThinking && 'opacity-80',
          )}
        />

        {/* Glass shine sweep */}
        <div className="absolute inset-0 rounded-full overflow-hidden pointer-events-none">
          <div
            className="absolute inset-0 animate-glass-shine opacity-20"
            style={{
              background: 'linear-gradient(135deg, transparent 30%, rgba(255,255,255,0.2) 45%, transparent 60%)',
              width: '200%',
              height: '200%',
            }}
          />
        </div>
      </div>

      {/* Thinking spinner ring */}
      {(isThinking || isStreaming) && (
        <div
          className="absolute inset-0 -m-1 rounded-full border-2 border-transparent border-t-cyan-500/60 animate-spin"
          style={{ animationDuration: '1s' }}
        />
      )}
    </button>
  )
}
