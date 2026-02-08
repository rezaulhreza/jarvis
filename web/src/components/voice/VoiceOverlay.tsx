import { cn } from '../../lib/utils'
import { Orb } from '../orb'
import { getOrbState } from '../orb/getOrbState'
import { WaveformVisualizer } from './WaveformVisualizer'
import { Mic, MicOff, X, Keyboard, Camera, CameraOff, Square, Settings } from 'lucide-react'
import type { OrbState } from '../../types'

interface VoiceOverlayProps {
  isOpen: boolean
  connected: boolean
  model?: string
  isListening: boolean
  isRecording: boolean
  isPlaying: boolean
  isLoading: boolean
  volume: number
  playbackVolume: number
  interimTranscript: string
  lastMessage?: string
  streaming: string
  loadingText: string
  isCameraActive: boolean
  videoRef: React.RefObject<HTMLVideoElement | null>
  cameraError: string | null
  getFrequencyData: () => Uint8Array | null
  getPlaybackFrequencyData: () => Uint8Array | null
  onMicToggle: () => void
  onStopSpeaking: () => void
  onStopGeneration: () => void
  onInterruptAndListen: () => Promise<void>
  onCameraToggle: () => void
  onOpenSettings: () => void
  onClose: () => void
}

export function VoiceOverlay({
  isOpen,
  connected,
  model,
  isListening,
  isRecording,
  isPlaying,
  isLoading,
  volume,
  playbackVolume,
  interimTranscript,
  lastMessage,
  streaming,
  loadingText,
  isCameraActive,
  videoRef,
  cameraError,
  getFrequencyData,
  getPlaybackFrequencyData,
  onMicToggle,
  onStopSpeaking,
  onStopGeneration,
  onInterruptAndListen,
  onCameraToggle,
  onOpenSettings,
  onClose,
}: VoiceOverlayProps) {
  if (!isOpen) return null

  const orbState: OrbState = getOrbState({ isPlaying, isLoading, isListening, isRecording })

  // Determine waveform color and data source
  const waveformColor = isPlaying ? 'green' : isLoading ? 'cyan' : 'purple'
  const waveformActive = isListening || isPlaying || isLoading
  const waveformGetData = isPlaying ? getPlaybackFrequencyData : getFrequencyData

  // Status text
  const getStatusText = () => {
    if (isLoading) return loadingText || 'Thinking...'
    if (isPlaying) return 'Speaking...'
    if (isListening && volume > 0.02) return 'Listening...'
    if (isListening) return 'Speak now...'
    return 'Tap to speak'
  }

  // Large display text
  const getDisplayText = () => {
    if (interimTranscript) return interimTranscript
    if (streaming) return streaming
    if (isPlaying && lastMessage) return lastMessage
    return ''
  }

  // Mic button action
  const handleMicAction = () => {
    if (isLoading) {
      onStopGeneration()
    } else if (isPlaying) {
      onStopSpeaking()
    } else if (isListening) {
      onMicToggle()
    } else {
      onInterruptAndListen()
    }
  }

  const displayText = getDisplayText()

  return (
    <div className="fixed inset-0 z-40 animate-overlay-enter">
      {/* Background */}
      <div className="absolute inset-0 bg-[#030306] pointer-events-none">
        {/* Animated background gradient */}
        <div className={cn(
          'absolute inset-0 transition-all duration-1000',
          isListening && 'bg-gradient-radial from-purple-900/20 via-transparent to-transparent',
          isPlaying && 'bg-gradient-radial from-emerald-900/20 via-transparent to-transparent',
          isLoading && 'bg-gradient-radial from-cyan-900/20 via-transparent to-transparent',
          !isListening && !isPlaying && !isLoading && 'bg-gradient-radial from-slate-900/10 via-transparent to-transparent'
        )} />
        {/* Subtle grid overlay */}
        <div className="absolute inset-0 opacity-[0.02]" style={{
          backgroundImage: 'linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)',
          backgroundSize: '50px 50px'
        }} />
      </div>

      {/* Header bar */}
      <header className="relative flex items-center justify-between px-6 py-4 z-10">
        <div className="flex items-center gap-3">
          <div className={cn(
            'w-2 h-2 rounded-full transition-colors',
            connected ? 'bg-emerald-400 shadow-lg shadow-emerald-400/50' : 'bg-red-500'
          )} />
          <span className="text-sm text-white/60 font-light tracking-wide">
            {connected ? 'ONLINE' : 'OFFLINE'}
          </span>
          <span className="text-xs text-white/30 font-mono ml-2">
            {model?.split('/').pop() || 'AI'}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Camera toggle */}
          <button
            onClick={onCameraToggle}
            className={cn(
              'p-2 rounded-full transition-all',
              isCameraActive
                ? 'bg-cyan-500/20 text-cyan-400'
                : 'bg-white/5 text-white/50 hover:text-white hover:bg-white/10'
            )}
            title={isCameraActive ? 'Stop camera' : 'Start camera'}
          >
            {isCameraActive ? <CameraOff size={18} /> : <Camera size={18} />}
          </button>

          {/* Settings */}
          <button
            onClick={onOpenSettings}
            className="p-2 rounded-full bg-white/5 text-white/50 hover:text-white hover:bg-white/10 transition-all"
            title="Settings (model, voice, TTS)"
          >
            <Settings size={18} />
          </button>

          {/* Switch to chat */}
          <button
            onClick={onClose}
            className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 text-white/70 hover:text-white transition-all"
          >
            <Keyboard size={16} />
            <span className="text-sm">Chat</span>
          </button>
        </div>
      </header>

      {/* Center content */}
      <div className="relative z-10 flex-1 flex flex-col items-center justify-center" style={{ height: 'calc(100vh - 140px)' }}>
        {/* Extra pulse rings */}
        <div className="relative">
          {isListening && (
            <>
              <div className="absolute inset-0 -m-8 rounded-full border border-purple-500/20 animate-ping" style={{ animationDuration: '2s' }} />
              <div className="absolute inset-0 -m-16 rounded-full border border-purple-500/10 animate-ping" style={{ animationDuration: '3s' }} />
            </>
          )}
          {isPlaying && (
            <>
              <div className="absolute inset-0 -m-8 rounded-full border border-emerald-500/20 animate-ping" style={{ animationDuration: '1.5s' }} />
              <div className="absolute inset-0 -m-16 rounded-full border border-emerald-500/10 animate-ping" style={{ animationDuration: '2.5s' }} />
            </>
          )}
          {isLoading && (
            <div className="absolute inset-0 -m-12 rounded-full border-2 border-transparent border-t-cyan-500/50 animate-spin" style={{ animationDuration: '1s' }} />
          )}

          <Orb
            state={orbState}
            volume={volume}
            playbackVolume={playbackVolume}
            onClick={handleMicAction}
            size="xl"
          />
        </div>

        {/* Waveform */}
        <div className="mt-8 w-64">
          <WaveformVisualizer
            getFrequencyData={waveformGetData}
            isActive={waveformActive}
            color={waveformColor}
            barCount={32}
            height={48}
            volume={isListening ? volume : playbackVolume}
          />
        </div>

        {/* Status text */}
        <div className="mt-6 text-center">
          <p className={cn(
            'text-lg font-light transition-colors',
            isListening && volume > 0.02 && 'text-purple-400',
            isListening && volume <= 0.02 && 'text-white/40',
            isPlaying && 'text-emerald-400',
            isLoading && 'text-cyan-400 animate-pulse',
            !isListening && !isPlaying && !isLoading && 'text-white/30',
          )}>
            {getStatusText()}
          </p>
        </div>

        {/* Large transcript/response text */}
        {displayText && (
          <div className="mt-6 max-w-xl mx-auto px-8">
            <p className={cn(
              'text-xl font-light text-center leading-relaxed',
              isPlaying ? 'text-white/70' : 'text-white/50',
              interimTranscript && 'italic'
            )}>
              &ldquo;{displayText}&rdquo;
            </p>
          </div>
        )}
      </div>

      {/* Action bar */}
      <div className="relative z-10 flex justify-center items-center gap-6 pb-8">
        {/* Close button */}
        <button
          onClick={onClose}
          className="w-12 h-12 rounded-full flex items-center justify-center bg-white/5 border border-white/10 text-white/50 hover:text-white hover:bg-white/10 transition-all"
          title="Close voice mode"
        >
          <X size={20} />
        </button>

        {/* Main mic/stop button */}
        <button
          onClick={handleMicAction}
          className={cn(
            'w-16 h-16 rounded-full flex items-center justify-center transition-all',
            (isLoading || isPlaying) && 'bg-red-500/20 border-2 border-red-500 text-red-400 hover:bg-red-500/30',
            isListening && !isLoading && !isPlaying && 'bg-purple-500/20 border-2 border-purple-500 text-purple-400 hover:bg-purple-500/30',
            !isListening && !isPlaying && !isLoading && 'bg-white/10 border-2 border-white/30 text-white hover:bg-white/20',
          )}
          title={isLoading ? 'Stop generating' : isPlaying ? 'Stop speaking' : isListening ? 'Stop listening' : 'Start listening'}
        >
          {(isLoading || isPlaying) ? <Square size={24} /> : isListening ? <MicOff size={24} /> : <Mic size={24} />}
        </button>

        {/* Keyboard/chat button */}
        <button
          onClick={onClose}
          className="w-12 h-12 rounded-full flex items-center justify-center bg-white/5 border border-white/10 text-white/50 hover:text-white hover:bg-white/10 transition-all"
          title="Switch to chat"
        >
          <Keyboard size={20} />
        </button>
      </div>

      {/* Camera preview in voice mode */}
      {isCameraActive && (
        <div className="absolute bottom-32 right-6 z-20">
          <div className="relative rounded-2xl overflow-hidden border-2 border-cyan-500/40 shadow-lg shadow-cyan-500/20">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-[160px] h-[120px] object-cover"
              style={{ transform: 'scaleX(-1)' }}
            />
            <div className="absolute top-2 left-2 flex items-center gap-1 px-2 py-0.5 rounded-full bg-black/60 backdrop-blur-sm">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
              <span className="text-[10px] text-cyan-400 font-medium">LIVE</span>
            </div>
          </div>
        </div>
      )}

      {/* Camera error */}
      {cameraError && (
        <div className="absolute bottom-32 right-6 z-20 px-4 py-2 rounded-xl bg-red-500/20 border border-red-500/30 text-red-400 text-sm backdrop-blur-sm">
          {cameraError}
        </div>
      )}
    </div>
  )
}
