import { useState, useEffect, useCallback } from 'react'
import { useWebSocket } from './hooks/useWebSocket'
import { useVoice } from './hooks/useVoice'
import { useFileUpload } from './hooks/useFileUpload'
import { useWakeWord } from './hooks/useWakeWord'
import { cn } from './lib/utils'

// Components
import { MessageList } from './components/chat'
import { UnifiedInput, FileUploadZone } from './components/input'
import { SettingsPanel } from './components/settings'
import { Orb, getOrbState } from './components/orb'

// Icons
import {
  Settings,
  Zap,
  Scale,
  Brain,
  Trash2,
  Volume2,
  VolumeX,
  Mic,
  Keyboard,
  AudioWaveform,
} from 'lucide-react'

// Types
import type { ReasoningLevel, TTSProvider, STTProvider } from './types'

// Loading text options
const LOADING_TEXTS = [
  'Thinking...', 'Processing...', 'Analyzing...', 'Computing...',
  'On it...', 'One moment...', 'Working on it...', 'Let me check...',
]

export default function App() {
  // Assistant configuration (customizable name, etc.)
  const [assistantName, setAssistantName] = useState('Jarvis')

  // State
  const [mode, setMode] = useState<'chat' | 'voice'>('chat')
  const [reasoningLevel, setReasoningLevel] = useState<ReasoningLevel>(null)
  const [voiceOutput, setVoiceOutput] = useState(false)
  const [input, setInput] = useState('')
  const [lastSpokenIndex, setLastSpokenIndex] = useState(-1)
  const [loadingText, setLoadingText] = useState('')
  const [wakeWordEnabled, setWakeWordEnabled] = useState(false) // Disabled for now
  const [wakeWord, setWakeWord] = useState('jarvis')

  // Settings state
  const [showSettings, setShowSettings] = useState(false)
  const [models, setModels] = useState<string[]>([])
  const [providers, setProviders] = useState<Record<string, { configured: boolean; model?: string | null }>>({})
  const [voices, setVoices] = useState<{ id: string; name: string }[]>([])
  const [elevenVoices, setElevenVoices] = useState<{ id: string; name: string }[]>([])
  const [kokoroVoices, setKokoroVoices] = useState<{ id: string; name: string }[]>([])
  const [chutesConfigured, setChutesConfigured] = useState(false)
  const [ttsProvider, setTtsProvider] = useState<TTSProvider>('browser')
  const [sttProvider, setSttProvider] = useState<STTProvider>('browser')
  const [currentVoice, setCurrentVoice] = useState('en-GB-SoniaNeural')
  const [systemPrompt, setSystemPromptState] = useState('')
  const [isDefaultPrompt, setIsDefaultPrompt] = useState(true)

  // WebSocket hook
  const {
    connected,
    messages,
    streaming,
    streamingThinking,
    isLoading,
    project,
    model,
    provider,
    intentInfo,
    contextStats,
    send,
    clear,
    switchModel,
    switchProvider,
  } = useWebSocket()

  // File upload hook
  const { files, addFiles, removeFile, clearFiles, getAttachmentIds } = useFileUpload()

  // Voice callbacks
  const handleVoiceInput = useCallback((transcript: string) => {
    if (transcript && connected) {
      send(transcript, true)
    }
  }, [connected, send])

  const handleInterrupt = useCallback(() => {
    console.log('User interrupted')
  }, [])

  // Voice hook - for push-to-talk and voice mode
  const {
    isListening,
    isRecording,
    isPlaying,
    volume,
    playbackVolume,
    startListening,
    stopListening,
    startRecording,
    stopRecording,
    speak,
    stopSpeaking,
  } = useVoice({
    onSpeechEnd: handleVoiceInput,
    onInterrupt: handleInterrupt,
    sttProvider,
  })

  // Load assistant config from localStorage
  useEffect(() => {
    const savedName = localStorage.getItem('jarvis_assistant_name')
    if (savedName) setAssistantName(savedName)
    const savedWakeWord = localStorage.getItem('jarvis_wake_word')
    if (savedWakeWord) setWakeWord(savedWakeWord)
  }, [])

  // Wake word detection - disabled for now
  useWakeWord({
    keyword: wakeWord,
    onWakeWord: () => {},
    enabled: false,
  })

  // Randomize loading text
  useEffect(() => {
    if (isLoading) {
      setLoadingText(LOADING_TEXTS[Math.floor(Math.random() * LOADING_TEXTS.length)])
    }
  }, [isLoading])

  // Fetch initial data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [providersRes, voicesRes, elevenRes, kokoroRes, kokoroStatusRes, voiceSettingsRes, promptRes] = await Promise.all([
          fetch('/api/providers').catch(() => null),
          fetch('/api/voices').catch(() => null),
          fetch('/api/elevenlabs/voices').catch(() => null),
          fetch('/api/kokoro/voices').catch(() => null),
          fetch('/api/kokoro/status').catch(() => null),
          fetch('/api/settings/voice').catch(() => null),
          fetch('/api/system-prompt').catch(() => null),
        ])

        if (providersRes?.ok) {
          const data = await providersRes.json()
          setProviders(data.providers || {})
        }
        if (voicesRes?.ok) {
          const data = await voicesRes.json()
          setVoices(data.voices || [])
        }
        if (elevenRes?.ok) {
          const data = await elevenRes.json()
          if (data.voices?.length > 0) setElevenVoices(data.voices)
        }
        if (kokoroRes?.ok) {
          const data = await kokoroRes.json()
          if (data.voices?.length > 0) setKokoroVoices(data.voices)
        }
        if (kokoroStatusRes?.ok) {
          const data = await kokoroStatusRes.json()
          setChutesConfigured(data.configured || false)
        }
        if (voiceSettingsRes?.ok) {
          const data = await voiceSettingsRes.json()
          if (data.tts_provider) setTtsProvider(data.tts_provider)
          if (data.tts_voice) setCurrentVoice(data.tts_voice)
          if (data.stt_provider) setSttProvider(data.stt_provider)
        }
        if (promptRes?.ok) {
          const data = await promptRes.json()
          setSystemPromptState(data.content || '')
          setIsDefaultPrompt(data.isDefault ?? true)
        }
      } catch (error) {
        console.error('Failed to fetch initial data:', error)
      }
    }
    fetchData()
  }, [])

  // Fetch models when provider changes
  useEffect(() => {
    const activeProvider = provider || 'ollama'
    fetch(`/api/models?provider=${encodeURIComponent(activeProvider)}`)
      .then(res => res.json())
      .then(data => setModels(data.models || []))
      .catch(() => {})
  }, [provider])

  // Auto-speak assistant responses in voice mode or when voice output is enabled
  useEffect(() => {
    if ((mode === 'voice' || voiceOutput) && messages.length > 0) {
      const lastIndex = messages.length - 1
      const last = messages[lastIndex]
      if (last.role === 'assistant' && lastIndex > lastSpokenIndex) {
        speak(last.content, ttsProvider)
        setLastSpokenIndex(lastIndex)
      }
    }
  }, [messages, mode, voiceOutput, speak, lastSpokenIndex, ttsProvider])

  // Auto-resume listening in voice mode after speaking
  useEffect(() => {
    if (mode === 'voice' && !isPlaying && !isLoading && connected && !isListening) {
      const timer = setTimeout(() => {
        if (mode === 'voice' && !isPlaying && !isLoading) {
          startListening()
        }
      }, 500)
      return () => clearTimeout(timer)
    }
  }, [mode, isPlaying, isLoading, connected, isListening, startListening])

  // Stop listening when leaving voice mode
  useEffect(() => {
    if (mode !== 'voice' && isListening) {
      stopListening()
    }
  }, [mode, isListening, stopListening])

  // Voice mode handlers
  const enterVoiceMode = useCallback(() => {
    setMode('voice')
    stopSpeaking()
    startListening()
  }, [stopSpeaking, startListening])

  const exitVoiceMode = useCallback(() => {
    stopListening()
    stopSpeaking()
    setMode('chat')
  }, [stopListening, stopSpeaking])

  // Handlers
  const handleSend = useCallback(() => {
    const attachments = getAttachmentIds()
    // Allow sending with just attachments (no text)
    if ((!input.trim() && attachments.length === 0) || !connected) return
    send(input.trim(), true, reasoningLevel, attachments)
    setInput('')
    clearFiles()
  }, [input, connected, send, reasoningLevel, clearFiles, getAttachmentIds])

  const handleVoiceToggle = useCallback(async () => {
    if (isRecording) {
      const transcript = await stopRecording()
      if (transcript) {
        send(transcript, true)
      }
    } else {
      stopSpeaking()
      startRecording()
    }
  }, [isRecording, stopRecording, send, stopSpeaking, startRecording])

  // Settings handlers
  const handleSetTTSProvider = useCallback(async (p: TTSProvider) => {
    setTtsProvider(p)
    await fetch('/api/settings/voice', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tts_provider: p }),
    })
  }, [])

  const handleSetSTTProvider = useCallback(async (p: STTProvider) => {
    setSttProvider(p)
    await fetch('/api/settings/voice', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stt_provider: p }),
    })
  }, [])

  const handleSetVoice = useCallback(async (voiceId: string, p: 'edge' | 'elevenlabs' | 'kokoro') => {
    setCurrentVoice(voiceId)
    if (p === 'elevenlabs') {
      await fetch('/api/settings/elevenlabs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ voice_id: voiceId }),
      })
    } else if (p === 'kokoro') {
      await fetch('/api/settings/voice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kokoro_voice: voiceId }),
      })
    } else {
      await fetch('/api/settings/voice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ voice: voiceId }),
      })
    }
  }, [])

  const handleSetSystemPrompt = useCallback(async (prompt: string) => {
    setSystemPromptState(prompt)
    setIsDefaultPrompt(false)
    await fetch('/api/system-prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: prompt }),
    })
  }, [])

  const handleResetSystemPrompt = useCallback(async () => {
    const res = await fetch('/api/system-prompt/reset', { method: 'POST' })
    if (res.ok) {
      const data = await res.json()
      setSystemPromptState(data.content || '')
      setIsDefaultPrompt(true)
    }
  }, [])

  // Smart defaults: Auto-configure TTS when switching to Chutes
  // Note: Keep browser STT as default since it's more reliable
  useEffect(() => {
    if (provider === 'chutes' && chutesConfigured) {
      // Auto-set Kokoro TTS for best voice quality
      if (ttsProvider !== 'kokoro') {
        handleSetTTSProvider('kokoro')
      }
      // Keep browser STT - it's more reliable than Chutes STT
    }
  }, [provider, chutesConfigured, ttsProvider, handleSetTTSProvider])

  // Get orb state
  const orbState = getOrbState({ isPlaying, isLoading, isListening, isRecording })

  // ============================================
  // VOICE MODE - Futuristic Full Screen Interface
  // ============================================
  if (mode === 'voice') {
    return (
      <div className="h-full bg-[#030306] flex flex-col overflow-hidden relative">
        {/* Animated background gradient */}
        <div className="absolute inset-0 overflow-hidden">
          <div className={cn(
            'absolute inset-0 transition-all duration-1000',
            isListening && 'bg-gradient-radial from-purple-900/20 via-transparent to-transparent',
            isPlaying && 'bg-gradient-radial from-emerald-900/20 via-transparent to-transparent',
            isLoading && 'bg-gradient-radial from-cyan-900/20 via-transparent to-transparent',
            !isListening && !isPlaying && !isLoading && 'bg-gradient-radial from-slate-900/20 via-transparent to-transparent'
          )} />
          {/* Subtle grid overlay */}
          <div className="absolute inset-0 opacity-[0.02]" style={{
            backgroundImage: 'linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)',
            backgroundSize: '50px 50px'
          }} />
        </div>

        {/* Top bar */}
        <header className="relative flex items-center justify-between px-6 py-4 z-10">
          <div className="flex items-center gap-3">
            <div className={cn(
              'w-2 h-2 rounded-full transition-colors',
              connected ? 'bg-emerald-400 shadow-lg shadow-emerald-400/50' : 'bg-red-500'
            )} />
            <span className="text-sm text-white/60 font-light tracking-wide">
              {connected ? 'ONLINE' : 'OFFLINE'}
            </span>
          </div>

          <div className="flex items-center gap-4">
            {/* Model info */}
            <span className="text-xs text-white/40 font-mono">{model?.split('/').pop() || 'AI'}</span>

            {/* Back to chat */}
            <button
              onClick={exitVoiceMode}
              className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 text-white/70 hover:text-white transition-all"
            >
              <Keyboard size={16} />
              <span className="text-sm">Chat</span>
            </button>
          </div>
        </header>

        {/* Center - The Orb */}
        <div className="flex-1 flex flex-col items-center justify-center relative z-10">
          {/* Orb container with extra glow effects */}
          <div className="relative">
            {/* Outer pulse rings */}
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
              onClick={() => isListening ? stopListening() : startListening()}
              size="lg"
            />
          </div>

          {/* Status text */}
          <div className="mt-12 text-center space-y-3">
            <div className="h-8 flex items-center justify-center">
              {isLoading && (
                <p className="text-cyan-400 text-lg font-light animate-pulse">{loadingText}</p>
              )}
              {isPlaying && (
                <p className="text-emerald-400 text-lg font-light">Speaking...</p>
              )}
              {isListening && !isLoading && !isPlaying && (
                <p className={cn(
                  'text-lg font-light transition-colors',
                  volume > 0.02 ? 'text-purple-400' : 'text-white/40'
                )}>
                  {volume > 0.02 ? 'Listening...' : 'Speak now...'}
                </p>
              )}
              {!isListening && !isLoading && !isPlaying && (
                <p className="text-white/30 text-lg font-light">Tap orb to speak</p>
              )}
            </div>

            {/* Volume indicator */}
            {isListening && (
              <div className="flex items-center justify-center gap-1">
                {[...Array(7)].map((_, i) => (
                  <div
                    key={i}
                    className={cn(
                      'w-1 rounded-full bg-purple-500 transition-all duration-75',
                      volume > i * 0.12 ? 'opacity-100' : 'opacity-20'
                    )}
                    style={{ height: `${12 + i * 4}px` }}
                  />
                ))}
              </div>
            )}

            {/* Mic control - close to orb */}
            <div className="flex justify-center mt-6">
              <button
                onClick={() => isListening ? stopListening() : startListening()}
                className={cn(
                  'w-14 h-14 rounded-full flex items-center justify-center transition-all',
                  isListening
                    ? 'bg-purple-500/20 border-2 border-purple-500 text-purple-400'
                    : 'bg-white/5 border border-white/20 text-white/60 hover:bg-white/10 hover:text-white'
                )}
              >
                <Mic size={22} />
              </button>
            </div>
          </div>
        </div>

        {/* Last message preview */}
        {messages.length > 0 && (
          <div className="relative z-10 px-8 py-6">
            <div className="max-w-2xl mx-auto">
              <div className={cn(
                'p-4 rounded-2xl backdrop-blur-xl border transition-colors',
                messages[messages.length - 1].role === 'assistant'
                  ? 'bg-white/5 border-white/10'
                  : 'bg-purple-500/10 border-purple-500/20'
              )}>
                <p className="text-white/80 text-sm leading-relaxed line-clamp-3">
                  {messages[messages.length - 1].content}
                </p>
              </div>
            </div>
          </div>
        )}

        
        {/* Settings Panel */}
        <SettingsPanel
          isOpen={showSettings}
          onClose={() => setShowSettings(false)}
          currentModel={model}
          currentProvider={provider}
          providers={providers}
          models={models}
          onSwitchModel={switchModel}
          onSwitchProvider={switchProvider}
          voiceSettings={{
            tts_provider: ttsProvider,
            tts_voice: currentVoice,
            stt_provider: sttProvider,
          }}
          onSetTTSProvider={handleSetTTSProvider}
          onSetSTTProvider={handleSetSTTProvider}
          onSetVoice={handleSetVoice}
          edgeVoices={voices}
          elevenVoices={elevenVoices}
          kokoroVoices={kokoroVoices}
          wakeWord={wakeWord}
          wakeWordEnabled={wakeWordEnabled}
          onSetWakeWord={setWakeWord}
          onSetWakeWordEnabled={setWakeWordEnabled}
          systemPrompt={systemPrompt}
          isDefaultPrompt={isDefaultPrompt}
          onSetSystemPrompt={handleSetSystemPrompt}
          onResetSystemPrompt={handleResetSystemPrompt}
          chutesConfigured={chutesConfigured}
        />
      </div>
    )
  }

  // ============================================
  // CHAT MODE - Clean and minimal
  // ============================================
  return (
    <FileUploadZone onFilesAdded={addFiles} disabled={isLoading}>
      <div className="flex flex-col h-full">
        {/* Compact Header */}
        <header className="flex items-center justify-between px-4 py-3 border-b border-border/20 bg-background/80 backdrop-blur-xl">
          <div className="flex items-center gap-3">
            <div className="relative">
              <img src="/jarvis.jpeg" alt={assistantName} className="w-8 h-8 rounded-xl" />
              <div className={cn(
                'absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full border-2 border-background',
                connected ? 'bg-success' : 'bg-error'
              )} />
            </div>
            <div className="flex flex-col">
              <h1 className="text-sm font-semibold leading-none">{assistantName}</h1>
              {project && (
                <span className="text-xs text-text-muted">{project}</span>
              )}
            </div>
          </div>

          <div className="flex items-center gap-1">
            {/* Reasoning Level - Compact */}
            <div className="flex items-center bg-surface/40 rounded-lg p-0.5 border border-border/10">
              {[
                { level: 'fast' as const, icon: Zap, color: 'warning', title: 'Fast' },
                { level: null, icon: Scale, color: 'primary', title: 'Auto' },
                { level: 'deep' as const, icon: Brain, color: 'listening', title: 'Deep' },
              ].map(({ level, icon: Icon, color, title }) => (
                <button
                  key={title}
                  onClick={() => setReasoningLevel(level === reasoningLevel ? null : level)}
                  className={cn(
                    'p-1.5 rounded-md transition-all',
                    reasoningLevel === level
                      ? `bg-${color}/20 text-${color}`
                      : 'text-text-muted/60 hover:text-text-muted'
                  )}
                  title={title}
                >
                  <Icon size={14} />
                </button>
              ))}
            </div>

            {/* Voice Mode Button */}
            <button
              onClick={enterVoiceMode}
              className="p-2 rounded-lg bg-surface/40 border border-border/10 text-text-muted hover:text-purple-400 hover:border-purple-400/30 transition-all"
              title="Voice mode"
            >
              <AudioWaveform size={16} />
            </button>

            {/* Voice Input Button - Push to Talk */}
            <button
              onClick={handleVoiceToggle}
              className={cn(
                'p-2 rounded-lg transition-all border',
                isRecording
                  ? 'bg-error/20 border-error/30 text-error animate-pulse'
                  : 'bg-surface/40 border-border/10 text-text-muted hover:text-primary hover:border-primary/30'
              )}
              title={isRecording ? 'Stop recording' : 'Voice input (push to talk)'}
            >
              <Mic size={16} />
            </button>

            {/* Voice Output */}
            <button
              onClick={() => { setVoiceOutput(!voiceOutput); if (voiceOutput) stopSpeaking() }}
              className={cn(
                'p-2 rounded-lg transition-all border',
                voiceOutput
                  ? 'bg-success/10 border-success/30 text-success'
                  : 'bg-surface/40 border-border/10 text-text-muted'
              )}
              title={voiceOutput ? 'Voice output on' : 'Voice output off'}
            >
              {voiceOutput ? <Volume2 size={16} /> : <VolumeX size={16} />}
            </button>

            {/* Clear */}
            <button
              onClick={clear}
              className="p-2 rounded-lg bg-surface/40 text-text-muted hover:text-error border border-border/10 transition-all"
              title="Clear"
            >
              <Trash2 size={16} />
            </button>

            {/* Settings */}
            <button
              onClick={() => setShowSettings(true)}
              className="p-2 rounded-lg bg-surface/40 text-text-muted hover:text-text border border-border/10 transition-colors"
              title="Settings"
            >
              <Settings size={16} />
            </button>
          </div>
        </header>

        {/* Context Bar - Only when significant */}
        {contextStats && contextStats.percentage > 30 && (
          <div className="px-4 py-1.5 border-b border-border/10 bg-surface/20">
            <div className="flex items-center gap-2 max-w-3xl mx-auto">
              <div className="flex-1 h-1 bg-surface-2 rounded-full overflow-hidden">
                <div
                  className={cn(
                    'h-full rounded-full transition-all duration-500',
                    contextStats.percentage > 80 ? 'bg-error' :
                    contextStats.percentage > 60 ? 'bg-warning' : 'bg-primary/60'
                  )}
                  style={{ width: `${Math.min(contextStats.percentage, 100)}%` }}
                />
              </div>
              <span className={cn(
                'text-[10px] tabular-nums',
                contextStats.percentage > 80 ? 'text-error' :
                contextStats.percentage > 60 ? 'text-warning' : 'text-text-muted/60'
              )}>
                {contextStats.percentage.toFixed(0)}%
              </span>
            </div>
          </div>
        )}

        {/* Intent Badge - Floating */}
        {intentInfo?.detected && intentInfo.intent && (
          <div className="absolute top-16 right-4 z-10">
            <div className="text-[10px] px-2 py-1 bg-surface/80 backdrop-blur-sm rounded-lg border border-border/20 text-text-muted flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-primary" />
              {intentInfo.intent}
            </div>
          </div>
        )}

        {/* Messages */}
        <MessageList
          messages={messages}
          streaming={streaming}
          streamingThinking={streamingThinking}
          isLoading={isLoading}
          loadingText={loadingText}
        />

        {/* Recording indicator */}
        {isRecording && (
          <div className="absolute bottom-24 left-1/2 -translate-x-1/2 z-10">
            <div className="flex items-center gap-2 px-4 py-2 bg-error/20 backdrop-blur-sm rounded-full border border-error/30 text-error">
              <span className="w-2 h-2 rounded-full bg-error animate-pulse" />
              <span className="text-sm font-medium">Listening...</span>
            </div>
          </div>
        )}

        {/* Input */}
        <UnifiedInput
          value={input}
          onChange={setInput}
          onSend={handleSend}
          onVoiceToggle={handleVoiceToggle}
          onFilesAdded={addFiles}
          files={files}
          onRemoveFile={removeFile}
          isRecording={isRecording}
          isLoading={isLoading}
          disabled={!connected}
        />

        {/* Settings Panel */}
        <SettingsPanel
          isOpen={showSettings}
          onClose={() => setShowSettings(false)}
          currentModel={model}
          currentProvider={provider}
          providers={providers}
          models={models}
          onSwitchModel={switchModel}
          onSwitchProvider={switchProvider}
          voiceSettings={{
            tts_provider: ttsProvider,
            tts_voice: currentVoice,
            stt_provider: sttProvider,
          }}
          onSetTTSProvider={handleSetTTSProvider}
          onSetSTTProvider={handleSetSTTProvider}
          onSetVoice={handleSetVoice}
          edgeVoices={voices}
          elevenVoices={elevenVoices}
          kokoroVoices={kokoroVoices}
          wakeWord={wakeWord}
          wakeWordEnabled={wakeWordEnabled}
          onSetWakeWord={setWakeWord}
          onSetWakeWordEnabled={setWakeWordEnabled}
          systemPrompt={systemPrompt}
          isDefaultPrompt={isDefaultPrompt}
          onSetSystemPrompt={handleSetSystemPrompt}
          onResetSystemPrompt={handleResetSystemPrompt}
          chutesConfigured={chutesConfigured}
        />
      </div>
    </FileUploadZone>
  )
}
