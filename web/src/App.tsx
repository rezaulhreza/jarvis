import { useState, useRef, useEffect, useCallback } from 'react'
import { useWebSocket } from './hooks/useWebSocket'
import { useVoice } from './hooks/useVoice'
import { cn } from './lib/utils'
import {
  Mic,
  MicOff,
  Send,
  Volume2,
  VolumeX,
  MessageSquare,
  Phone,
  Trash2,
  Loader2,
  ChevronDown,
  Settings,
} from 'lucide-react'

type Mode = 'chat' | 'voice'

export default function App() {
  const [mode, setMode] = useState<Mode>('chat')
  const [chatMode, setChatMode] = useState(true)
  const [voiceOutput, setVoiceOutput] = useState(false)
  const [input, setInput] = useState('')
  const [lastSpokenIndex, setLastSpokenIndex] = useState(-1)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const [models, setModels] = useState<string[]>([])
  const [voices, setVoices] = useState<{id: string, name: string}[]>([])
  const [elevenVoices, setElevenVoices] = useState<{id: string, name: string}[]>([])
  const [currentVoice, setCurrentVoice] = useState('en-GB-SoniaNeural')
  const [showSettings, setShowSettings] = useState(false)
  const [settingsTab, setSettingsTab] = useState<'model' | 'voice' | 'stt'>('model')
  const [isMuted, setIsMuted] = useState(false)
  const [loadingText, setLoadingText] = useState('')
  const [ttsProvider, setTtsProvider] = useState<'browser' | 'edge' | 'elevenlabs'>('browser')
  const [sttProvider, setSttProvider] = useState<'browser' | 'whisper'>('browser')
  const [elevenLabsKey, setElevenLabsKey] = useState('')
  const [showApiKeyInput, setShowApiKeyInput] = useState(false)

  const {
    connected,
    messages,
    streaming,
    isLoading,
    project,
    model,
    send,
    clear,
    switchModel,
  } = useWebSocket()

  // Randomize loading text when loading starts
  useEffect(() => {
    if (isLoading) {
      const texts = [
        'Crunching...', 'Cooking...', 'Brewing...', 'Conjuring...', 'Summoning...',
        'Accomplishing...', 'Manifesting...', 'Crafting...', 'Whipping up...',
        'On it...', 'One sec...', 'Hmm...', 'Let me see...', 'Working on it...',
        'Hold tight...', 'Spinning up...', 'Pondering...', 'Assembling...'
      ]
      setLoadingText(texts[Math.floor(Math.random() * texts.length)])
    }
  }, [isLoading])

  // Fetch available models, voices, and settings
  useEffect(() => {
    fetch('/api/models')
      .then(res => res.json())
      .then(data => setModels(data.models || []))
      .catch(() => {})
    fetch('/api/voices')
      .then(res => res.json())
      .then(data => setVoices(data.voices || []))
      .catch(() => {})
    // Fetch ElevenLabs voices if configured
    fetch('/api/elevenlabs/voices')
      .then(res => res.json())
      .then(data => {
        if (data.voices?.length > 0) {
          setElevenVoices(data.voices)
        }
      })
      .catch(() => {})
    // Load saved voice settings
    fetch('/api/settings/voice')
      .then(res => res.json())
      .then(data => {
        if (data.tts_provider) setTtsProvider(data.tts_provider)
        if (data.tts_voice) setCurrentVoice(data.tts_voice)
        if (data.stt_provider) setSttProvider(data.stt_provider)
      })
      .catch(() => {})
  }, [])

  // Save TTS provider when changed
  const changeTtsProvider = (provider: 'browser' | 'edge' | 'elevenlabs') => {
    setTtsProvider(provider)
    fetch('/api/settings/voice', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tts_provider: provider })
    }).catch(() => {})
  }

  // Save STT provider when changed
  const changeSttProvider = (provider: 'browser' | 'whisper') => {
    setSttProvider(provider)
    fetch('/api/settings/voice', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stt_provider: provider })
    }).catch(() => {})
  }

  const switchVoice = async (voiceId: string, provider: 'edge' | 'elevenlabs' = 'edge') => {
    setCurrentVoice(voiceId)
    if (provider === 'elevenlabs') {
      await fetch('/api/settings/elevenlabs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ voice_id: voiceId })
      }).catch(() => {})
    } else {
      await fetch('/api/settings/voice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ voice: voiceId })
      }).catch(() => {})
    }
  }

  const saveElevenLabsKey = async () => {
    if (!elevenLabsKey) return
    await fetch('/api/settings/elevenlabs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: elevenLabsKey })
    })
    // Refresh voices
    const res = await fetch('/api/elevenlabs/voices')
    const data = await res.json()
    if (data.voices?.length > 0) {
      setElevenVoices(data.voices)
      setShowApiKeyInput(false)
    }
  }

  // Callback for when voice input is detected and processed
  const handleVoiceInput = useCallback((transcript: string) => {
    if (transcript && connected) {
      send(transcript, true) // Always chat mode for voice
    }
  }, [connected, send])

  // Handle interruption - user started speaking while Jarvis was talking
  const handleInterrupt = useCallback(() => {
    console.log('User interrupted')
    // Could cancel pending requests here if needed
  }, [])

  const {
    isListening,
    isRecording,
    isPlaying,
    volume,
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

  // Close settings dropdown when clicking outside
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (showSettings && !(e.target as Element).closest('.relative')) {
        setShowSettings(false)
      }
    }
    document.addEventListener('click', handleClick)
    return () => document.removeEventListener('click', handleClick)
  }, [showSettings])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streaming])

  // Auto-speak in voice mode OR when voice output is on
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

  // Auto-resume listening after Jarvis stops speaking (voice mode only, if not muted)
  useEffect(() => {
    if (mode === 'voice' && !isPlaying && !isLoading && connected && !isListening && !isMuted) {
      // Small delay before resuming listening
      const timer = setTimeout(() => {
        if (mode === 'voice' && !isPlaying && !isLoading && !isMuted) {
          startListening()
        }
      }, 500)
      return () => clearTimeout(timer)
    }
  }, [mode, isPlaying, isLoading, connected, isListening, isMuted, startListening])

  // Stop listening when leaving voice mode
  useEffect(() => {
    if (mode !== 'voice' && isListening) {
      stopListening()
    }
  }, [mode, isListening, stopListening])

  const handleSend = () => {
    if (!input.trim() || !connected) return
    send(input.trim(), chatMode)
    setInput('')
  }

  const handleVoiceToggle = async () => {
    if (isRecording) {
      const transcript = await stopRecording()
      if (transcript) {
        send(transcript, true)
      }
    } else {
      stopSpeaking()
      startRecording()
    }
  }

  const enterVoiceMode = () => {
    setMode('voice')
    stopSpeaking()
    startListening()
  }

  const exitVoiceMode = () => {
    stopListening()
    stopSpeaking()
    setMode('chat')
  }

  // Voice mode - immersive always-listening experience
  if (mode === 'voice') {
    return (
      <div className="flex flex-col h-full bg-gradient-to-b from-[#0a0a0f] to-[#12121a]">
        {/* Minimal header */}
        <header className="flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <img src="/jarvis.jpeg" alt="Jarvis" className="w-10 h-10 rounded-full" />
            <span className="text-lg font-medium">Jarvis</span>
          </div>
          <button
            onClick={exitVoiceMode}
            className="p-2 rounded-lg hover:bg-[#1a1a24] text-[#71717a]"
          >
            <MessageSquare size={20} />
          </button>
        </header>

        {/* Center - Voice interface */}
        <div className="flex-1 flex flex-col items-center justify-center gap-6">
          {/* Animated orb with volume indicator */}
          <div className={cn(
            'relative w-48 h-48 rounded-full flex items-center justify-center transition-all duration-300',
            isRecording && volume > 0.02 ? 'scale-110' : '',
            isPlaying ? 'bg-green-500/20' :
            isLoading ? 'bg-blue-500/20' :
            isListening ? 'bg-purple-500/10' :
            'bg-[#1a1a24]'
          )}>
            {/* Volume ring */}
            {isListening && (
              <div
                className="absolute inset-0 rounded-full border-2 border-purple-500/50 transition-transform duration-75"
                style={{ transform: `scale(${1 + volume * 2})`, opacity: volume > 0.01 ? 1 : 0.3 }}
              />
            )}

            <div className={cn(
              'w-36 h-36 rounded-full flex items-center justify-center transition-all duration-300',
              isPlaying ? 'bg-green-500/30 animate-pulse' :
              isLoading ? 'bg-blue-500/30 animate-pulse' :
              isRecording && volume > 0.02 ? 'bg-purple-500/30' :
              'bg-[#22222a]'
            )}>
              <img
                src="/jarvis.jpeg"
                alt="Jarvis"
                className={cn(
                  'w-24 h-24 rounded-full object-cover transition-all duration-300',
                  isPlaying && 'scale-105'
                )}
              />
            </div>
          </div>

          {/* Status text */}
          <div className="text-center h-8">
            {isLoading && <p className="text-blue-400">{loadingText}</p>}
            {isPlaying && <p className="text-green-400">Speaking...</p>}
            {isListening && !isLoading && !isPlaying && (
              <p className={cn(
                'transition-colors',
                volume > 0.02 ? 'text-purple-400' : 'text-[#71717a]'
              )}>
                {volume > 0.02 ? 'Listening...' : 'Speak anytime'}
              </p>
            )}
            {!isListening && !isLoading && !isPlaying && !connected && (
              <p className="text-red-400">Disconnected</p>
            )}
          </div>

          {/* Control buttons */}
          <div className="flex gap-4">
            {/* Mute button - stops listening so you can type/use keyboard */}
            <button
              onClick={() => {
                setIsMuted(!isMuted)
                if (!isMuted) stopListening()
              }}
              className={cn(
                'w-12 h-12 rounded-full flex items-center justify-center transition-all duration-200',
                isMuted
                  ? 'bg-red-500/20 text-red-400'
                  : 'bg-[#2a2a3a] text-[#71717a] hover:bg-[#3a3a4a]'
              )}
              title={isMuted ? 'Unmute' : 'Mute (use keyboard)'}
            >
              {isMuted ? <VolumeX size={20} /> : <Volume2 size={20} />}
            </button>

            {/* Mic toggle */}
            <button
              onClick={() => isListening ? stopListening() : startListening()}
              disabled={isLoading || !connected || isMuted}
              className={cn(
                'w-16 h-16 rounded-full flex items-center justify-center transition-all duration-200',
                isListening
                  ? 'bg-purple-500 shadow-lg shadow-purple-500/30'
                  : 'bg-[#2a2a3a] hover:bg-[#3a3a4a]',
                (isLoading || !connected || isMuted) && 'opacity-50 cursor-not-allowed'
              )}
            >
              {isListening ? <Mic size={24} className="text-white" /> : <MicOff size={24} className="text-[#71717a]" />}
            </button>
          </div>
        </div>

        {/* Last message preview */}
        {messages.length > 0 && !isLoading && (
          <div className="px-6 py-4 text-center">
            <p className="text-[#71717a] text-sm max-w-md mx-auto line-clamp-2">
              {messages[messages.length - 1].content}
            </p>
          </div>
        )}
      </div>
    )
  }

  // Chat mode
  return (
    <div className="flex flex-col h-full">
      <header className="flex items-center justify-between px-6 py-4 border-b border-[#2a2a3a]">
        <div className="flex items-center gap-3">
          <img src="/jarvis.jpeg" alt="Jarvis" className="w-8 h-8 rounded-full" />
          <h1 className="text-xl font-semibold">Jarvis</h1>
          {project && (
            <span className="text-sm px-2 py-1 rounded bg-[#1a1a24] text-[#71717a]">{project}</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {/* Model selector */}
          <div className="relative">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm bg-[#1a1a24] text-[#71717a] hover:text-white"
            >
              <Settings size={14} />
              <span className="max-w-[100px] truncate">{model || 'Model'}</span>
              <ChevronDown size={14} />
            </button>
            {showSettings && (
              <div className="absolute right-0 top-full mt-1 w-72 bg-[#1a1a24] border border-[#2a2a3a] rounded-lg shadow-xl z-50 max-h-96 overflow-hidden">
                {/* Tabs */}
                <div className="flex border-b border-[#2a2a3a]">
                  <button
                    onClick={() => setSettingsTab('model')}
                    className={cn(
                      'flex-1 px-3 py-2 text-xs',
                      settingsTab === 'model' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-[#71717a]'
                    )}
                  >
                    Model
                  </button>
                  <button
                    onClick={() => setSettingsTab('voice')}
                    className={cn(
                      'flex-1 px-3 py-2 text-xs',
                      settingsTab === 'voice' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-[#71717a]'
                    )}
                  >
                    TTS
                  </button>
                  <button
                    onClick={() => setSettingsTab('stt')}
                    className={cn(
                      'flex-1 px-3 py-2 text-xs',
                      settingsTab === 'stt' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-[#71717a]'
                    )}
                  >
                    STT
                  </button>
                </div>
                {/* Content */}
                <div className="max-h-72 overflow-y-auto">
                  {settingsTab === 'model' && (
                    <>
                      {models.map((m) => (
                        <button
                          key={m}
                          onClick={() => { switchModel(m); setShowSettings(false) }}
                          className={cn(
                            'w-full px-3 py-2 text-left text-sm hover:bg-[#2a2a3a] transition-colors',
                            m === model ? 'text-blue-400' : 'text-white'
                          )}
                        >
                          {m}
                        </button>
                      ))}
                      {models.length === 0 && (
                        <div className="px-3 py-2 text-sm text-[#71717a]">No models found</div>
                      )}
                    </>
                  )}
                  {settingsTab === 'voice' && (
                    <>
                      {/* TTS Provider Selection */}
                      <div className="px-3 py-2 border-b border-[#2a2a3a]">
                        <span className="text-xs text-[#71717a] block mb-2">TTS Provider</span>
                        <div className="flex gap-1">
                          {(['browser', 'edge', 'elevenlabs'] as const).map((p) => (
                            <button
                              key={p}
                              onClick={() => changeTtsProvider(p)}
                              className={cn(
                                'px-2 py-1 text-xs rounded transition-colors flex-1',
                                ttsProvider === p
                                  ? p === 'elevenlabs' ? 'bg-purple-500/30 text-purple-300'
                                    : p === 'edge' ? 'bg-blue-500/30 text-blue-300'
                                    : 'bg-green-500/30 text-green-300'
                                  : 'bg-[#2a2a3a] text-[#71717a] hover:text-white'
                              )}
                            >
                              {p === 'browser' ? 'Fast' : p === 'edge' ? 'Edge' : '11Labs'}
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* Browser TTS info */}
                      {ttsProvider === 'browser' && (
                        <div className="px-3 py-2 text-xs text-[#51515a]">
                          Instant response using browser's built-in voice.
                        </div>
                      )}

                      {/* Edge TTS voices */}
                      {ttsProvider === 'edge' && (
                        <>
                          <div className="px-3 py-1 text-xs text-[#51515a]">Neural voices (free)</div>
                          {voices.map((v) => (
                            <button
                              key={v.id}
                              onClick={() => { switchVoice(v.id, 'edge'); setShowSettings(false) }}
                              className={cn(
                                'w-full px-3 py-2 text-left text-sm hover:bg-[#2a2a3a] transition-colors',
                                v.id === currentVoice ? 'text-blue-400' : 'text-white'
                              )}
                            >
                              {v.name}
                            </button>
                          ))}
                        </>
                      )}

                      {/* ElevenLabs */}
                      {ttsProvider === 'elevenlabs' && (
                        <>
                          {elevenVoices.length === 0 ? (
                            <div className="px-3 py-2">
                              <p className="text-xs text-[#71717a] mb-2">Enter your ElevenLabs API key:</p>
                              <input
                                type="password"
                                value={elevenLabsKey}
                                onChange={(e) => setElevenLabsKey(e.target.value)}
                                placeholder="xi-xxxxxxxx..."
                                className="w-full px-2 py-1 text-sm bg-[#12121a] border border-[#3a3a4a] rounded mb-2"
                              />
                              <button
                                onClick={saveElevenLabsKey}
                                disabled={!elevenLabsKey}
                                className="w-full px-2 py-1 text-xs bg-purple-500/30 text-purple-300 rounded hover:bg-purple-500/50 disabled:opacity-50"
                              >
                                Save & Load Voices
                              </button>
                            </div>
                          ) : (
                            <>
                              <div className="px-3 py-1 text-xs text-[#51515a]">ElevenLabs voices (streaming)</div>
                              {elevenVoices.map((v) => (
                                <button
                                  key={v.id}
                                  onClick={() => { switchVoice(v.id, 'elevenlabs'); setShowSettings(false) }}
                                  className={cn(
                                    'w-full px-3 py-2 text-left text-sm hover:bg-[#2a2a3a] transition-colors',
                                    v.id === currentVoice ? 'text-purple-400' : 'text-white'
                                  )}
                                >
                                  {v.name}
                                </button>
                              ))}
                              <button
                                onClick={() => setElevenVoices([])}
                                className="w-full px-3 py-1 text-xs text-[#51515a] hover:text-red-400"
                              >
                                Change API Key
                              </button>
                            </>
                          )}
                        </>
                      )}
                    </>
                  )}
                  {settingsTab === 'stt' && (
                    <>
                      {/* STT Provider Selection */}
                      <div className="px-3 py-2 border-b border-[#2a2a3a]">
                        <span className="text-xs text-[#71717a] block mb-2">Speech-to-Text</span>
                        <div className="flex gap-1">
                          <button
                            onClick={() => changeSttProvider('browser')}
                            className={cn(
                              'px-3 py-1.5 text-xs rounded transition-colors flex-1',
                              sttProvider === 'browser'
                                ? 'bg-green-500/30 text-green-300'
                                : 'bg-[#2a2a3a] text-[#71717a] hover:text-white'
                            )}
                          >
                            Browser (instant)
                          </button>
                          <button
                            onClick={() => changeSttProvider('whisper')}
                            className={cn(
                              'px-3 py-1.5 text-xs rounded transition-colors flex-1',
                              sttProvider === 'whisper'
                                ? 'bg-purple-500/30 text-purple-300'
                                : 'bg-[#2a2a3a] text-[#71717a] hover:text-white'
                            )}
                          >
                            Whisper (accurate)
                          </button>
                        </div>
                      </div>
                      <div className="px-3 py-2 text-xs text-[#51515a]">
                        {sttProvider === 'browser'
                          ? 'Real-time transcription using browser API. Instant but may be less accurate.'
                          : 'OpenAI Whisper for accurate transcription. Processes after you stop speaking.'}
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>

          <button
            onClick={() => setChatMode(!chatMode)}
            className={cn(
              'px-3 py-1.5 rounded-lg text-sm transition-colors',
              chatMode ? 'bg-blue-500 text-white' : 'bg-[#1a1a24] text-[#71717a]'
            )}
          >
            {chatMode ? 'Chat' : 'Agent'}
          </button>

          <button
            onClick={() => { setVoiceOutput(!voiceOutput); if (voiceOutput) stopSpeaking() }}
            className={cn(
              'p-2 rounded-lg transition-colors',
              voiceOutput ? 'bg-green-500 text-white' : 'bg-[#1a1a24] text-[#71717a]'
            )}
          >
            {voiceOutput ? <Volume2 size={18} /> : <VolumeX size={18} />}
          </button>

          <button
            onClick={enterVoiceMode}
            className="p-2 rounded-lg bg-[#1a1a24] text-[#71717a] hover:text-white hover:bg-purple-500"
            title="Voice Mode"
          >
            <Phone size={18} />
          </button>

          <button onClick={clear} className="p-2 rounded-lg bg-[#1a1a24] text-[#71717a] hover:text-red-500">
            <Trash2 size={18} />
          </button>

          <span className={cn('w-2 h-2 rounded-full ml-2', connected ? 'bg-green-500' : 'bg-red-500')} />
        </div>
      </header>

      <main className="flex-1 overflow-y-auto p-4">
        <div className="max-w-3xl mx-auto space-y-4">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={cn(
                'p-4 rounded-xl',
                msg.role === 'user' ? 'bg-[#1a1a24] ml-12' :
                msg.role === 'system' ? 'bg-transparent text-center text-[#71717a] text-sm' :
                'bg-[#12121a] mr-12'
              )}
            >
              {msg.role !== 'system' && (
                <div className="flex justify-between items-center mb-2">
                  <p className="text-xs text-[#71717a]">{msg.role === 'user' ? 'You' : 'Jarvis'}</p>
                  <p className="text-xs text-[#51515a]">
                    {msg.timestamp?.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </p>
                </div>
              )}
              <p className="whitespace-pre-wrap">{msg.content}</p>
            </div>
          ))}

          {streaming && (
            <div className="p-4 rounded-xl bg-[#12121a] mr-12">
              <p className="text-xs text-[#71717a] mb-2">Jarvis</p>
              <p className="whitespace-pre-wrap">{streaming}</p>
            </div>
          )}

          {isLoading && !streaming && (
            <div className="flex items-center gap-2 text-[#71717a]">
              <Loader2 size={16} className="animate-spin" />
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>

      <footer className="p-4 border-t border-[#2a2a3a]">
        <div className="max-w-3xl mx-auto flex gap-2">
          <button
            onClick={handleVoiceToggle}
            disabled={isLoading}
            className={cn(
              'p-3 rounded-xl transition-colors',
              isRecording ? 'bg-red-500 text-white' : 'bg-[#1a1a24] text-[#71717a] hover:text-white'
            )}
          >
            {isRecording ? <MicOff size={20} /> : <Mic size={20} />}
          </button>

          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder={isRecording ? 'Listening...' : 'Type a message...'}
            disabled={isRecording}
            className="flex-1 px-4 py-3 rounded-xl bg-[#12121a] border border-[#2a2a3a] focus:border-blue-500 focus:outline-none"
          />

          <button
            onClick={handleSend}
            disabled={!input.trim() || !connected || isLoading}
            className="p-3 rounded-xl bg-blue-500 text-white disabled:opacity-50 hover:bg-blue-600"
          >
            <Send size={20} />
          </button>
        </div>
      </footer>
    </div>
  )
}
