import { useState, useEffect, useCallback } from 'react'
import { BrowserRouter, Routes, Route, useNavigate, useParams } from 'react-router-dom'
import { useWebSocket } from './hooks/useWebSocket'
import { useVoice } from './hooks/useVoice'
import { useCamera } from './hooks/useCamera'
import { useFileUpload } from './hooks/useFileUpload'
import { useWakeWord } from './hooks/useWakeWord'
import { useTheme } from './hooks/useTheme'
import { cn, apiFetch } from './lib/utils'

// Auth
import { AuthProvider } from './contexts/AuthContext'
import { ProtectedRoute } from './components/auth/ProtectedRoute'
import { LoginPage } from './components/auth/LoginPage'

// Components
import { MessageList } from './components/chat'
import { UnifiedInput, FileUploadZone } from './components/input'
import { SettingsPanel } from './components/settings'
import { VoiceOverlay } from './components/voice/VoiceOverlay'
import { FloatingOrb } from './components/orb'
import { getOrbState } from './components/orb/getOrbState'
import { DraggableCamera } from './components/camera/DraggableCamera'
import { ChatSidebar } from './components/sidebar/ChatSidebar'

// Icons
import {
  Settings,
  Zap,
  Scale,
  Brain,
  AudioWaveform,
  Sun,
  Moon,
} from 'lucide-react'

// Types
import type { ReasoningLevel, TTSProvider, STTProvider } from './types'

// Loading text options
const LOADING_TEXTS = [
  'Thinking...', 'Processing...', 'Analyzing...', 'Computing...',
  'On it...', 'One moment...', 'Working on it...', 'Let me check...',
]

function AppLayout() {
  const navigate = useNavigate()
  const { chatId: urlChatId } = useParams<{ chatId?: string }>()

  // Theme
  const { theme, toggleTheme } = useTheme()

  // Assistant configuration (customizable name, etc.)
  const [assistantName, setAssistantName] = useState('Jarvis')

  // State
  const [mode, setMode] = useState<'chat' | 'voice'>('chat')
  const [reasoningLevel, setReasoningLevel] = useState<ReasoningLevel>(null)
  const [voiceOutput] = useState(false)
  const [input, setInput] = useState('')
  const [lastSpokenIndex, setLastSpokenIndex] = useState(-1)
  const [loadingText, setLoadingText] = useState('')
  const [wakeWordEnabled, setWakeWordEnabled] = useState(() => {
    return localStorage.getItem('jarvis_wake_word_enabled') === 'true'
  })
  const [wakeWord, setWakeWord] = useState(() => {
    return localStorage.getItem('jarvis_wake_word') || 'jarvis'
  })

  // Sidebar state
  const [sidebarOpen, setSidebarOpen] = useState(() => {
    const saved = localStorage.getItem('jarvis_sidebar_open')
    return saved !== null ? saved === 'true' : window.innerWidth >= 768
  })

  // Settings state
  const [showSettings, setShowSettings] = useState(false)
  const [models, setModels] = useState<string[]>([])
  const [providers, setProviders] = useState<Record<string, { configured: boolean; model?: string | null }>>({})
  const [voices, setVoices] = useState<{ id: string; name: string }[]>([])
  const [elevenVoices, setElevenVoices] = useState<{ id: string; name: string }[]>([])
  const [kokoroVoices, setKokoroVoices] = useState<{ id: string; name: string }[]>([])
  const [chutesConfigured, setChutesConfigured] = useState(false)
  const [ttsProvider, setTtsProvider] = useState<TTSProvider>('edge')
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
    assistantName: wsAssistantName,
    chatId,
    contextStats,
    liveToolStatus,
    send,
    sendWithVideo,
    switchModel,
    switchProvider,
    switchChat,
    createChat,
    stopGeneration,
    setOnAuthError,
  } = useWebSocket()

  // File upload hook
  const { files, addFiles, removeFile, clearFiles, getAttachmentIds } = useFileUpload()

  // Camera hook for video chat
  const {
    videoRef,
    isActive: isCameraActive,
    error: cameraError,
    startCamera,
    stopCamera,
    captureFrame,
  } = useCamera()

  // Voice callbacks
  const handleVoiceInput = useCallback((transcript: string) => {
    if (transcript && connected) {
      if (isCameraActive) {
        const frame = captureFrame()
        sendWithVideo(transcript, frame)
      } else {
        send(transcript, true)
      }
    }
  }, [connected, send, sendWithVideo, isCameraActive, captureFrame])

  const handleInterrupt = useCallback(() => {
    console.log('User interrupted')
  }, [])

  // Voice hook
  const {
    isListening,
    isRecording,
    isPlaying,
    volume,
    playbackVolume,
    interimTranscript,
    startListening,
    stopListening,
    startRecording,
    stopRecording,
    speak,
    stopSpeaking,
    getFrequencyData,
    getPlaybackFrequencyData,
    interruptAndListen,
  } = useVoice({
    onSpeechEnd: handleVoiceInput,
    onInterrupt: handleInterrupt,
    sttProvider,
  })

  // Auth error handler — redirect to login
  useEffect(() => {
    setOnAuthError(() => {
      navigate('/login')
    })
  }, [setOnAuthError, navigate])

  // Sync chat ID from URL on initial load
  useEffect(() => {
    if (urlChatId && connected && urlChatId !== chatId) {
      switchChat(urlChatId)
    }
  }, [urlChatId, connected]) // eslint-disable-line react-hooks/exhaustive-deps -- only on mount/connect

  // Update URL when chatId changes
  useEffect(() => {
    if (chatId) {
      const currentPath = window.location.pathname
      const expectedPath = `/chat/${chatId}`
      if (currentPath !== expectedPath) {
        navigate(expectedPath, { replace: true })
      }
    }
  }, [chatId, navigate])

  // Sync assistant name from WebSocket connection
  useEffect(() => {
    if (wsAssistantName) setAssistantName(wsAssistantName) // eslint-disable-line react-hooks/set-state-in-effect -- syncing external WS data
  }, [wsAssistantName])

  // Voice mode handlers
  const enterVoiceMode = useCallback(() => {
    setMode('voice')
    stopSpeaking()
    startListening()
  }, [stopSpeaking, startListening])

  const exitVoiceMode = useCallback(() => {
    stopListening()
    stopSpeaking()
    stopCamera()
    setMode('chat')
  }, [stopListening, stopSpeaking, stopCamera])

  // Wake word detection
  // Stop wake word when voice mode is active (browsers allow only one SpeechRecognition)
  const wakeWordActive = wakeWordEnabled && mode !== 'voice' && !isListening && !isRecording
  useWakeWord({
    keyword: wakeWord,
    onWakeWord: enterVoiceMode,
    enabled: wakeWordActive,
  })

  // Persist wake word settings to both localStorage (fast) and backend (durable)
  const handleSetWakeWordEnabled = useCallback(async (enabled: boolean) => {
    setWakeWordEnabled(enabled)
    localStorage.setItem('jarvis_wake_word_enabled', String(enabled))
    await apiFetch('/api/settings/voice', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ wake_word_enabled: enabled }),
    }).catch(() => { /* ignore network errors */ })
  }, [])

  const handleSetWakeWord = useCallback(async (word: string) => {
    setWakeWord(word)
    localStorage.setItem('jarvis_wake_word', word)
    await apiFetch('/api/settings/voice', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ wake_word: word }),
    }).catch(() => { /* ignore network errors */ })
  }, [])

  // Randomize loading text
  useEffect(() => {
    if (isLoading) {
      setLoadingText(LOADING_TEXTS[Math.floor(Math.random() * LOADING_TEXTS.length)]) // eslint-disable-line react-hooks/set-state-in-effect -- derived state from prop
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
          fetch('/api/system-instructions').catch(() => null),
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
          if (data.wake_word) {
            setWakeWord(data.wake_word)
            localStorage.setItem('jarvis_wake_word', data.wake_word)
          }
          if (data.wake_word_enabled !== undefined) {
            setWakeWordEnabled(data.wake_word_enabled)
            localStorage.setItem('jarvis_wake_word_enabled', String(data.wake_word_enabled))
          }
        }
        if (promptRes?.ok) {
          const data = await promptRes.json()
          setSystemPromptState(data.content || '')
          setIsDefaultPrompt(data.isEmpty ?? true)
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
        setLastSpokenIndex(lastIndex) // eslint-disable-line react-hooks/set-state-in-effect -- tracking spoken index
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

  // Handlers
  const handleSend = useCallback(() => {
    const attachments = getAttachmentIds()
    if ((!input.trim() && attachments.length === 0) || !connected) return
    send(input.trim(), true, reasoningLevel, attachments)
    setInput('')
    clearFiles()
  }, [input, connected, send, reasoningLevel, clearFiles, getAttachmentIds])

  const handleQuickAction = useCallback((prompt: string) => {
    setInput(prompt)
    // If it's a complete prompt (no trailing space), send immediately
    if (!prompt.endsWith(' ') && connected) {
      send(prompt, true)
      setInput('')
    }
  }, [connected, send])

  const handleCameraToggle = useCallback(() => {
    if (isCameraActive) {
      stopCamera()
    } else {
      startCamera()
    }
  }, [isCameraActive, startCamera, stopCamera])

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

  const handleVoiceMicToggle = useCallback(() => {
    if (isListening) {
      stopListening()
    } else {
      startListening()
    }
  }, [isListening, stopListening, startListening])

  // Settings handlers
  const handleSetTTSProvider = useCallback(async (p: TTSProvider) => {
    setTtsProvider(p)
    await apiFetch('/api/settings/voice', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tts_provider: p }),
    })
  }, [])

  const handleSetSTTProvider = useCallback(async (p: STTProvider) => {
    setSttProvider(p)
    await apiFetch('/api/settings/voice', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stt_provider: p }),
    })
  }, [])

  const handleSetVoice = useCallback(async (voiceId: string, p: 'edge' | 'elevenlabs' | 'kokoro') => {
    setCurrentVoice(voiceId)
    if (p === 'elevenlabs') {
      await apiFetch('/api/settings/elevenlabs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ voice_id: voiceId }),
      })
    } else if (p === 'kokoro') {
      await apiFetch('/api/settings/voice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kokoro_voice: voiceId }),
      })
    } else {
      await apiFetch('/api/settings/voice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ voice: voiceId }),
      })
    }
  }, [])

  const handleSetSystemPrompt = useCallback(async (prompt: string) => {
    setSystemPromptState(prompt)
    setIsDefaultPrompt(!prompt.trim())
    await apiFetch('/api/system-instructions', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: prompt }),
    })
  }, [])

  const handleResetSystemPrompt = useCallback(async () => {
    const res = await apiFetch('/api/system-instructions', { method: 'DELETE' })
    if (res.ok) {
      setSystemPromptState('')
      setIsDefaultPrompt(true)
    }
  }, [])

  // Smart defaults: Auto-configure TTS when switching to Chutes
  useEffect(() => {
    if (provider === 'chutes' && chutesConfigured) {
      if (ttsProvider !== 'kokoro') {
        handleSetTTSProvider('kokoro') // eslint-disable-line react-hooks/set-state-in-effect -- auto-config on provider change
      }
    }
  }, [provider, chutesConfigured, ttsProvider, handleSetTTSProvider])

  // New chat handler - also navigates to root
  const handleNewChat = useCallback(() => {
    createChat()
    navigate('/', { replace: true })
    // Close sidebar on mobile
    if (window.innerWidth < 768) setSidebarOpen(false)
  }, [createChat, navigate])

  // Select chat handler - closes sidebar on mobile
  const handleSelectChat = useCallback((id: string) => {
    switchChat(id)
    if (window.innerWidth < 768) setSidebarOpen(false)
  }, [switchChat])

  // Sidebar toggle with persistence
  const handleSidebarToggle = useCallback(() => {
    setSidebarOpen(prev => {
      const next = !prev
      localStorage.setItem('jarvis_sidebar_open', String(next))
      return next
    })
  }, [])

  // Get last assistant message for voice overlay
  const lastAssistantMessage = messages.length > 0 && messages[messages.length - 1].role === 'assistant'
    ? messages[messages.length - 1].content
    : undefined

  return (
    <FileUploadZone onFilesAdded={addFiles} disabled={isLoading}>
      <div className="flex h-full overflow-hidden">
        {/* Chat Sidebar */}
        <ChatSidebar
          activeChatId={chatId}
          onSelectChat={handleSelectChat}
          onNewChat={handleNewChat}
          isOpen={sidebarOpen}
          onToggle={handleSidebarToggle}
        />

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col h-full min-w-0 overflow-hidden">
          {/* ============================================ */}
          {/* Compact Header */}
          {/* ============================================ */}
          <header className="flex items-center justify-between px-2 sm:px-4 py-2.5 border-b border-border/20 bg-background/80 backdrop-blur-xl">
            {/* Left: Avatar + name + model + wake word indicator */}
            <div className="flex items-center gap-2 sm:gap-3 min-w-0">
              {!sidebarOpen && <div className="w-8" />} {/* Spacer for sidebar toggle button */}
              <div className="relative">
                <img src="/jarvis.jpeg" alt={assistantName} className="w-7 h-7 rounded-lg" />
                <div className={cn(
                  'absolute -bottom-0.5 -right-0.5 w-2 h-2 rounded-full border-[1.5px] border-background',
                  connected ? 'bg-success' : 'bg-error'
                )} />
              </div>
              <div className="flex flex-col">
                <div className="flex items-center gap-2">
                  <h1 className="text-sm font-semibold leading-none">{assistantName}</h1>
                  {wakeWordEnabled && (
                    <div className="flex items-center gap-1" title="Wake word active">
                      <span className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-pulse" />
                    </div>
                  )}
                </div>
                <span className="text-[11px] text-text-muted leading-tight mt-0.5">
                  {model?.split('/').pop() || project || 'AI'}
                </span>
              </div>
            </div>

            {/* Right: Reasoning + Voice mode + Settings */}
            <div className="flex items-center gap-0.5 sm:gap-1 flex-shrink-0">
              {/* Reasoning Level */}
              <div className="hidden sm:flex items-center bg-surface/40 rounded-lg p-0.5 border border-border/10">
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

              {/* Theme Toggle */}
              <button
                onClick={toggleTheme}
                className="p-2 rounded-lg bg-surface/40 text-text-muted hover:text-text border border-border/10 transition-colors"
                title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              >
                {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
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

          {/* Messages */}
          <MessageList
            messages={messages}
            streaming={streaming}
            streamingThinking={streamingThinking}
            isLoading={isLoading}
            loadingText={loadingText}
            liveToolStatus={liveToolStatus}
            assistantName={assistantName}
            onQuickAction={handleQuickAction}
          />

          {/* Recording indicator */}
          {isRecording && mode === 'chat' && (
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
            onStop={stopGeneration}
            onVoiceToggle={handleVoiceToggle}
            onFilesAdded={addFiles}
            files={files}
            onRemoveFile={removeFile}
            isRecording={isRecording}
            isLoading={isLoading}
            isStreaming={!!streaming}
            disabled={!connected}
          />

          {/* ============================================ */}
          {/* Floating Orb - mobile FAB for voice mode */}
          {/* ============================================ */}
          {mode === 'chat' && (
            <FloatingOrb
              state={getOrbState({ isPlaying, isLoading, isListening, isRecording })}
              volume={volume}
              isStreaming={!!streaming}
              onClick={enterVoiceMode}
            />
          )}

          {/* ============================================ */}
          {/* Voice Overlay - sits on top of everything */}
          {/* ============================================ */}
          <VoiceOverlay
            isOpen={mode === 'voice'}
            connected={connected}
            model={model}
            isListening={isListening}
            isRecording={isRecording}
            isPlaying={isPlaying}
            isLoading={isLoading}
            volume={volume}
            playbackVolume={playbackVolume}
            interimTranscript={interimTranscript}
            lastMessage={lastAssistantMessage}
            streaming={streaming}
            loadingText={loadingText}
            isCameraActive={isCameraActive}
            videoRef={videoRef}
            cameraError={cameraError}
            getFrequencyData={getFrequencyData}
            getPlaybackFrequencyData={getPlaybackFrequencyData}
            onMicToggle={handleVoiceMicToggle}
            onStopSpeaking={stopSpeaking}
            onStopGeneration={stopGeneration}
            onInterruptAndListen={interruptAndListen}
            onCameraToggle={handleCameraToggle}
            onOpenSettings={() => setShowSettings(true)}
            onClose={exitVoiceMode}
          />

          {/* ============================================ */}
          {/* Draggable Camera - floating in chat mode */}
          {/* ============================================ */}
          <DraggableCamera
            isActive={isCameraActive && mode === 'chat'}
            videoRef={videoRef}
            onClose={stopCamera}
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
            onSetWakeWord={handleSetWakeWord}
            onSetWakeWordEnabled={handleSetWakeWordEnabled}
            systemPrompt={systemPrompt}
            isDefaultPrompt={isDefaultPrompt}
            onSetSystemPrompt={handleSetSystemPrompt}
            onResetSystemPrompt={handleResetSystemPrompt}
            chutesConfigured={chutesConfigured}
          />
        </div>
      </div>
    </FileUploadZone>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/chat/:chatId" element={
            <ProtectedRoute>
              <AppLayout />
            </ProtectedRoute>
          } />
          <Route path="/*" element={
            <ProtectedRoute>
              <AppLayout />
            </ProtectedRoute>
          } />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  )
}
