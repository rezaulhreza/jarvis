import { useState, useEffect } from 'react'
import { cn } from '../../lib/utils'
import { SystemInstructions } from './SystemInstructions'
import { MemoryPanel } from './MemoryPanel'
import {
  X,
  Settings,
  Mic,
  Volume2,
  Brain,
  BookOpen,
  Check,
  Link2,
  MessageCircle,
  AlertCircle,
  Loader2,
} from 'lucide-react'
import type {
  VoiceSettings,
  TTSProvider,
  STTProvider,
  ProviderInfo,
} from '../../types'

type SettingsTab = 'general' | 'voice' | 'stt' | 'integrations' | 'system' | 'memory'

interface SettingsPanelProps {
  isOpen: boolean
  onClose: () => void
  // Model/Provider
  currentModel: string
  currentProvider: string
  providers: Record<string, ProviderInfo>
  models: string[]
  onSwitchModel: (model: string) => void
  onSwitchProvider: (provider: string) => void
  // Voice
  voiceSettings: VoiceSettings
  onSetTTSProvider: (provider: TTSProvider) => Promise<void>
  onSetSTTProvider: (provider: STTProvider) => Promise<void>
  onSetVoice: (voiceId: string, provider: 'edge' | 'elevenlabs' | 'kokoro') => Promise<void>
  edgeVoices: { id: string; name: string }[]
  elevenVoices: { id: string; name: string }[]
  kokoroVoices: { id: string; name: string }[]
  // Wake word
  wakeWord: string
  wakeWordEnabled: boolean
  onSetWakeWord: (word: string) => void
  onSetWakeWordEnabled: (enabled: boolean) => void
  // System prompt
  systemPrompt: string
  isDefaultPrompt: boolean
  onSetSystemPrompt: (prompt: string) => Promise<void>
  onResetSystemPrompt: () => Promise<void>
  // Chutes status
  chutesConfigured: boolean
}

export function SettingsPanel({
  isOpen,
  onClose,
  currentModel,
  currentProvider,
  providers,
  models,
  onSwitchModel,
  onSwitchProvider,
  voiceSettings,
  onSetTTSProvider,
  onSetSTTProvider,
  onSetVoice,
  edgeVoices,
  elevenVoices,
  kokoroVoices,
  wakeWord,
  wakeWordEnabled,
  onSetWakeWord,
  onSetWakeWordEnabled,
  systemPrompt,
  isDefaultPrompt,
  onSetSystemPrompt,
  onResetSystemPrompt,
  chutesConfigured,
}: SettingsPanelProps) {
  const [activeTab, setActiveTab] = useState<SettingsTab>('general')

  if (!isOpen) return null

  const tabs: { id: SettingsTab; label: string; icon: React.ReactNode }[] = [
    { id: 'general', label: 'Model', icon: <Brain size={18} /> },
    { id: 'voice', label: 'TTS', icon: <Volume2 size={18} /> },
    { id: 'stt', label: 'STT', icon: <Mic size={18} /> },
    { id: 'integrations', label: 'Connect', icon: <Link2 size={18} /> },
    { id: 'memory', label: 'Memory', icon: <BookOpen size={18} /> },
    { id: 'system', label: 'System', icon: <Settings size={18} /> },
  ]

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-background/60 backdrop-blur-sm z-40 animate-fade-in"
        onClick={onClose}
      />

      {/* Panel */}
      <div
        className={cn(
          'fixed right-0 top-0 bottom-0 w-full max-w-md z-50',
          'bg-surface border-l border-border/30',
          'flex flex-col',
          'animate-slide-in-right'
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border/30">
          <h2 className="text-lg font-semibold text-text">Settings</h2>
          <button
            onClick={onClose}
            className="p-2 rounded-lg text-text-muted hover:text-text hover:bg-surface-2 transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 p-2 border-b border-border/30 overflow-x-auto">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                'flex items-center gap-2 px-3 py-2 rounded-lg text-sm whitespace-nowrap',
                'transition-colors',
                activeTab === tab.id
                  ? 'bg-cyan-500/20 text-cyan-400'
                  : 'text-text-muted hover:text-text hover:bg-surface-2'
              )}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 scrollbar-thin">
          {activeTab === 'general' && (
            <ModelSettings
              currentModel={currentModel}
              currentProvider={currentProvider}
              providers={providers}
              models={models}
              onSwitchModel={onSwitchModel}
              onSwitchProvider={onSwitchProvider}
            />
          )}

          {activeTab === 'voice' && (
            <VoiceSettingsTab
              voiceSettings={voiceSettings}
              onSetTTSProvider={onSetTTSProvider}
              onSetVoice={onSetVoice}
              edgeVoices={edgeVoices}
              elevenVoices={elevenVoices}
              kokoroVoices={kokoroVoices}
              chutesConfigured={chutesConfigured}
            />
          )}

          {activeTab === 'stt' && (
            <STTSettingsTab
              voiceSettings={voiceSettings}
              onSetSTTProvider={onSetSTTProvider}
              chutesConfigured={chutesConfigured}
              wakeWord={wakeWord}
              wakeWordEnabled={wakeWordEnabled}
              onSetWakeWord={onSetWakeWord}
              onSetWakeWordEnabled={onSetWakeWordEnabled}
            />
          )}

          {activeTab === 'integrations' && (
            <IntegrationsTab />
          )}

          {activeTab === 'memory' && (
            <MemoryPanel />
          )}

          {activeTab === 'system' && (
            <SystemInstructions
              value={systemPrompt}
              onChange={onSetSystemPrompt}
              onReset={onResetSystemPrompt}
              isDefault={isDefaultPrompt}
            />
          )}
        </div>
      </div>
    </>
  )
}

// Model Settings Sub-component
function ModelSettings({
  currentModel,
  currentProvider,
  providers,
  models,
  onSwitchModel,
  onSwitchProvider,
}: {
  currentModel: string
  currentProvider: string
  providers: Record<string, ProviderInfo>
  models: string[]
  onSwitchModel: (model: string) => void
  onSwitchProvider: (provider: string) => void
}) {
  return (
    <div className="space-y-6">
      {/* Provider Selection */}
      <div>
        <h3 className="text-sm font-medium text-text mb-3">Provider</h3>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(providers).map(([name, info]) => (
            <button
              key={name}
              onClick={() => onSwitchProvider(name)}
              className={cn(
                'p-3 rounded-xl text-sm text-left',
                'border transition-all',
                currentProvider === name
                  ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-400'
                  : info.configured
                    ? 'bg-surface-2 border-border/30 text-text hover:border-border'
                    : 'bg-surface border-border/20 text-text-muted/50'
              )}
            >
              <div className="flex items-center justify-between">
                <span className="font-medium capitalize">{name}</span>
                {currentProvider === name && (
                  <Check size={14} className="text-cyan-400" />
                )}
              </div>
              <div className="text-xs mt-1 opacity-60">
                {info.configured ? 'Configured' : 'Not configured'}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Model Selection */}
      <div>
        <h3 className="text-sm font-medium text-text mb-3">Model</h3>
        <div className="space-y-1 max-h-64 overflow-y-auto scrollbar-thin">
          {models.length === 0 ? (
            <p className="text-sm text-text-muted p-3">No models available</p>
          ) : (
            models.map((model) => (
              <button
                key={model}
                onClick={() => onSwitchModel(model)}
                className={cn(
                  'w-full p-3 rounded-xl text-sm text-left',
                  'border transition-all',
                  currentModel === model
                    ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-400'
                    : 'bg-surface border-border/20 text-text hover:border-border/50'
                )}
              >
                <div className="flex items-center justify-between">
                  <span className="truncate">{model}</span>
                  {currentModel === model && (
                    <Check size={14} className="text-cyan-400 flex-shrink-0" />
                  )}
                </div>
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

// Voice Settings Sub-component
function VoiceSettingsTab({
  voiceSettings,
  onSetTTSProvider,
  onSetVoice,
  edgeVoices,
  elevenVoices,
  kokoroVoices,
  chutesConfigured,
}: {
  voiceSettings: VoiceSettings
  onSetTTSProvider: (provider: TTSProvider) => Promise<void>
  onSetVoice: (voiceId: string, provider: 'edge' | 'elevenlabs' | 'kokoro') => Promise<void>
  edgeVoices: { id: string; name: string }[]
  elevenVoices: { id: string; name: string }[]
  kokoroVoices: { id: string; name: string }[]
  chutesConfigured: boolean
}) {
  const [elevenLabsKey, setElevenLabsKey] = useState('')

  const saveElevenLabsKey = async () => {
    if (!elevenLabsKey) return
    await fetch('/api/settings/elevenlabs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: elevenLabsKey }),
    })
    window.location.reload()
  }

  return (
    <div className="space-y-6">
      {/* TTS Provider */}
      <div>
        <h3 className="text-sm font-medium text-text mb-3">TTS Provider</h3>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => onSetTTSProvider('browser')}
            className={cn(
              'p-3 rounded-xl text-sm',
              'border transition-all',
              voiceSettings.tts_provider === 'browser'
                ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-400'
                : 'bg-surface border-border/20 text-text-muted hover:text-text'
            )}
          >
            Browser (Fast)
          </button>
          <button
            onClick={() => onSetTTSProvider('edge')}
            className={cn(
              'p-3 rounded-xl text-sm',
              'border transition-all',
              voiceSettings.tts_provider === 'edge'
                ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-400'
                : 'bg-surface border-border/20 text-text-muted hover:text-text'
            )}
          >
            Edge (Neural)
          </button>
          <button
            onClick={() => onSetTTSProvider('kokoro')}
            disabled={!chutesConfigured}
            className={cn(
              'p-3 rounded-xl text-sm',
              'border transition-all',
              voiceSettings.tts_provider === 'kokoro'
                ? 'bg-purple-500/20 border-purple-500/50 text-purple-400'
                : chutesConfigured
                  ? 'bg-surface border-border/20 text-text-muted hover:text-text'
                  : 'bg-surface border-border/20 text-text-muted/40 cursor-not-allowed'
            )}
            title={chutesConfigured ? 'Kokoro via Chutes' : 'Requires CHUTES_API_KEY'}
          >
            Kokoro (Chutes)
          </button>
          <button
            onClick={() => onSetTTSProvider('elevenlabs')}
            className={cn(
              'p-3 rounded-xl text-sm',
              'border transition-all',
              voiceSettings.tts_provider === 'elevenlabs'
                ? 'bg-purple-500/20 border-purple-500/50 text-purple-400'
                : 'bg-surface border-border/20 text-text-muted hover:text-text'
            )}
          >
            ElevenLabs
          </button>
        </div>
      </div>

      {/* Voice Selection */}
      {voiceSettings.tts_provider === 'browser' && (
        <p className="text-sm text-text-muted p-3 bg-surface/50 rounded-xl">
          Instant response using browser's built-in voice synthesis.
        </p>
      )}

      {voiceSettings.tts_provider === 'edge' && (
        <div>
          <h3 className="text-sm font-medium text-text mb-3">Voice</h3>
          <div className="space-y-1 max-h-48 overflow-y-auto scrollbar-thin">
            {edgeVoices.map((voice) => (
              <button
                key={voice.id}
                onClick={() => onSetVoice(voice.id, 'edge')}
                className={cn(
                  'w-full p-3 rounded-xl text-sm text-left',
                  'border transition-all',
                  voiceSettings.tts_voice === voice.id
                    ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-400'
                    : 'bg-surface border-border/20 text-text hover:border-border/50'
                )}
              >
                {voice.name}
              </button>
            ))}
          </div>
        </div>
      )}

      {voiceSettings.tts_provider === 'kokoro' && (
        <div>
          <h3 className="text-sm font-medium text-text mb-3">Voice</h3>
          <div className="space-y-1 max-h-48 overflow-y-auto scrollbar-thin">
            {kokoroVoices.map((voice) => (
              <button
                key={voice.id}
                onClick={() => onSetVoice(voice.id, 'kokoro')}
                className={cn(
                  'w-full p-3 rounded-xl text-sm text-left',
                  'border transition-all',
                  voiceSettings.tts_voice === voice.id
                    ? 'bg-purple-500/20 border-purple-500/50 text-purple-400'
                    : 'bg-surface border-border/20 text-text hover:border-border/50'
                )}
              >
                {voice.name}
              </button>
            ))}
          </div>
        </div>
      )}

      {voiceSettings.tts_provider === 'elevenlabs' && (
        <div>
          {elevenVoices.length === 0 ? (
            <div className="space-y-3">
              <p className="text-sm text-text-muted">Enter your ElevenLabs API key:</p>
              <input
                type="password"
                value={elevenLabsKey}
                onChange={(e) => setElevenLabsKey(e.target.value)}
                placeholder="xi-xxxxxxxx..."
                className="w-full p-3 rounded-xl bg-surface border border-border/30 text-text text-sm focus:outline-none focus:border-purple-500/50"
              />
              <button
                onClick={saveElevenLabsKey}
                disabled={!elevenLabsKey}
                className="w-full p-3 rounded-xl bg-purple-500/20 text-purple-400 text-sm hover:bg-purple-500/30 disabled:opacity-50"
              >
                Save & Load Voices
              </button>
            </div>
          ) : (
            <div className="space-y-1 max-h-48 overflow-y-auto scrollbar-thin">
              {elevenVoices.map((voice) => (
                <button
                  key={voice.id}
                  onClick={() => onSetVoice(voice.id, 'elevenlabs')}
                  className={cn(
                    'w-full p-3 rounded-xl text-sm text-left',
                    'border transition-all',
                    voiceSettings.tts_voice === voice.id
                      ? 'bg-purple-500/20 border-purple-500/50 text-purple-400'
                      : 'bg-surface border-border/20 text-text hover:border-border/50'
                  )}
                >
                  {voice.name}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// STT Settings Sub-component
function STTSettingsTab({
  voiceSettings,
  onSetSTTProvider,
  chutesConfigured,
  wakeWord,
  wakeWordEnabled,
  onSetWakeWord,
  onSetWakeWordEnabled,
}: {
  voiceSettings: VoiceSettings
  onSetSTTProvider: (provider: STTProvider) => Promise<void>
  chutesConfigured: boolean
  wakeWord: string
  wakeWordEnabled: boolean
  onSetWakeWord: (word: string) => void
  onSetWakeWordEnabled: (enabled: boolean) => void
}) {
  const [editingWakeWord, setEditingWakeWord] = useState(wakeWord)

  return (
    <div className="space-y-6">
      {/* Wake Word Settings */}
      <div>
        <h3 className="text-sm font-medium text-text mb-3">Wake Word</h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between p-3 rounded-xl bg-surface border border-border/20">
            <div>
              <div className="font-medium text-sm">Enable Wake Word</div>
              <div className="text-xs text-text-muted mt-0.5">
                Say "Hey {wakeWord}" to activate voice mode
              </div>
            </div>
            <button
              onClick={() => onSetWakeWordEnabled(!wakeWordEnabled)}
              className={cn(
                'w-12 h-6 rounded-full transition-all relative',
                wakeWordEnabled ? 'bg-listening' : 'bg-surface-2'
              )}
            >
              <div className={cn(
                'absolute top-1 w-4 h-4 rounded-full bg-white transition-all',
                wakeWordEnabled ? 'right-1' : 'left-1'
              )} />
            </button>
          </div>

          <div className="p-3 rounded-xl bg-surface border border-border/20">
            <div className="font-medium text-sm mb-2">Custom Wake Word</div>
            <div className="flex gap-2">
              <input
                type="text"
                value={editingWakeWord}
                onChange={(e) => setEditingWakeWord(e.target.value.toLowerCase())}
                placeholder="jarvis"
                className="flex-1 px-3 py-2 rounded-lg bg-surface-2 border border-border/30 text-sm text-text focus:outline-none focus:border-listening/50"
              />
              <button
                onClick={() => {
                  if (editingWakeWord.trim()) {
                    onSetWakeWord(editingWakeWord.trim())
                  }
                }}
                disabled={!editingWakeWord.trim() || editingWakeWord === wakeWord}
                className="px-4 py-2 rounded-lg bg-listening/20 text-listening text-sm hover:bg-listening/30 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Save
              </button>
            </div>
            <div className="text-xs text-text-muted mt-2">
              Triggered by: "hey {editingWakeWord || 'jarvis'}", "{editingWakeWord || 'jarvis'}", "ok {editingWakeWord || 'jarvis'}"
            </div>
          </div>
        </div>
      </div>

      {/* STT Provider */}
      <div>
        <h3 className="text-sm font-medium text-text mb-3">Speech-to-Text Provider</h3>
        <div className="space-y-2">
          <button
            onClick={() => onSetSTTProvider('browser')}
            className={cn(
              'w-full p-3 rounded-xl text-sm text-left',
              'border transition-all',
              voiceSettings.stt_provider === 'browser'
                ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-400'
                : 'bg-surface border-border/20 text-text-muted hover:text-text'
            )}
          >
            <div className="font-medium">Browser (Instant)</div>
            <div className="text-xs opacity-70 mt-1">
              Real-time transcription using browser API. Fast but less accurate.
            </div>
          </button>
          <button
            onClick={() => onSetSTTProvider('whisper')}
            className={cn(
              'w-full p-3 rounded-xl text-sm text-left',
              'border transition-all',
              voiceSettings.stt_provider === 'whisper'
                ? 'bg-purple-500/20 border-purple-500/50 text-purple-400'
                : 'bg-surface border-border/20 text-text-muted hover:text-text'
            )}
          >
            <div className="font-medium">Whisper (Local)</div>
            <div className="text-xs opacity-70 mt-1">
              OpenAI Whisper running locally. Accurate but requires installation.
            </div>
          </button>
          <button
            onClick={() => onSetSTTProvider('chutes')}
            disabled={!chutesConfigured}
            className={cn(
              'w-full p-3 rounded-xl text-sm text-left',
              'border transition-all',
              voiceSettings.stt_provider === 'chutes'
                ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-400'
                : chutesConfigured
                  ? 'bg-surface border-border/20 text-text-muted hover:text-text'
                  : 'bg-surface border-border/20 text-text-muted/40 cursor-not-allowed'
            )}
            title={chutesConfigured ? 'Whisper Fast via Chutes' : 'Requires CHUTES_API_KEY'}
          >
            <div className="font-medium">Whisper Fast (Chutes)</div>
            <div className="text-xs opacity-70 mt-1">
              {chutesConfigured
                ? 'Fast & accurate Whisper via Chutes API. Recommended.'
                : 'Requires CHUTES_API_KEY in .env'}
            </div>
          </button>
        </div>
      </div>
    </div>
  )
}

// Integrations Tab - Telegram, Calendar, etc.
function IntegrationsTab() {
  const [telegramStatus, setTelegramStatus] = useState<{
    configured: boolean
    bot_username?: string
    webhook_url?: string
  } | null>(null)
  const [telegramToken, setTelegramToken] = useState('')
  const [webhookUrl, setWebhookUrl] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  // Fetch Telegram status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch('/api/integrations/telegram/status')
        if (res.ok) {
          const data = await res.json()
          setTelegramStatus(data)
        }
      } catch (err) {
        console.error('Failed to fetch Telegram status:', err)
      } finally {
        setLoading(false)
      }
    }
    fetchStatus()
  }, [])

  const saveTelegramToken = async () => {
    if (!telegramToken.trim()) return
    setSaving(true)
    setError('')
    setSuccess('')
    try {
      const res = await fetch('/api/integrations/telegram/configure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: telegramToken.trim() }),
      })
      if (res.ok) {
        setSuccess('Telegram bot token saved!')
        setTelegramToken('')
        // Refresh status
        const statusRes = await fetch('/api/integrations/telegram/status')
        if (statusRes.ok) {
          setTelegramStatus(await statusRes.json())
        }
      } else {
        const data = await res.json()
        setError(data.error || 'Failed to save token')
      }
    } catch (err) {
      setError('Network error')
    } finally {
      setSaving(false)
    }
  }

  const setupWebhook = async () => {
    if (!webhookUrl.trim()) return
    setSaving(true)
    setError('')
    setSuccess('')
    try {
      const res = await fetch('/api/integrations/telegram/webhook/setup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: webhookUrl.trim() }),
      })
      const data = await res.json()
      if (res.ok && data.ok) {
        setSuccess('Webhook configured!')
        setWebhookUrl('')
        // Refresh status
        const statusRes = await fetch('/api/integrations/telegram/status')
        if (statusRes.ok) {
          setTelegramStatus(await statusRes.json())
        }
      } else {
        setError(data.error || data.description || 'Failed to setup webhook')
      }
    } catch (err) {
      setError('Network error')
    } finally {
      setSaving(false)
    }
  }

  const removeWebhook = async () => {
    setSaving(true)
    setError('')
    try {
      const res = await fetch('/api/integrations/telegram/webhook/remove', {
        method: 'POST',
      })
      if (res.ok) {
        setSuccess('Webhook removed')
        const statusRes = await fetch('/api/integrations/telegram/status')
        if (statusRes.ok) {
          setTelegramStatus(await statusRes.json())
        }
      }
    } catch (err) {
      setError('Network error')
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="w-6 h-6 animate-spin text-text-muted" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Telegram */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <MessageCircle size={18} className="text-blue-400" />
          <h3 className="text-sm font-medium text-text">Telegram</h3>
          {telegramStatus?.configured && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-success/20 text-success">Connected</span>
          )}
        </div>

        {error && (
          <div className="mb-3 p-3 rounded-xl bg-error/20 border border-error/30 text-error text-sm flex items-center gap-2">
            <AlertCircle size={16} />
            {error}
          </div>
        )}

        {success && (
          <div className="mb-3 p-3 rounded-xl bg-success/20 border border-success/30 text-success text-sm flex items-center gap-2">
            <Check size={16} />
            {success}
          </div>
        )}

        {telegramStatus?.configured ? (
          <div className="space-y-3">
            <div className="p-3 rounded-xl bg-surface border border-border/20">
              <div className="text-sm text-text-muted">Bot Username</div>
              <div className="text-text font-medium">@{telegramStatus.bot_username || 'unknown'}</div>
            </div>

            {telegramStatus.webhook_url ? (
              <div className="p-3 rounded-xl bg-surface border border-border/20">
                <div className="text-sm text-text-muted">Webhook URL</div>
                <div className="text-text text-sm truncate">{telegramStatus.webhook_url}</div>
                <button
                  onClick={removeWebhook}
                  disabled={saving}
                  className="mt-2 text-xs text-error hover:underline"
                >
                  Remove Webhook
                </button>
              </div>
            ) : (
              <div className="p-3 rounded-xl bg-surface border border-border/20 space-y-2">
                <div className="text-sm text-text-muted">Setup Webhook (for production)</div>
                <input
                  type="url"
                  value={webhookUrl}
                  onChange={(e) => setWebhookUrl(e.target.value)}
                  placeholder="https://your-domain.com"
                  className="w-full px-3 py-2 rounded-lg bg-surface-2 border border-border/30 text-sm text-text focus:outline-none focus:border-blue-500/50"
                />
                <button
                  onClick={setupWebhook}
                  disabled={!webhookUrl.trim() || saving}
                  className="w-full py-2 rounded-lg bg-blue-500/20 text-blue-400 text-sm hover:bg-blue-500/30 disabled:opacity-50"
                >
                  {saving ? 'Setting up...' : 'Setup Webhook'}
                </button>
              </div>
            )}

            <div className="p-3 rounded-xl bg-blue-500/10 border border-blue-500/20 text-sm text-text-muted">
              <strong className="text-blue-400">Commands available:</strong>
              <div className="mt-2 grid grid-cols-2 gap-1 text-xs">
                <span>/chat - Send message</span>
                <span>/model - Change model</span>
                <span>/weather - Get weather</span>
                <span>/search - Web search</span>
                <span>/status - System status</span>
                <span>/help - All commands</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <p className="text-sm text-text-muted">
              Connect your Telegram bot to chat with the assistant from anywhere.
            </p>
            <div className="space-y-2">
              <input
                type="password"
                value={telegramToken}
                onChange={(e) => setTelegramToken(e.target.value)}
                placeholder="Bot token from @BotFather"
                className="w-full px-3 py-2 rounded-lg bg-surface-2 border border-border/30 text-sm text-text focus:outline-none focus:border-blue-500/50"
              />
              <button
                onClick={saveTelegramToken}
                disabled={!telegramToken.trim() || saving}
                className="w-full py-2 rounded-lg bg-blue-500/20 text-blue-400 text-sm hover:bg-blue-500/30 disabled:opacity-50"
              >
                {saving ? 'Saving...' : 'Connect Bot'}
              </button>
            </div>
            <div className="p-3 rounded-xl bg-surface border border-border/20 text-sm text-text-muted">
              <strong>How to get a bot token:</strong>
              <ol className="list-decimal list-inside mt-2 space-y-1 text-xs">
                <li>Open Telegram and search for @BotFather</li>
                <li>Send /newbot and follow the prompts</li>
                <li>Copy the token and paste it above</li>
              </ol>
            </div>
          </div>
        )}
      </div>

      {/* Google Calendar */}
      <CalendarIntegration />

      {/* Home Assistant - Coming Soon */}
      <div className="opacity-50">
        <div className="flex items-center gap-2 mb-3">
          <span className="text-lg">🏠</span>
          <h3 className="text-sm font-medium text-text">Home Assistant</h3>
          <span className="text-xs px-2 py-0.5 rounded-full bg-surface-2 text-text-muted">Coming soon</span>
        </div>
        <p className="text-sm text-text-muted">
          Control your smart home devices through voice and chat.
        </p>
      </div>
    </div>
  )
}

// Calendar Integration Component
function CalendarIntegration() {
  const [status, setStatus] = useState<{
    google: { connected: boolean; configured: boolean }
    apple: { connected: boolean; configured: boolean }
    connected: boolean
    configured: boolean
  } | null>(null)
  const [loading, setLoading] = useState(true)
  const [connecting, setConnecting] = useState(false)
  const [showAppleSetup, setShowAppleSetup] = useState(false)
  const [appleId, setAppleId] = useState('')
  const [appleAppPassword, setAppleAppPassword] = useState('')
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  useEffect(() => {
    fetchStatus()
  }, [])

  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/calendar/status')
      if (res.ok) {
        setStatus(await res.json())
      }
    } catch (err) {
      console.error('Failed to fetch calendar status:', err)
    } finally {
      setLoading(false)
    }
  }

  const connectApple = async () => {
    if (!appleId.trim() || !appleAppPassword.trim()) return
    setConnecting(true)
    setError('')
    setSuccess('')

    try {
      const res = await fetch('/api/calendar/apple/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: appleId.trim(), app_password: appleAppPassword.trim() }),
      })
      const data = await res.json()
      if (data.success) {
        setSuccess('Apple Calendar connected!')
        setShowAppleSetup(false)
        setAppleId('')
        setAppleAppPassword('')
        fetchStatus()
      } else {
        setError(data.error || 'Failed to connect')
      }
    } catch (err) {
      setError('Network error')
    } finally {
      setConnecting(false)
    }
  }

  const disconnectApple = async () => {
    try {
      await fetch('/api/calendar/apple/disconnect', { method: 'POST' })
      fetchStatus()
    } catch (err) {
      console.error('Failed to disconnect:', err)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center gap-2 mb-3">
        <Loader2 className="w-4 h-4 animate-spin text-text-muted" />
        <span className="text-sm text-text-muted">Loading calendar...</span>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <span className="text-lg">📅</span>
        <h3 className="text-sm font-medium text-text">Calendar</h3>
        {status?.apple?.connected && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-success/20 text-success">Connected</span>
        )}
      </div>

      {error && (
        <div className="p-3 rounded-xl bg-error/20 border border-error/30 text-error text-sm flex items-center gap-2">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {success && (
        <div className="p-3 rounded-xl bg-success/20 border border-success/30 text-success text-sm flex items-center gap-2">
          <Check size={16} />
          {success}
        </div>
      )}

      {status?.apple?.connected ? (
        <div className="space-y-3">
          <div className="p-3 rounded-xl bg-surface border border-border/20 flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-text">Apple Calendar</p>
              <p className="text-xs text-text-muted">iCloud synced</p>
            </div>
            <Check className="w-5 h-5 text-success" />
          </div>
          <button onClick={disconnectApple} className="text-xs text-error hover:underline">
            Disconnect
          </button>
        </div>
      ) : showAppleSetup ? (
        <div className="space-y-3">
          <input
            type="email"
            value={appleId}
            onChange={(e) => setAppleId(e.target.value)}
            placeholder="Apple ID (email)"
            className="w-full px-3 py-2 rounded-lg bg-surface-2 border border-border/30 text-sm text-text focus:outline-none focus:border-blue-500/50"
          />
          <input
            type="password"
            value={appleAppPassword}
            onChange={(e) => setAppleAppPassword(e.target.value)}
            placeholder="App-Specific Password"
            className="w-full px-3 py-2 rounded-lg bg-surface-2 border border-border/30 text-sm text-text focus:outline-none focus:border-blue-500/50"
          />
          <div className="flex gap-2">
            <button
              onClick={() => setShowAppleSetup(false)}
              className="flex-1 py-2 rounded-lg bg-surface-2 text-text-muted text-sm"
            >
              Cancel
            </button>
            <button
              onClick={connectApple}
              disabled={!appleId.trim() || !appleAppPassword.trim() || connecting}
              className="flex-1 py-2 rounded-lg bg-blue-500/20 text-blue-400 text-sm disabled:opacity-50"
            >
              {connecting ? 'Connecting...' : 'Connect'}
            </button>
          </div>
          <div className="p-3 rounded-lg bg-surface-2/50 text-xs text-text-muted">
            <strong>⚠️ Use an App-Specific Password:</strong>
            <ol className="list-decimal list-inside mt-2 space-y-1">
              <li>Go to <a href="https://appleid.apple.com" target="_blank" rel="noopener" className="text-primary hover:underline">appleid.apple.com</a></li>
              <li>Sign in → App-Specific Passwords</li>
              <li>Generate a password for "Jarvis"</li>
            </ol>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <p className="text-sm text-text-muted">
            Sync your iCloud calendar to see events and get reminders.
          </p>
          <button
            onClick={() => setShowAppleSetup(true)}
            className="w-full py-3 rounded-xl bg-blue-500/20 text-blue-400 text-sm font-medium hover:bg-blue-500/30"
          >
            Connect Apple Calendar
          </button>
        </div>
      )}
    </div>
  )
}
