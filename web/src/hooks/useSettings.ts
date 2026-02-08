import { useState, useCallback, useEffect } from 'react'
import type {
  VoiceSettings,
  TTSProvider,
  STTProvider,
  ProviderInfo,
} from '../types'

interface UseSettingsReturn {
  // Voice settings
  voiceSettings: VoiceSettings
  setTTSProvider: (provider: TTSProvider) => Promise<void>
  setSTTProvider: (provider: STTProvider) => Promise<void>
  setVoice: (voiceId: string, provider: 'edge' | 'elevenlabs' | 'kokoro') => Promise<void>

  // System prompt
  systemPrompt: string
  setSystemPrompt: (prompt: string) => Promise<void>
  resetSystemPrompt: () => Promise<void>
  isDefaultPrompt: boolean

  // Providers
  providers: Record<string, ProviderInfo>
  refreshProviders: () => Promise<void>

  // Voices
  edgeVoices: { id: string; name: string }[]
  elevenVoices: { id: string; name: string }[]
  kokoroVoices: { id: string; name: string }[]
  refreshVoices: () => Promise<void>

  // Loading states
  isLoading: boolean
}

const DEFAULT_VOICE_SETTINGS: VoiceSettings = {
  tts_provider: 'browser',
  tts_voice: 'en-GB-SoniaNeural',
  stt_provider: 'browser',
}

export function useSettings(): UseSettingsReturn {
  const [voiceSettings, setVoiceSettings] = useState<VoiceSettings>(DEFAULT_VOICE_SETTINGS)
  const [systemPrompt, setSystemPromptState] = useState('')
  const [isDefaultPrompt, setIsDefaultPrompt] = useState(true)
  const [providers, setProviders] = useState<Record<string, ProviderInfo>>({})
  const [edgeVoices, setEdgeVoices] = useState<{ id: string; name: string }[]>([])
  const [elevenVoices, setElevenVoices] = useState<{ id: string; name: string }[]>([])
  const [kokoroVoices, setKokoroVoices] = useState<{ id: string; name: string }[]>([])
  const [isLoading, setIsLoading] = useState(true)

  // Load initial settings
  useEffect(() => {
    const loadSettings = async () => {
      setIsLoading(true)
      try {
        // Load voice settings
        const voiceRes = await fetch('/api/settings/voice')
        if (voiceRes.ok) {
          const data = await voiceRes.json()
          setVoiceSettings({
            tts_provider: data.tts_provider || 'browser',
            tts_voice: data.tts_voice || 'en-GB-SoniaNeural',
            stt_provider: data.stt_provider || 'browser',
          })
        }

        // Load user custom instructions (not the soul)
        const promptRes = await fetch('/api/system-instructions')
        if (promptRes.ok) {
          const data = await promptRes.json()
          setSystemPromptState(data.content || '')
          setIsDefaultPrompt(data.isEmpty ?? true)
        }

        // Load providers
        const providersRes = await fetch('/api/providers')
        if (providersRes.ok) {
          const data = await providersRes.json()
          setProviders(data.providers || {})
        }

        // Load voices
        const [voicesRes, elevenRes, kokoroRes] = await Promise.all([
          fetch('/api/voices'),
          fetch('/api/elevenlabs/voices'),
          fetch('/api/kokoro/voices').catch(() => null),
        ])
        if (voicesRes.ok) {
          const data = await voicesRes.json()
          setEdgeVoices(data.voices || [])
        }
        if (elevenRes.ok) {
          const data = await elevenRes.json()
          setElevenVoices(data.voices || [])
        }
        if (kokoroRes?.ok) {
          const data = await kokoroRes.json()
          setKokoroVoices(data.voices || [])
        }
      } catch (error) {
        console.error('Failed to load settings:', error)
      } finally {
        setIsLoading(false)
      }
    }

    loadSettings()
  }, [])

  // TTS provider
  const setTTSProvider = useCallback(async (provider: TTSProvider) => {
    setVoiceSettings((prev) => ({ ...prev, tts_provider: provider }))
    await fetch('/api/settings/voice', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tts_provider: provider }),
    })
  }, [])

  // STT provider
  const setSTTProvider = useCallback(async (provider: STTProvider) => {
    setVoiceSettings((prev) => ({ ...prev, stt_provider: provider }))
    await fetch('/api/settings/voice', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stt_provider: provider }),
    })
  }, [])

  // Set voice
  const setVoice = useCallback(
    async (voiceId: string, provider: 'edge' | 'elevenlabs' | 'kokoro') => {
      setVoiceSettings((prev) => ({ ...prev, tts_voice: voiceId }))
      if (provider === 'elevenlabs') {
        await fetch('/api/settings/elevenlabs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ voice_id: voiceId }),
        })
      } else if (provider === 'kokoro') {
        await fetch('/api/settings/kokoro', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ voice_id: voiceId }),
        })
      } else {
        await fetch('/api/settings/voice', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ voice: voiceId }),
        })
      }
    },
    []
  )

  // User custom instructions (not the soul)
  const setSystemPrompt = useCallback(async (prompt: string) => {
    setSystemPromptState(prompt)
    setIsDefaultPrompt(!prompt.trim())
    await fetch('/api/system-instructions', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: prompt }),
    })
  }, [])

  const resetSystemPrompt = useCallback(async () => {
    const res = await fetch('/api/system-instructions', { method: 'DELETE' })
    if (res.ok) {
      setSystemPromptState('')
      setIsDefaultPrompt(true)
    }
  }, [])

  // Refresh functions
  const refreshProviders = useCallback(async () => {
    const res = await fetch('/api/providers')
    if (res.ok) {
      const data = await res.json()
      setProviders(data.providers || {})
    }
  }, [])

  const refreshVoices = useCallback(async () => {
    const [voicesRes, elevenRes, kokoroRes] = await Promise.all([
      fetch('/api/voices'),
      fetch('/api/elevenlabs/voices'),
      fetch('/api/kokoro/voices').catch(() => null),
    ])
    if (voicesRes.ok) {
      const data = await voicesRes.json()
      setEdgeVoices(data.voices || [])
    }
    if (elevenRes.ok) {
      const data = await elevenRes.json()
      setElevenVoices(data.voices || [])
    }
    if (kokoroRes?.ok) {
      const data = await kokoroRes.json()
      setKokoroVoices(data.voices || [])
    }
  }, [])

  return {
    voiceSettings,
    setTTSProvider,
    setSTTProvider,
    setVoice,
    systemPrompt,
    setSystemPrompt,
    resetSystemPrompt,
    isDefaultPrompt,
    providers,
    refreshProviders,
    edgeVoices,
    elevenVoices,
    kokoroVoices,
    refreshVoices,
    isLoading,
  }
}
