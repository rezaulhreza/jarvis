// Shared TypeScript types for Jarvis UI

// === Message Types ===
export interface ToolEvent {
  name: string
  display: string
  duration_s: number
  id?: string | null
  args?: Record<string, unknown>
  result_preview?: string | null
  success?: boolean
}

export interface MediaInfo {
  type: 'image' | 'video' | 'audio'
  path: string
  filename: string
  url?: string
}

export interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  tools?: ToolEvent[]
  media?: MediaInfo
  thinking?: string
  thinkingDuration?: number
}

// === RAG Types ===
export interface RagInfo {
  enabled: boolean
  chunks: number
  total_chunks?: number
  sources: string[]
  error?: string
}

// === Intent Types ===
export interface IntentInfo {
  detected: boolean
  intent?: string
  confidence?: number
  reasoning_level?: string
  requires_tools?: boolean
}

// === Context Types ===
export interface ContextStats {
  tokens_used: number
  max_tokens: number
  percentage: number
  messages: number
  needs_compact: boolean
  tokens_remaining: number
}

// === Voice Types ===
export type TTSProvider = 'browser' | 'edge' | 'elevenlabs' | 'kokoro'
export type STTProvider = 'browser' | 'whisper' | 'chutes'

export interface VoiceSettings {
  tts_provider: TTSProvider
  tts_voice: string
  stt_provider: STTProvider
  wake_word?: string
  wake_word_enabled?: boolean
}

// === Mode Types ===
export type ViewMode = 'chat' | 'voice'
export type ReasoningLevel = 'fast' | 'balanced' | 'deep' | null

// === Orb Types ===
export type OrbState = 'idle' | 'listening' | 'speaking' | 'thinking'

export interface OrbProps {
  state: OrbState
  volume: number
  playbackVolume: number
  onClick: () => void
  size?: 'sm' | 'md' | 'lg' | 'xl'
}

// === File Upload Types ===
export type FileType = 'image' | 'video' | 'audio' | 'document'
export type UploadStatus = 'pending' | 'uploading' | 'uploaded' | 'error'

export interface UploadedFile {
  id: string
  name: string
  type: FileType
  mimeType: string
  size: number
  preview?: string  // Base64 for images
  status: UploadStatus
  error?: string
  uploadProgress?: number
}

// === Settings Types ===
export interface SystemPromptSettings {
  content: string
  isDefault: boolean
  lastModified?: Date
}

export interface ProviderInfo {
  configured: boolean
  model?: string | null
}

// === WebSocket Types ===
export interface WSMessage {
  type: string
  content?: string
  model?: string
  provider?: string
  project?: string
  done?: boolean
  error?: string
  message?: string
  rag?: RagInfo
  tools?: ToolEvent[]
  intent?: IntentInfo
  reasoning_level?: string
  mode?: string
  context?: ContextStats
}
