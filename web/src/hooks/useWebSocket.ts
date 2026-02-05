import { useEffect, useRef, useState, useCallback } from 'react'

interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  tools?: ToolEvent[]
}

interface RagInfo {
  enabled: boolean
  chunks: number
  total_chunks?: number
  sources: string[]
  error?: string
}

interface IntentInfo {
  detected: boolean
  intent?: string
  confidence?: number
  reasoning_level?: string
  requires_tools?: boolean
}

interface ContextStats {
  tokens_used: number
  max_tokens: number
  percentage: number
  messages: number
  needs_compact: boolean
  tokens_remaining: number
}

interface WSMessage {
  type: string
  content?: string
  model?: string
  provider?: string
  project?: string
  done?: boolean
  error?: string
  message?: string  // Error message from backend
  rag?: RagInfo
  tools?: ToolEvent[]
  intent?: IntentInfo
  reasoning_level?: string
  mode?: string
  context?: ContextStats
}

interface ToolEvent {
  name: string
  display: string
  duration_s: number
  id?: string | null
  args?: Record<string, unknown>
  result_preview?: string | null
  success?: boolean
}

export function useWebSocket() {
  const ws = useRef<WebSocket | null>(null)
  const [connected, setConnected] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [streaming, setStreaming] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [model, setModel] = useState('')
  const [provider, setProvider] = useState('')
  const [project, setProject] = useState('')
  const [ragStatus, setRagStatus] = useState<RagInfo | null>(null)
  const [toolTimeline, setToolTimeline] = useState<ToolEvent[]>([])
  const [intentInfo, setIntentInfo] = useState<IntentInfo | null>(null)
  const [contextStats, setContextStats] = useState<ContextStats | null>(null)
  // pendingTools is used via setPendingTools callback form, not directly referenced
  const [, setPendingTools] = useState<ToolEvent[]>([])

  useEffect(() => {
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null
    let isCleaningUp = false

    const connect = () => {
      if (isCleaningUp) return

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      ws.current = new WebSocket(`${protocol}//${window.location.host}/ws`)

      ws.current.onopen = () => {
        if (!isCleaningUp) setConnected(true)
      }

      ws.current.onclose = () => {
        if (!isCleaningUp) {
          setConnected(false)
          reconnectTimer = setTimeout(connect, 2000)
        }
      }

      ws.current.onerror = () => {
        // Suppress error logging - reconnect will handle it
      }

      ws.current.onmessage = (event) => {
        const data: WSMessage = JSON.parse(event.data)
        handleMessage(data)
      }
    }

    connect()

    return () => {
      isCleaningUp = true
      if (reconnectTimer) clearTimeout(reconnectTimer)
      if (ws.current?.readyState === WebSocket.OPEN) {
        ws.current.close()
      }
    }
  }, [])

  const handleMessage = (data: WSMessage) => {
    switch (data.type) {
      case 'connected':
        setModel(data.model || '')
        setProvider(data.provider || '')
        setProject(data.project || '')
        break

      case 'stream':
        setStreaming((prev) => prev + (data.content || ''))
        setIsLoading(false)
        break

      case 'response':
        if (data.done) {
          // Update context stats if provided
          if (data.context) {
            setContextStats(data.context)
          }
          // Add message with content from response or accumulated streaming
          setStreaming((currentStreaming) => {
            const finalContent = data.content || currentStreaming
            if (finalContent) {
              // Get pending tools and add message
              setPendingTools((currentTools) => {
                setMessages((prev) => {
                  // Prevent duplicate messages
                  const lastMsg = prev[prev.length - 1]
                  if (lastMsg?.role === 'assistant' && lastMsg?.content === finalContent) {
                    return prev  // Skip duplicate
                  }
                  return [
                    ...prev,
                    {
                      role: 'assistant',
                      content: finalContent,
                      timestamp: new Date(),
                      tools: currentTools.length ? currentTools : undefined
                    }
                  ]
                })
                return []  // Clear pending tools
              })
            }
            return ''  // Clear streaming
          })
        }
        setIsLoading(false)
        break

      case 'error':
        const errorMsg = data.error || data.message
        if (errorMsg) {
          setMessages((prev) => [...prev, { role: 'system', content: `Error: ${errorMsg}`, timestamp: new Date() }])
        }
        setIsLoading(false)
        setStreaming('')
        break

      case 'model_changed':
        setModel(data.model || '')
        break

      case 'provider_changed':
        setProvider(data.provider || '')
        setModel(data.model || '')
        break

      case 'rag_status':
        if (data.rag) {
          setRagStatus(data.rag)
        }
        break

      case 'tool_timeline':
        if (data.tools?.length) {
          setToolTimeline(data.tools)
          setPendingTools(data.tools)
        } else {
          setToolTimeline([])
          setPendingTools([])
        }
        break

      case 'intent':
        if (data.intent) {
          setIntentInfo(data.intent)
        }
        break

      case 'cleared':
        setMessages([])
        setStreaming('')
        setRagStatus(null)
        setToolTimeline([])
        setIntentInfo(null)
        setContextStats(null)
        break
    }
  }

  const send = useCallback((content: string, chatMode: boolean = true, reasoningLevel: string | null = null) => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return

    setMessages((prev) => [...prev, { role: 'user', content, timestamp: new Date() }])
    setIsLoading(true)
    setStreaming('')
    setRagStatus(null)  // Reset RAG status for new query
    setToolTimeline([])
    setPendingTools([])
    setIntentInfo(null)  // Reset intent for new query

    const message: Record<string, unknown> = {
      type: 'message',
      content,
      chat_mode: chatMode
    }

    // Add reasoning level if user specified an override
    if (reasoningLevel) {
      message.reasoning_level = reasoningLevel
    }

    ws.current.send(JSON.stringify(message))
  }, [])

  const switchModel = useCallback((newModel: string) => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return
    ws.current.send(JSON.stringify({ type: 'switch_model', model: newModel }))
  }, [])

  const switchProvider = useCallback((newProvider: string) => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return
    ws.current.send(JSON.stringify({ type: 'switch_provider', provider: newProvider }))
  }, [])

  const clear = useCallback(() => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return
    ws.current.send(JSON.stringify({ type: 'clear' }))
  }, [])

  return {
    connected,
    messages,
    streaming,
    isLoading,
    model,
    provider,
    project,
    ragStatus,
    toolTimeline,
    intentInfo,
    contextStats,
    send,
    switchModel,
    switchProvider,
    clear,
  }
}
