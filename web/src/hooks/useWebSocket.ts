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

      case 'cleared':
        setMessages([])
        setStreaming('')
        setRagStatus(null)
        setToolTimeline([])
        break
    }
  }

  const send = useCallback((content: string, chatMode: boolean = false) => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return

    setMessages((prev) => [...prev, { role: 'user', content, timestamp: new Date() }])
    setIsLoading(true)
    setStreaming('')
    setRagStatus(null)  // Reset RAG status for new query
    setToolTimeline([])
    setPendingTools([])

    ws.current.send(JSON.stringify({ type: 'message', content, chat_mode: chatMode }))
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
    send,
    switchModel,
    switchProvider,
    clear,
  }
}
