import { useEffect, useRef, useState, useCallback } from 'react'

interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
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
  rag?: RagInfo
  chat_id?: string
  messages?: Array<{ role: 'user' | 'assistant' | 'system', content: string }>
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
  const [chatId, setChatId] = useState<string | null>(null)

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
        // Restore chat state if available
        if (data.chat_id) {
          setChatId(data.chat_id)
        }
        if (data.messages && data.messages.length > 0) {
          setMessages(data.messages.map(m => ({
            role: m.role,
            content: m.content,
            timestamp: new Date()
          })))
        }
        break

      case 'stream':
        setStreaming((prev) => prev + (data.content || ''))
        setIsLoading(false)
        break

      case 'response':
        if (data.done && data.content) {
          setMessages((prev) => [...prev, { role: 'assistant', content: data.content!, timestamp: new Date() }])
          setStreaming('')
        }
        setIsLoading(false)
        break

      case 'error':
        setMessages((prev) => [...prev, { role: 'system', content: `Error: ${data.error}`, timestamp: new Date() }])
        setIsLoading(false)
        setStreaming('')
        break

      case 'model_changed':
        setModel(data.model || '')
        break

      case 'rag_status':
        if (data.rag) {
          setRagStatus(data.rag)
        }
        break

      case 'cleared':
        setMessages([])
        setStreaming('')
        setRagStatus(null)
        if (data.chat_id) {
          setChatId(data.chat_id)
        }
        break

      case 'chat_switched':
        if (data.chat_id) {
          setChatId(data.chat_id)
        }
        if (data.messages) {
          setMessages(data.messages.map((m: { role: 'user' | 'assistant' | 'system', content: string }) => ({
            role: m.role,
            content: m.content,
            timestamp: new Date()
          })))
        } else {
          setMessages([])
        }
        setStreaming('')
        setRagStatus(null)
        break
    }
  }

  const send = useCallback((content: string, chatMode: boolean = false) => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return

    setMessages((prev) => [...prev, { role: 'user', content, timestamp: new Date() }])
    setIsLoading(true)
    setStreaming('')
    setRagStatus(null)  // Reset RAG status for new query

    ws.current.send(JSON.stringify({ type: 'message', content, chat_mode: chatMode }))
  }, [])

  const switchModel = useCallback((newModel: string) => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return
    ws.current.send(JSON.stringify({ type: 'switch_model', model: newModel }))
  }, [])

  const clear = useCallback(() => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return
    ws.current.send(JSON.stringify({ type: 'clear' }))
  }, [])

  const switchChat = useCallback((chatId: string) => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return
    ws.current.send(JSON.stringify({ type: 'switch_chat', chat_id: chatId }))
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
    chatId,
    send,
    switchModel,
    clear,
    switchChat,
  }
}
