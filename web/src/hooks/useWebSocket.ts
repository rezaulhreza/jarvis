import { useEffect, useRef, useState, useCallback } from 'react'

interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
}

interface WSMessage {
  type: string
  content?: string
  model?: string
  provider?: string
  project?: string
  done?: boolean
  error?: string
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

  useEffect(() => {
    const connect = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      ws.current = new WebSocket(`${protocol}//${window.location.host}/ws`)

      ws.current.onopen = () => {
        setConnected(true)
      }

      ws.current.onclose = () => {
        setConnected(false)
        setTimeout(connect, 3000)
      }

      ws.current.onmessage = (event) => {
        const data: WSMessage = JSON.parse(event.data)
        handleMessage(data)
      }
    }

    connect()

    return () => {
      ws.current?.close()
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
        if (data.done && data.content) {
          setMessages((prev) => [...prev, { role: 'assistant', content: data.content! }])
          setStreaming('')
        }
        setIsLoading(false)
        break

      case 'error':
        setMessages((prev) => [...prev, { role: 'system', content: `Error: ${data.error}` }])
        setIsLoading(false)
        setStreaming('')
        break

      case 'model_changed':
        setModel(data.model || '')
        break

      case 'cleared':
        setMessages([])
        setStreaming('')
        break
    }
  }

  const send = useCallback((content: string, chatMode: boolean = false) => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return

    setMessages((prev) => [...prev, { role: 'user', content }])
    setIsLoading(true)
    setStreaming('')

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

  return {
    connected,
    messages,
    streaming,
    isLoading,
    model,
    provider,
    project,
    send,
    switchModel,
    clear,
  }
}
