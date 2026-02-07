import { useEffect, useRef, useState, useCallback } from 'react'

interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  tools?: ToolEvent[]
  media?: MediaInfo
}

interface MediaInfo {
  type: 'image' | 'video' | 'audio'
  path: string
  filename: string
  url?: string
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
  assistantName?: string
  done?: boolean
  error?: string
  message?: string  // Error message from backend
  rag?: RagInfo
  tools?: ToolEvent[]
  intent?: IntentInfo
  reasoning_level?: string
  mode?: string
  context?: ContextStats
  // Media generation fields
  media_type?: 'image' | 'video' | 'audio'
  path?: string
  filename?: string
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

interface LiveToolStatus {
  name: string
  display: string
  status: 'running' | 'complete' | 'error'
  duration?: number
}

export function useWebSocket() {
  const ws = useRef<WebSocket | null>(null)
  const [connected, setConnected] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [streaming, setStreaming] = useState('')
  const [streamingThinking, setStreamingThinking] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [model, setModel] = useState('')
  const [provider, setProvider] = useState('')
  const [project, setProject] = useState('')
  const [assistantName, setAssistantName] = useState('Jarvis')
  const [ragStatus, setRagStatus] = useState<RagInfo | null>(null)
  const [toolTimeline, setToolTimeline] = useState<ToolEvent[]>([])
  const [intentInfo, setIntentInfo] = useState<IntentInfo | null>(null)
  const [contextStats, setContextStats] = useState<ContextStats | null>(null)
  // pendingTools is used via setPendingTools callback form, not directly referenced
  const [liveToolStatus, setLiveToolStatus] = useState<LiveToolStatus[]>([])
  const [, setPendingTools] = useState<ToolEvent[]>([])
  const thinkingStartRef = useRef<number | null>(null)
  const isInsideThinkingRef = useRef(false)  // Track if currently inside <think> block

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
        if (data.assistantName) setAssistantName(data.assistantName)
        break

      case 'thinking':
        // Dedicated thinking stream from backend
        if (!thinkingStartRef.current) {
          thinkingStartRef.current = Date.now()
        }
        setStreamingThinking((prev) => prev + (data.content || ''))
        setIsLoading(false)
        break

      case 'stream':
        // Parse and separate thinking from response content using state tracking
        let chunk = data.content || ''
        let thinkingContent = ''
        let responseContent = ''

        // Process character by character to handle tag boundaries
        while (chunk.length > 0) {
          if (isInsideThinkingRef.current) {
            // Currently inside thinking block - look for closing tag
            const closeMatch = chunk.match(/<\/think(?:ing)?>/i)
            if (closeMatch) {
              // Found closing tag
              const closeIdx = closeMatch.index!
              thinkingContent += chunk.slice(0, closeIdx)
              chunk = chunk.slice(closeIdx + closeMatch[0].length)
              isInsideThinkingRef.current = false
            } else {
              // No closing tag - all content is thinking
              thinkingContent += chunk
              chunk = ''
            }
          } else {
            // Outside thinking block - look for opening tag
            const openMatch = chunk.match(/<think(?:ing)?>/i)
            if (openMatch) {
              // Found opening tag
              const openIdx = openMatch.index!
              responseContent += chunk.slice(0, openIdx)
              chunk = chunk.slice(openIdx + openMatch[0].length)
              isInsideThinkingRef.current = true
              // Start timer when thinking begins
              if (!thinkingStartRef.current) {
                thinkingStartRef.current = Date.now()
              }
            } else {
              // No opening tag - all content is response
              responseContent += chunk
              chunk = ''
            }
          }
        }

        // Update states
        if (thinkingContent) {
          setStreamingThinking((prev) => prev + thinkingContent)
        }
        if (responseContent) {
          setStreaming((prev) => prev + responseContent)
        }
        setIsLoading(false)
        break

      case 'response':
        if (data.done) {
          // Update context stats if provided
          if (data.context) {
            setContextStats(data.context)
          }

          // Calculate thinking duration
          const thinkingDuration = thinkingStartRef.current
            ? (Date.now() - thinkingStartRef.current) / 1000
            : undefined
          thinkingStartRef.current = null
          isInsideThinkingRef.current = false  // Reset for next message

          // Add message with content from response or accumulated streaming
          setStreaming((currentStreaming) => {
            setStreamingThinking((currentThinking) => {
              // Prefer already-parsed streamed content over data.content
              let finalContent = currentStreaming || ''
              let finalThinking = currentThinking?.trim() || ''

              // If we have no streamed content, fall back to data.content and extract thinking
              if (!finalContent && data.content) {
                // Extract thinking from data.content first
                const thinkMatch = data.content.match(/<think(?:ing)?>([\s\S]*?)<\/think(?:ing)?>/i)
                if (thinkMatch && !finalThinking) {
                  finalThinking = thinkMatch[1].trim()
                }
                // Use data.content as fallback
                finalContent = data.content
              }

              // Clean any remaining thinking tags from final content
              if (finalContent) {
                finalContent = finalContent.replace(/<think(?:ing)?>([\s\S]*?)<\/think(?:ing)?>/gi, '')
                finalContent = finalContent.replace(/<\/?think(ing)?>/gi, '')
                finalContent = finalContent.trim()
              }

              // Convert empty string to undefined for thinking
              const thinkingValue = finalThinking || undefined

              if (finalContent || thinkingValue) {
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
                        content: finalContent || '',
                        timestamp: new Date(),
                        tools: currentTools.length ? currentTools : undefined,
                        thinking: thinkingValue,
                        thinkingDuration
                      }
                    ]
                  })
                  return []  // Clear pending tools
                })
              }
              return ''  // Clear streaming thinking
            })
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

      case 'tool_status':
        // Live tool status updates (running/complete/error)
        if (data.tools) {
          setLiveToolStatus(data.tools as unknown as LiveToolStatus[])
        }
        setIsLoading(false)
        break

      case 'tool_timeline':
        if (data.tools?.length) {
          setToolTimeline(data.tools)
          setPendingTools(data.tools)
        } else {
          setToolTimeline([])
          setPendingTools([])
        }
        setLiveToolStatus([])  // Clear live status when timeline arrives
        break

      case 'intent':
        if (data.intent) {
          setIntentInfo(data.intent)
        }
        break

      case 'media':
        // Handle generated media (image, video, audio)
        if (data.media_type && data.path) {
          const mediaInfo: MediaInfo = {
            type: data.media_type as 'image' | 'video' | 'audio',
            path: data.path,
            filename: data.filename || '',
            url: `/api/generated/${data.filename || ''}`
          }
          // Add media message
          setMessages((prev) => [...prev, {
            role: 'assistant',
            content: '',
            timestamp: new Date(),
            media: mediaInfo
          }])
        }
        break

      case 'status':
        // Status updates during long operations (e.g., "Generating image...")
        // Could show as a toast or status indicator
        break

      case 'cleared':
        setMessages([])
        setStreaming('')
        setStreamingThinking('')
        setRagStatus(null)
        setToolTimeline([])
        setLiveToolStatus([])
        setIntentInfo(null)
        setContextStats(null)
        thinkingStartRef.current = null
        isInsideThinkingRef.current = false
        break
    }
  }

  const send = useCallback((content: string, chatMode: boolean = true, reasoningLevel: string | null = null, attachments: string[] = []) => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return

    setMessages((prev) => [...prev, { role: 'user', content, timestamp: new Date() }])
    setIsLoading(true)
    setStreaming('')
    setStreamingThinking('')
    thinkingStartRef.current = null
    isInsideThinkingRef.current = false  // Reset thinking state
    setRagStatus(null)  // Reset RAG status for new query
    setToolTimeline([])
    setPendingTools([])
    setLiveToolStatus([])
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

    // Add file attachments if present
    if (attachments && attachments.length > 0) {
      message.attachments = attachments
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

  const sendWithVideo = useCallback((transcript: string, frame: string | null) => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return

    setMessages((prev) => [...prev, { role: 'user', content: transcript, timestamp: new Date() }])
    setIsLoading(true)
    setStreaming('')
    setStreamingThinking('')
    thinkingStartRef.current = null
    isInsideThinkingRef.current = false
    setRagStatus(null)
    setToolTimeline([])
    setPendingTools([])
    setLiveToolStatus([])
    setIntentInfo(null)

    ws.current.send(JSON.stringify({
      type: 'voice_with_video',
      transcript,
      frame,
    }))
  }, [])

  const clear = useCallback(() => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return
    ws.current.send(JSON.stringify({ type: 'clear' }))
  }, [])

  return {
    connected,
    messages,
    streaming,
    streamingThinking,
    isLoading,
    model,
    provider,
    project,
    assistantName,
    ragStatus,
    toolTimeline,
    liveToolStatus,
    intentInfo,
    contextStats,
    send,
    sendWithVideo,
    switchModel,
    switchProvider,
    clear,
  }
}
