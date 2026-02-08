import { useRef, useEffect, memo } from 'react'
import { MessageBubble } from './MessageBubble'
import { ThinkingBlock } from './ThinkingBlock'
import { ToolStatus, type LiveToolStatus } from './ToolStatus'
import { EmptyState } from './EmptyState'
import type { Message } from '../../types'
import { Loader2 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface MessageListProps {
  messages: Message[]
  streaming: string
  streamingThinking?: string
  isLoading: boolean
  loadingText?: string
  liveToolStatus?: LiveToolStatus[]
  assistantName?: string
  onQuickAction?: (prompt: string) => void
}

// Memoize completed messages to avoid re-renders during streaming
const MemoizedMessageBubble = memo(MessageBubble)

export function MessageList({
  messages,
  streaming,
  streamingThinking,
  isLoading,
  loadingText = 'Thinking...',
  liveToolStatus = [],
  assistantName,
  onQuickAction,
}: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streaming, streamingThinking, liveToolStatus])

  return (
    <div className="flex-1 overflow-y-auto p-4 scrollbar-thin">
      <div className="max-w-3xl mx-auto space-y-4">
        {/* Empty state with quick actions */}
        {messages.length === 0 && !streaming && !isLoading && (
          <EmptyState
            assistantName={assistantName}
            onQuickAction={onQuickAction || (() => {})}
          />
        )}

        {/* Message list - use memoized bubbles for completed messages */}
        {messages.map((msg, i) => (
          <MemoizedMessageBubble key={i} message={msg} />
        ))}

        {/* Live tool status - shows during tool execution */}
        {liveToolStatus.length > 0 && (
          <ToolStatus tools={liveToolStatus} />
        )}

        {/* Streaming thinking */}
        {streamingThinking && !streaming && (
          <div className="max-w-[85%]">
            <ThinkingBlock content={streamingThinking} isStreaming={true} />
          </div>
        )}

        {/* Streaming response */}
        {(streaming || streamingThinking) && (
          <div className="flex flex-col items-start">
            <div className="max-w-[85%] p-4 rounded-2xl rounded-bl-md glass-card">
              {/* Avatar + name header */}
              <div className="flex items-center gap-2 mb-2">
                <img
                  src="/jarvis.jpeg"
                  alt="Jarvis"
                  className="w-5 h-5 rounded-full object-cover"
                />
                <span className="text-xs font-medium text-emerald-400">Jarvis</span>
              </div>

              {/* Show thinking block while streaming response */}
              {streamingThinking && streaming && (
                <ThinkingBlock content={streamingThinking} isStreaming={!streaming} />
              )}

              {/* Response content */}
              {streaming && (
                <div className="prose prose-invert prose-sm max-w-none text-text leading-relaxed
                  prose-p:my-2 prose-p:leading-relaxed
                  prose-headings:text-text prose-headings:font-semibold
                  prose-code:text-cyan-400 prose-code:bg-surface-2 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-code:before:content-none prose-code:after:content-none
                  prose-pre:bg-surface-2 prose-pre:border prose-pre:border-border/30 prose-pre:rounded-lg
                  prose-a:text-cyan-400 prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5
                ">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{streaming}</ReactMarkdown>
                  <span className="inline-block w-2 ml-0.5 animate-pulse text-cyan-400">▍</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Loading indicator */}
        {isLoading && !streaming && liveToolStatus.length === 0 && (
          <div className="flex items-center gap-3 text-text-muted py-2">
            <div className="relative">
              <Loader2 size={18} className="animate-spin text-cyan-500" />
              <div className="absolute inset-0 blur-md bg-cyan-500/30" />
            </div>
            <span className="text-sm">{loadingText}</span>
          </div>
        )}

        {/* Scroll anchor */}
        <div ref={messagesEndRef} />
      </div>
    </div>
  )
}
