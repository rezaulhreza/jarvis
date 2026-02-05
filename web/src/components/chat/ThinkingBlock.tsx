import { useState } from 'react'
import { ChevronDown, ChevronRight, Brain } from 'lucide-react'
import { cn } from '../../lib/utils'

interface ThinkingBlockProps {
  content: string
  isStreaming?: boolean
  duration?: number
}

export function ThinkingBlock({ content, isStreaming = false, duration }: ThinkingBlockProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  if (!content) return null

  // Count approximate tokens/words for display
  const wordCount = content.split(/\s+/).length

  return (
    <div className="mb-3">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={cn(
          "flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-all w-full",
          "bg-surface-2/50 hover:bg-surface-2/70 border border-border/30",
          isStreaming && "animate-pulse"
        )}
      >
        <Brain size={16} className={cn(
          "text-purple-400",
          isStreaming && "animate-spin"
        )} />
        <span className="text-text-muted">
          {isStreaming ? "Thinking..." : "Thought process"}
        </span>
        <span className="text-text-muted/60 text-xs ml-auto">
          {isStreaming ? (
            <span className="inline-flex items-center gap-1">
              <span className="w-1 h-1 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <span className="w-1 h-1 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <span className="w-1 h-1 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </span>
          ) : (
            <>
              {wordCount} words
              {duration && ` · ${duration.toFixed(1)}s`}
            </>
          )}
        </span>
        {isExpanded ? (
          <ChevronDown size={16} className="text-text-muted" />
        ) : (
          <ChevronRight size={16} className="text-text-muted" />
        )}
      </button>

      {isExpanded && (
        <div className={cn(
          "mt-2 p-3 rounded-lg text-sm",
          "bg-purple-500/5 border border-purple-500/20",
          "text-text-muted/80 leading-relaxed",
          "max-h-64 overflow-y-auto scrollbar-thin"
        )}>
          <pre className="whitespace-pre-wrap font-sans text-xs">
            {content}
            {isStreaming && <span className="inline-block w-1.5 h-4 ml-0.5 bg-purple-400 animate-pulse" />}
          </pre>
        </div>
      )}
    </div>
  )
}
