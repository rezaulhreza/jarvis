import { cn } from '../../lib/utils'
import type { Message, ToolEvent } from '../../types'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ThinkingBlock } from './ThinkingBlock'

interface MessageBubbleProps {
  message: Message
}

// Extract URLs from text
const extractUrls = (text?: string): string[] => {
  if (!text) return []
  const matches = text.match(/https?:\/\/[^\s)]+/g) || []
  return Array.from(new Set(matches))
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const { role, content, timestamp, tools, media, thinking, thinkingDuration } = message

  if (role === 'system') {
    return (
      <div className="p-4 rounded-xl bg-transparent text-center text-text-muted text-sm">
        {content}
      </div>
    )
  }

  const isUser = role === 'user'

  const ts = timestamp instanceof Date ? timestamp : timestamp ? new Date(timestamp) : null
  const timeStr = ts?.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  })

  return (
    <div className={cn('space-y-3', isUser ? 'flex flex-col items-end' : 'flex flex-col items-start')}>
      {/* Message bubble */}
      <div
        className={cn(
          'p-4 transition-colors',
          isUser
            ? 'max-w-[80%] bg-cyan-500/15 border border-cyan-500/20 rounded-2xl rounded-br-md'
            : 'max-w-[85%] glass-card rounded-2xl rounded-bl-md'
        )}
      >
        {isUser ? (
          <>
            <p className="whitespace-pre-wrap text-text leading-relaxed">{content}</p>
            {/* Timestamp inline bottom-right */}
            <div className="flex justify-end mt-1.5">
              <span className="text-[10px] text-text-muted/50">{timeStr}</span>
            </div>
          </>
        ) : (
          <>
            {/* Avatar + name header */}
            <div className="flex items-center gap-2 mb-2">
              <img
                src="/jarvis.jpeg"
                alt="Jarvis"
                className="w-5 h-5 rounded-full object-cover"
              />
              <span className="text-xs font-medium text-emerald-400">Jarvis</span>
              <span className="text-[10px] text-text-muted/50 ml-auto">{timeStr}</span>
            </div>

            {/* Thinking block */}
            {thinking && (
              <ThinkingBlock content={thinking} duration={thinkingDuration} />
            )}
            {/* Media content */}
            {media && (
              <div className="mb-3">
                {media.type === 'image' && media.url && (
                  <div className="rounded-lg overflow-hidden border border-border/30">
                    <img
                      src={media.url}
                      alt={media.filename || 'Generated image'}
                      className="max-w-full h-auto"
                      loading="lazy"
                    />
                    <div className="p-2 bg-surface-2/50 text-xs text-text-muted flex items-center justify-between">
                      <span>{media.filename}</span>
                      <a
                        href={media.url}
                        download={media.filename}
                        className="text-cyan-400 hover:text-cyan-300 transition-colors"
                      >
                        Download
                      </a>
                    </div>
                  </div>
                )}
                {media.type === 'video' && media.url && (
                  <div className="rounded-lg overflow-hidden border border-border/30">
                    <video
                      src={media.url}
                      controls
                      className="max-w-full h-auto"
                    />
                    <div className="p-2 bg-surface-2/50 text-xs text-text-muted flex items-center justify-between">
                      <span>{media.filename}</span>
                      <a
                        href={media.url}
                        download={media.filename}
                        className="text-cyan-400 hover:text-cyan-300 transition-colors"
                      >
                        Download
                      </a>
                    </div>
                  </div>
                )}
                {media.type === 'audio' && media.url && (
                  <div className="rounded-lg overflow-hidden border border-border/30 p-3 bg-surface-2/50">
                    <audio
                      src={media.url}
                      controls
                      className="w-full"
                    />
                    <div className="mt-2 text-xs text-text-muted flex items-center justify-between">
                      <span>{media.filename}</span>
                      <a
                        href={media.url}
                        download={media.filename}
                        className="text-cyan-400 hover:text-cyan-300 transition-colors"
                      >
                        Download
                      </a>
                    </div>
                  </div>
                )}
              </div>
            )}
            {/* Text content */}
            {content && (
              <div className="prose prose-invert prose-sm max-w-none text-text leading-relaxed
                prose-p:my-2 prose-p:leading-relaxed
                prose-headings:text-text prose-headings:font-semibold prose-headings:mt-4 prose-headings:mb-2
                prose-h1:text-xl prose-h2:text-lg prose-h3:text-base
                prose-strong:text-text prose-strong:font-semibold
                prose-em:text-text/90
                prose-code:text-cyan-400 prose-code:bg-surface-2 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-code:before:content-none prose-code:after:content-none
                prose-pre:bg-surface-2 prose-pre:border prose-pre:border-border/30 prose-pre:rounded-lg prose-pre:my-3
                prose-a:text-cyan-400 prose-a:no-underline hover:prose-a:underline
                prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5
                prose-blockquote:border-l-primary prose-blockquote:text-text-muted prose-blockquote:not-italic
                prose-table:text-sm prose-th:bg-surface-2 prose-td:border-border/30
              ">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
              </div>
            )}
          </>
        )}
      </div>

      {/* Tool timeline */}
      {tools && tools.length > 0 && !isUser && (
        <div className="max-w-[85%]">
          <ToolTimeline tools={tools} />
        </div>
      )}
    </div>
  )
}

interface ToolTimelineProps {
  tools: ToolEvent[]
}

function ToolTimeline({ tools }: ToolTimelineProps) {
  return (
    <div className="p-4 rounded-2xl glass-card">
      <div className="flex items-center justify-between mb-3">
        <p className="text-xs text-text-muted uppercase tracking-wider font-medium">
          Tools
        </p>
        <p className="text-xs text-text-muted/60">
          {tools.length} step{tools.length > 1 ? 's' : ''}
        </p>
      </div>
      <div className="space-y-2">
        {tools.map((tool, idx) => (
          <ToolItem key={`${tool.name}-${idx}`} tool={tool} />
        ))}
      </div>
    </div>
  )
}

interface ToolItemProps {
  tool: ToolEvent
}

function ToolItem({ tool }: ToolItemProps) {
  const urls = extractUrls(tool.result_preview ?? undefined)

  return (
    <details className="group rounded-xl bg-surface/50 border border-border/30 p-3 transition-colors hover:border-border/50">
      <summary className="flex items-start gap-3 text-sm cursor-pointer list-none select-none">
        <span
          className={cn(
            'mt-1.5 h-2 w-2 rounded-full flex-shrink-0',
            'transition-all duration-300',
            tool.success === false
              ? 'bg-red-400 shadow-glow-red'
              : 'bg-emerald-400 shadow-glow-green-sm'
          )}
        />
        <div className="flex-1 min-w-0">
          <div className="text-text font-medium">{tool.display}</div>
          {tool.result_preview && (
            <div className="text-xs text-text-muted mt-1 line-clamp-2">
              {tool.result_preview}
            </div>
          )}
        </div>
        <div className="text-xs text-text-muted/60 flex-shrink-0">
          {tool.duration_s?.toFixed(1)}s
        </div>
      </summary>
      <div className="mt-3 text-xs text-text-muted space-y-3 border-t border-border/20 pt-3">
        {tool.args && Object.keys(tool.args).length > 0 && (
          <div>
            <div className="text-text-muted/60 mb-1.5 uppercase tracking-wider text-[10px]">
              Arguments
            </div>
            <pre className="whitespace-pre-wrap text-cyan-300/80 bg-background/50 p-2.5 rounded-lg border border-border/20 font-mono text-[11px]">
              {JSON.stringify(tool.args, null, 2)}
            </pre>
          </div>
        )}
        {tool.result_preview && (
          <div>
            <div className="text-text-muted/60 mb-1.5 uppercase tracking-wider text-[10px]">
              Result
            </div>
            <div className="whitespace-pre-wrap text-text/80 bg-background/50 p-2.5 rounded-lg border border-border/20">
              {tool.result_preview}
            </div>
          </div>
        )}
        {urls.length > 0 && (
          <div>
            <div className="text-text-muted/60 mb-1.5 uppercase tracking-wider text-[10px]">
              Sources
            </div>
            <div className="flex flex-col gap-1.5">
              {urls.map((url) => (
                <a
                  key={url}
                  href={url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-cyan-400 hover:text-cyan-300 hover:underline break-all transition-colors"
                >
                  {url}
                </a>
              ))}
            </div>
          </div>
        )}
      </div>
    </details>
  )
}

export { ToolTimeline }
