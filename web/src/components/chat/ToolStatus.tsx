import { cn } from '../../lib/utils'
import { Loader2, Check, X } from 'lucide-react'

export interface LiveToolStatus {
  name: string
  display: string
  status: 'running' | 'complete' | 'error'
  duration?: number
  args?: Record<string, unknown>
}

interface ToolStatusProps {
  tools: LiveToolStatus[]
}

export function ToolStatus({ tools }: ToolStatusProps) {
  if (tools.length === 0) return null

  return (
    <div className="p-3 rounded-xl bg-surface/60 border border-border/20 mr-12 backdrop-blur-sm space-y-1.5">
      {tools.map((tool, idx) => (
        <ToolStatusItem key={`${tool.name}-${idx}`} tool={tool} />
      ))}
    </div>
  )
}

function ToolStatusItem({ tool }: { tool: LiveToolStatus }) {
  return (
    <div className="flex items-center gap-2.5 text-sm py-1">
      {tool.status === 'running' ? (
        <Loader2 size={14} className="animate-spin text-cyan-400 flex-shrink-0" />
      ) : tool.status === 'complete' ? (
        <Check size={14} className="text-emerald-400 flex-shrink-0" />
      ) : (
        <X size={14} className="text-red-400 flex-shrink-0" />
      )}
      <span className={cn(
        'flex-1 min-w-0 truncate',
        tool.status === 'running' ? 'text-text' : 'text-text-muted'
      )}>
        {tool.display}
      </span>
      {tool.duration !== undefined && tool.status !== 'running' && (
        <span className="text-xs text-text-muted/60 flex-shrink-0">
          {tool.duration.toFixed(1)}s
        </span>
      )}
    </div>
  )
}
