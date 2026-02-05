import { cn } from '../../lib/utils'
import type { ReactNode } from 'react'

interface WidgetCardProps {
  title?: string
  icon?: ReactNode
  children: ReactNode
  className?: string
  noPadding?: boolean
  glowing?: boolean
}

export function WidgetCard({
  title,
  icon,
  children,
  className = '',
  noPadding = false,
  glowing = false,
}: WidgetCardProps) {
  return (
    <div
      className={cn(
        'rounded-2xl border border-primary/20 bg-surface/80 backdrop-blur-sm',
        'transition-all duration-300',
        glowing && 'shadow-glow-cyan',
        className
      )}
    >
      {title && (
        <div className="flex items-center gap-2 px-4 py-3 border-b border-border/20">
          {icon && <span className="text-primary">{icon}</span>}
          <h3 className="text-xs font-semibold uppercase tracking-wider text-text-muted">
            {title}
          </h3>
        </div>
      )}
      <div className={cn(!noPadding && 'p-4')}>{children}</div>
    </div>
  )
}
