import { Sparkles } from 'lucide-react'

interface QuickCommand {
  id: string
  label: string
  icon?: string
}

interface QuickCommandsWidgetProps {
  className?: string
  commands?: QuickCommand[]
  onCommand: (command: string) => void
}

const DEFAULT_COMMANDS: QuickCommand[] = [
  { id: '1', label: 'What time is it?' },
  { id: '2', label: "How's the weather?" },
  { id: '3', label: 'Show system status' },
  { id: '4', label: 'Create temp email' },
]

export function QuickCommandsWidget({
  className = '',
  commands = DEFAULT_COMMANDS,
  onCommand,
}: QuickCommandsWidgetProps) {
  return (
    <div className={className}>
      <p className="text-xs text-text-muted mb-3">Quick commands:</p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
        {commands.map((cmd) => (
          <button
            key={cmd.id}
            onClick={() => onCommand(cmd.label)}
            className="flex items-center gap-2 p-2.5 rounded-lg bg-surface/50 border border-border/20 text-sm text-text-muted hover:text-text hover:border-primary/30 hover:bg-primary/5 transition-all text-left"
          >
            <Sparkles size={12} className="text-primary shrink-0" />
            <span className="truncate">{cmd.label}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
