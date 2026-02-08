import { MessageSquare, Image, Globe, Code } from 'lucide-react'

interface EmptyStateProps {
  assistantName?: string
  onQuickAction: (prompt: string) => void
}

const QUICK_ACTIONS = [
  {
    icon: MessageSquare,
    label: 'Start Conversation',
    description: 'Ask anything on your mind',
    prompt: 'Hello! What can you help me with today?',
  },
  {
    icon: Image,
    label: 'Generate Image',
    description: 'Create images from text',
    prompt: 'Generate an image of ',
  },
  {
    icon: Globe,
    label: 'Web Search',
    description: 'Search the internet for info',
    prompt: 'Search the web for ',
  },
  {
    icon: Code,
    label: 'Write Code',
    description: 'Help with programming tasks',
    prompt: 'Help me write code to ',
  },
]

export function EmptyState({ assistantName = 'Jarvis', onQuickAction }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-4">
      {/* Avatar and greeting */}
      <div className="w-20 h-20 rounded-full bg-surface-2/50 flex items-center justify-center mb-5 ring-1 ring-border/30">
        <img
          src="/jarvis.jpeg"
          alt={assistantName}
          className="w-16 h-16 rounded-full object-cover"
        />
      </div>
      <h2 className="text-xl font-medium text-text mb-1">
        Welcome back
      </h2>
      <p className="text-text-muted mb-8">
        How can I help you today?
      </p>

      {/* Quick action cards */}
      <div className="grid grid-cols-2 gap-3 max-w-md w-full">
        {QUICK_ACTIONS.map(({ icon: Icon, label, description, prompt }) => (
          <button
            key={label}
            onClick={() => onQuickAction(prompt)}
            className="glass-card rounded-xl p-4 text-left transition-all hover:border-cyan-500/20 hover:bg-glass-highlight/5 group"
          >
            <Icon
              size={20}
              className="text-cyan-400 mb-2 group-hover:scale-110 transition-transform"
            />
            <p className="text-sm font-medium text-text mb-0.5">{label}</p>
            <p className="text-xs text-text-muted">{description}</p>
          </button>
        ))}
      </div>
    </div>
  )
}
