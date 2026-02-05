import { useState } from 'react'
import {
  BrainCircuit,
  Zap,
  Code,
  Sparkles,
  ChevronDown,
  Loader2,
  CheckCircle,
  XCircle,
  Search,
} from 'lucide-react'
import { cn } from '../../lib/utils'

interface AnalysisResult {
  model_type: string
  model_name: string
  response: string
  success: boolean
}

interface AnalysisResponse {
  query: string
  profile: string
  models_used: string[]
  results: AnalysisResult[]
  synthesis?: string
  success_count: number
  total_count: number
  error?: string
}

interface Profile {
  description: string
  models: string[]
}

interface MultiModelAnalysisWidgetProps {
  className?: string
  compact?: boolean
}

const PROFILE_ICONS = {
  comprehensive: Sparkles,
  quick: Zap,
  technical: Code,
  reasoning: BrainCircuit,
}

const MODEL_TYPE_ICONS = {
  fast: Zap,
  reasoning: BrainCircuit,
  code: Code,
  thinking: Sparkles,
}

export function MultiModelAnalysisWidget({ className = '', compact = false }: MultiModelAnalysisWidgetProps) {
  const [query, setQuery] = useState('')
  const [profile, setProfile] = useState('comprehensive')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<AnalysisResponse | null>(null)
  const [expandedModel, setExpandedModel] = useState<string | null>(null)
  const [profiles, setProfiles] = useState<Record<string, Profile>>({})
  const [showProfiles, setShowProfiles] = useState(false)

  // Load profiles on first render
  useState(() => {
    fetch('/api/analyze/profiles')
      .then(res => res.json())
      .then(data => setProfiles(data.profiles || {}))
      .catch(console.error)
  })

  const runAnalysis = async () => {
    if (!query.trim()) return

    setLoading(true)
    setResult(null)

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, profile }),
      })
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setResult({
        error: 'Analysis failed',
        query,
        profile,
        models_used: [],
        results: [],
        success_count: 0,
        total_count: 0
      })
    } finally {
      setLoading(false)
    }
  }

  if (compact) {
    return (
      <div className={cn('p-4', className)}>
        <div className="flex items-center gap-2 mb-3">
          <BrainCircuit className="w-5 h-5 text-primary" />
          <span className="font-medium">Multi-Model Analysis</span>
        </div>
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter query..."
            className="flex-1 bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
            onKeyDown={(e) => e.key === 'Enter' && runAnalysis()}
          />
          <button
            onClick={runAnalysis}
            disabled={loading || !query.trim()}
            className="px-4 py-2 bg-primary text-white rounded-lg text-sm font-medium disabled:opacity-50 hover:bg-primary/90 transition-colors"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className={cn('p-6 space-y-4', className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BrainCircuit className="w-6 h-6 text-primary" />
          <h3 className="text-lg font-semibold">Multi-Model Analysis</h3>
        </div>
        <span className="text-xs text-text-muted">Powered by Chutes AI</span>
      </div>

      {/* Query Input */}
      <div className="space-y-3">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your question or topic to analyze..."
          className="w-full bg-surface-2 border border-border rounded-lg px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none"
          rows={3}
        />

        {/* Profile Selector */}
        <div className="flex items-center gap-3">
          <div className="relative">
            <button
              onClick={() => setShowProfiles(!showProfiles)}
              className="flex items-center gap-2 px-3 py-2 bg-surface-2 border border-border rounded-lg text-sm hover:bg-surface-3 transition-colors"
            >
              {(() => {
                const Icon = PROFILE_ICONS[profile as keyof typeof PROFILE_ICONS] || Sparkles
                return <Icon className="w-4 h-4" />
              })()}
              <span className="capitalize">{profile}</span>
              <ChevronDown className={cn('w-4 h-4 transition-transform', showProfiles && 'rotate-180')} />
            </button>

            {showProfiles && (
              <div className="absolute top-full left-0 mt-1 w-64 bg-surface border border-border rounded-lg shadow-lg z-10 overflow-hidden">
                {Object.entries(profiles).map(([key, p]) => {
                  const Icon = PROFILE_ICONS[key as keyof typeof PROFILE_ICONS] || Sparkles
                  return (
                    <button
                      key={key}
                      onClick={() => { setProfile(key); setShowProfiles(false) }}
                      className={cn(
                        'w-full px-4 py-3 text-left hover:bg-surface-2 transition-colors',
                        profile === key && 'bg-primary/10 text-primary'
                      )}
                    >
                      <div className="flex items-center gap-2">
                        <Icon className="w-4 h-4" />
                        <span className="capitalize font-medium">{key}</span>
                      </div>
                      <p className="text-xs text-text-muted mt-1">{p.description}</p>
                      <p className="text-xs text-text-muted/60 mt-0.5">Models: {p.models.join(', ')}</p>
                    </button>
                  )
                })}
              </div>
            )}
          </div>

          <button
            onClick={runAnalysis}
            disabled={loading || !query.trim()}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-primary text-white rounded-lg font-medium disabled:opacity-50 hover:bg-primary/90 transition-colors"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Search className="w-4 h-4" />
                Analyze
              </>
            )}
          </button>
        </div>
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-4 pt-4 border-t border-border">
          {result.error ? (
            <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400">
              {result.error}
            </div>
          ) : (
            <>
              {/* Stats */}
              <div className="flex items-center gap-4 text-sm text-text-muted">
                <span>
                  <CheckCircle className="w-4 h-4 inline mr-1 text-green-400" />
                  {result.success_count} succeeded
                </span>
                {result.total_count - result.success_count > 0 && (
                  <span>
                    <XCircle className="w-4 h-4 inline mr-1 text-red-400" />
                    {result.total_count - result.success_count} failed
                  </span>
                )}
              </div>

              {/* Model Results */}
              <div className="space-y-2">
                {result.results.map((r, i) => {
                  const Icon = MODEL_TYPE_ICONS[r.model_type as keyof typeof MODEL_TYPE_ICONS] || BrainCircuit
                  const isExpanded = expandedModel === `${i}-${r.model_type}`

                  return (
                    <div
                      key={i}
                      className={cn(
                        'border border-border rounded-lg overflow-hidden transition-all',
                        r.success ? 'bg-surface' : 'bg-red-500/5 border-red-500/30'
                      )}
                    >
                      <button
                        onClick={() => setExpandedModel(isExpanded ? null : `${i}-${r.model_type}`)}
                        className="w-full px-4 py-3 flex items-center justify-between hover:bg-surface-2 transition-colors"
                      >
                        <div className="flex items-center gap-2">
                          {r.success ? (
                            <CheckCircle className="w-4 h-4 text-green-400" />
                          ) : (
                            <XCircle className="w-4 h-4 text-red-400" />
                          )}
                          <Icon className="w-4 h-4 text-primary" />
                          <span className="font-medium capitalize">{r.model_type}</span>
                          <span className="text-xs text-text-muted">
                            ({r.model_name.split('/').pop()})
                          </span>
                        </div>
                        <ChevronDown className={cn('w-4 h-4 transition-transform', isExpanded && 'rotate-180')} />
                      </button>

                      {isExpanded && (
                        <div className="px-4 pb-4 border-t border-border">
                          <div className="pt-3 text-sm text-text whitespace-pre-wrap">
                            {r.response}
                          </div>
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>

              {/* Synthesis */}
              {result.synthesis && (
                <div className="p-4 bg-primary/10 border border-primary/30 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Sparkles className="w-5 h-5 text-primary" />
                    <h4 className="font-semibold text-primary">Synthesis</h4>
                  </div>
                  <div className="text-sm text-text whitespace-pre-wrap">
                    {result.synthesis}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
