import { useState, useEffect } from 'react'
import { Cpu, HardDrive, MemoryStick, Wifi, WifiOff } from 'lucide-react'
import { cn } from '../../lib/utils'

interface SystemStatusResponse {
  cpu: number
  memory: number
  disk: number
  network?: {
    sent: number
    recv: number
  }
  error?: string
}

interface SystemStatusWidgetProps {
  className?: string
}

function ProgressBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="h-1.5 bg-surface-2 rounded-full overflow-hidden">
      <div
        className={cn('h-full rounded-full transition-all duration-500', color)}
        style={{ width: `${Math.min(value, 100)}%` }}
      />
    </div>
  )
}

export function SystemStatusWidget({ className = '' }: SystemStatusWidgetProps) {
  const [status, setStatus] = useState<SystemStatusResponse | null>(null)
  const [networkStatus, setNetworkStatus] = useState<'connected' | 'disconnected'>('connected')

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch('/api/status')
        if (res.ok) {
          const data = await res.json()
          if (!data.error) {
            setStatus(data)
            setNetworkStatus('connected')
          }
        } else {
          setNetworkStatus('disconnected')
        }
      } catch {
        setNetworkStatus('disconnected')
      }
    }

    fetchStatus()
    const interval = setInterval(fetchStatus, 5000)
    return () => clearInterval(interval)
  }, [])

  if (!status) {
    return (
      <div className={`animate-pulse space-y-3 ${className}`}>
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-6 bg-surface-2 rounded" />
        ))}
      </div>
    )
  }

  const getColor = (value: number) => {
    if (value > 80) return 'bg-error'
    if (value > 60) return 'bg-warning'
    return 'bg-primary'
  }

  const metrics = [
    { label: 'CPU', value: status.cpu, icon: Cpu },
    { label: 'Memory', value: status.memory, icon: MemoryStick },
    { label: 'Disk', value: status.disk, icon: HardDrive },
  ]

  return (
    <div className={`space-y-3 ${className}`}>
      {metrics.map(({ label, value, icon: Icon }) => (
        <div key={label}>
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-2 text-xs text-text-muted">
              <Icon size={12} />
              <span>{label}</span>
            </div>
            <span className="text-xs font-mono text-text">{value.toFixed(0)}%</span>
          </div>
          <ProgressBar value={value} color={getColor(value)} />
        </div>
      ))}

      <div className="flex items-center justify-between pt-2 border-t border-border/20">
        <div className="flex items-center gap-2 text-xs text-text-muted">
          {networkStatus === 'connected' ? (
            <Wifi size={12} className="text-success" />
          ) : (
            <WifiOff size={12} className="text-error" />
          )}
          <span>Network</span>
        </div>
        <span className={cn(
          'text-xs capitalize',
          networkStatus === 'connected' ? 'text-success' : 'text-error'
        )}>
          {networkStatus}
        </span>
      </div>
    </div>
  )
}
