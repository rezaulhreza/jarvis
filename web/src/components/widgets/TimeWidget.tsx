import { useState, useEffect } from 'react'

interface TimeWidgetProps {
  className?: string
  showDate?: boolean
  showSeconds?: boolean
}

export function TimeWidget({ className = '', showDate = true, showSeconds = true }: TimeWidgetProps) {
  const [time, setTime] = useState(new Date())

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  const formatTime = () => {
    const hours = time.getHours().toString().padStart(2, '0')
    const minutes = time.getMinutes().toString().padStart(2, '0')
    const seconds = time.getSeconds().toString().padStart(2, '0')
    return showSeconds ? `${hours}:${minutes}:${seconds}` : `${hours}:${minutes}`
  }

  const formatDate = () => {
    return time.toLocaleDateString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    })
  }

  return (
    <div className={`text-center ${className}`}>
      {showDate && (
        <p className="text-xs text-text-muted uppercase tracking-widest mb-1">
          {formatDate()}
        </p>
      )}
      <p className="text-4xl md:text-5xl font-mono font-light text-primary tabular-nums tracking-wider">
        {formatTime()}
      </p>
    </div>
  )
}
