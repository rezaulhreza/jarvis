import { useState, useEffect } from 'react'
import { Cloud, Sun, CloudRain, CloudSnow, Wind, Droplets, CloudFog, Zap, MapPin, CloudSun } from 'lucide-react'
import { cn } from '../../lib/utils'

interface WeatherResponse {
  location: string
  temperature: number
  condition: string
  humidity?: number
  wind?: number
  unit: string
  error?: string
}

interface WeatherWidgetProps {
  className?: string
  compact?: boolean
}

function getWeatherIconAndColor(condition: string): { icon: typeof Sun; color: string } {
  const lower = condition.toLowerCase()
  if (lower.includes('clear') || lower.includes('sunny')) {
    return { icon: Sun, color: 'text-yellow-400' }
  }
  if (lower.includes('partly')) {
    return { icon: CloudSun, color: 'text-yellow-300' }
  }
  if (lower.includes('rain') || lower.includes('drizzle') || lower.includes('shower')) {
    return { icon: CloudRain, color: 'text-blue-400' }
  }
  if (lower.includes('snow')) {
    return { icon: CloudSnow, color: 'text-slate-300' }
  }
  if (lower.includes('fog') || lower.includes('mist')) {
    return { icon: CloudFog, color: 'text-slate-400' }
  }
  if (lower.includes('thunder') || lower.includes('storm')) {
    return { icon: Zap, color: 'text-purple-400' }
  }
  if (lower.includes('cloud') || lower.includes('overcast')) {
    return { icon: Cloud, color: 'text-slate-400' }
  }
  return { icon: Cloud, color: 'text-slate-400' }
}

export function WeatherWidget({ className = '', compact = false }: WeatherWidgetProps) {
  const [weather, setWeather] = useState<WeatherResponse | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchWeather = async () => {
      try {
        const res = await fetch('/api/weather')
        if (res.ok) {
          const data = await res.json()
          if (!data.error) {
            setWeather(data)
          }
        }
      } catch (err) {
        console.error('Failed to fetch weather:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchWeather()
    const interval = setInterval(fetchWeather, 10 * 60 * 1000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className={`animate-pulse ${className}`}>
        <div className="h-8 w-24 bg-surface-2 rounded mb-2" />
        <div className="h-4 w-16 bg-surface-2 rounded" />
      </div>
    )
  }

  if (!weather) {
    return (
      <div className={`text-text-muted text-sm ${className}`}>
        Weather unavailable
      </div>
    )
  }

  const { icon: WeatherIcon, color: iconColor } = getWeatherIconAndColor(weather.condition)

  if (compact) {
    return (
      <div className={`flex items-center gap-3 ${className}`}>
        <WeatherIcon className={cn('w-8 h-8', iconColor)} />
        <div>
          <p className="text-2xl font-light">
            {weather.temperature}°{weather.unit}
          </p>
          <p className="text-xs text-text-muted capitalize">{weather.condition}</p>
        </div>
      </div>
    )
  }

  return (
    <div className={className}>
      <div className="flex items-start justify-between mb-4">
        <div>
          <p className="text-4xl font-light text-text">
            {weather.temperature}°{weather.unit}
          </p>
          <p className="text-sm text-text-muted capitalize">{weather.condition}</p>
          <p className="text-xs text-text-muted/60 flex items-center gap-1">
            <MapPin size={10} />
            {weather.location}
          </p>
        </div>
        <WeatherIcon className={cn('w-12 h-12', iconColor)} />
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs">
        {weather.humidity !== undefined && (
          <div className="flex items-center gap-1 text-text-muted">
            <Droplets size={12} />
            <span>{weather.humidity}%</span>
          </div>
        )}
        {weather.wind !== undefined && (
          <div className="flex items-center gap-1 text-text-muted">
            <Wind size={12} />
            <span>{weather.wind} km/h</span>
          </div>
        )}
      </div>
    </div>
  )
}
