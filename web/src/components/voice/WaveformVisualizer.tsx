import { useRef, useEffect } from 'react'
import { cn } from '../../lib/utils'

interface WaveformVisualizerProps {
  getFrequencyData?: () => Uint8Array | null
  isActive: boolean
  color?: 'purple' | 'green' | 'cyan'
  barCount?: number
  height?: number
  volume?: number
}

const COLOR_MAP = {
  purple: 'bg-purple-500',
  green: 'bg-emerald-500',
  cyan: 'bg-cyan-500',
}

const GLOW_MAP = {
  purple: 'shadow-[0_0_6px_rgba(168,85,247,0.5)]',
  green: 'shadow-[0_0_6px_rgba(34,197,94,0.5)]',
  cyan: 'shadow-[0_0_6px_rgba(6,182,212,0.5)]',
}

export function WaveformVisualizer({
  getFrequencyData,
  isActive,
  color = 'purple',
  barCount = 32,
  height = 64,
  volume = 0,
}: WaveformVisualizerProps) {
  const barsRef = useRef<(HTMLDivElement | null)[]>([])
  const animFrameRef = useRef<number | null>(null)
  const getFrequencyDataRef = useRef(getFrequencyData)
  const volumeRef = useRef(volume)

  // Keep refs in sync
  useEffect(() => {
    getFrequencyDataRef.current = getFrequencyData
    volumeRef.current = volume
  })

  useEffect(() => {
    if (!isActive) return

    const loop = () => {
      const data = getFrequencyDataRef.current?.()

      if (data && data.length > 0) {
        const step = Math.floor(data.length / barCount)
        for (let i = 0; i < barCount; i++) {
          const bar = barsRef.current[i]
          if (!bar) continue
          const idx = Math.min(i * step, data.length - 1)
          const value = data[idx] / 255
          bar.style.transform = `scaleY(${Math.max(0.08, value)})`
        }
      } else {
        const time = Date.now() / 1000
        for (let i = 0; i < barCount; i++) {
          const bar = barsRef.current[i]
          if (!bar) continue
          const sine = Math.sin(time * 3 + i * 0.4) * 0.5 + 0.5
          bar.style.transform = `scaleY(${Math.max(0.08, sine * Math.max(volumeRef.current, 0.1))})`
        }
      }

      animFrameRef.current = requestAnimationFrame(loop)
    }

    animFrameRef.current = requestAnimationFrame(loop)

    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current)
        animFrameRef.current = null
      }
    }
  }, [isActive, barCount])

  // Reset bars when not active
  useEffect(() => {
    if (!isActive) {
      barsRef.current.forEach(bar => {
        if (bar) bar.style.transform = 'scaleY(0.08)'
      })
    }
  }, [isActive])

  return (
    <div
      className="flex items-end justify-center gap-[2px]"
      style={{ height }}
    >
      {Array.from({ length: barCount }).map((_, i) => (
        <div
          key={i}
          ref={el => { barsRef.current[i] = el }}
          className={cn(
            'w-[3px] rounded-full origin-bottom transition-transform duration-75',
            COLOR_MAP[color],
            isActive && GLOW_MAP[color],
          )}
          style={{
            height: '100%',
            transform: 'scaleY(0.08)',
            opacity: isActive ? 0.9 : 0.3,
          }}
        />
      ))}
    </div>
  )
}
