import { useEffect, useRef, useCallback, useState } from 'react'

// Type for SpeechRecognition (not in standard TypeScript lib)
interface SpeechRecognitionResult {
  transcript: string
  confidence: number
}

interface SpeechRecognitionResultList {
  length: number
  item(index: number): SpeechRecognitionResult[]
  [index: number]: SpeechRecognitionResult[]
}

interface SpeechRecognitionEvent {
  results: SpeechRecognitionResultList
}

interface SpeechRecognitionErrorEvent {
  error: string
}

interface SpeechRecognitionInstance {
  continuous: boolean
  interimResults: boolean
  lang: string
  onstart: (() => void) | null
  onresult: ((event: SpeechRecognitionEvent) => void) | null
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null
  onend: (() => void) | null
  start: () => void
  abort: () => void
}

interface UseWakeWordOptions {
  keyword?: string
  onWakeWord: () => void
  enabled?: boolean
}

/**
 * Wake word detection hook - listens for "Jarvis" or custom keyword
 * Uses continuous browser speech recognition to detect the wake word
 */
export function useWakeWord({
  keyword = 'jarvis',
  onWakeWord,
  enabled = true,
}: UseWakeWordOptions) {
  const [isListening, setIsListening] = useState(false)
  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null)
  const onWakeWordRef = useRef(onWakeWord)
  const cooldownRef = useRef(false)
  const enabledRef = useRef(enabled)

  // Keep refs in sync
  useEffect(() => {
    onWakeWordRef.current = onWakeWord
    enabledRef.current = enabled
  }, [onWakeWord, enabled])

  const stopWakeWordDetection = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.onend = null
      recognitionRef.current.abort()
      recognitionRef.current = null
    }
    // Note: setIsListening is handled by the onend callback
  }, [])

  const startWakeWordDetection = useCallback(() => {
    if (!enabledRef.current || recognitionRef.current) return

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SpeechRecognition) {
      console.warn('Wake word detection not supported - no SpeechRecognition API')
      return
    }

    const recognition = new SpeechRecognition() as SpeechRecognitionInstance
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = 'en-US'

    recognition.onstart = () => {
      setIsListening(true)
    }

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      // Check only the most recent result
      const lastResult = event.results[event.results.length - 1]
      const transcript = lastResult[0].transcript.toLowerCase().trim()

      // Check for wake word patterns
      const wakePatterns = [
        keyword.toLowerCase(),
        `hey ${keyword.toLowerCase()}`,
        `hi ${keyword.toLowerCase()}`,
        `ok ${keyword.toLowerCase()}`,
        `okay ${keyword.toLowerCase()}`,
      ]

      const detected = wakePatterns.some(pattern => transcript.includes(pattern))

      if (detected && !cooldownRef.current) {
        console.log('[WakeWord] Detected:', transcript)
        cooldownRef.current = true

        // Trigger callback
        onWakeWordRef.current()

        // Cooldown to prevent rapid re-triggers
        setTimeout(() => {
          cooldownRef.current = false
        }, 2000)
      }
    }

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      if (event.error !== 'no-speech' && event.error !== 'aborted') {
        console.warn('[WakeWord] Error:', event.error)
      }
    }

    recognition.onend = () => {
      setIsListening(false)
      // Auto-restart if still enabled
      if (enabledRef.current && recognitionRef.current) {
        setTimeout(() => {
          try {
            recognitionRef.current?.start()
          } catch {
            // Ignore - already started
          }
        }, 100)
      }
    }

    recognitionRef.current = recognition
    try {
      recognition.start()
    } catch {
      console.warn('[WakeWord] Failed to start')
    }
  }, [keyword])

  // Handle enabled changes - use refs to avoid setState in effect
  useEffect(() => {
    if (enabled) {
      startWakeWordDetection()
    } else if (recognitionRef.current) {
      // Stop without calling the wrapped function to avoid setState
      recognitionRef.current.onend = null
      recognitionRef.current.abort()
      recognitionRef.current = null
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.onend = null
        recognitionRef.current.abort()
        recognitionRef.current = null
      }
    }
  }, [enabled, startWakeWordDetection])

  return {
    isListening,
    start: startWakeWordDetection,
    stop: stopWakeWordDetection,
  }
}
