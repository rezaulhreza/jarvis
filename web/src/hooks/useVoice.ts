import { useState, useRef, useCallback, useEffect } from 'react'

interface UseVoiceOptions {
  onSpeechEnd?: (transcript: string) => void
  onInterrupt?: () => void
  sttProvider?: 'browser' | 'whisper'
}

export function useVoice(options: UseVoiceOptions = {}) {
  const { onSpeechEnd, onInterrupt, sttProvider = 'browser' } = options

  const [isListening, setIsListening] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [volume, setVolume] = useState(0)

  const recognition = useRef<any>(null)
  const mediaRecorder = useRef<MediaRecorder | null>(null)
  const audioChunks = useRef<Blob[]>([])
  const currentAudio = useRef<HTMLAudioElement | null>(null)
  const audioContext = useRef<AudioContext | null>(null)
  const analyser = useRef<AnalyserNode | null>(null)
  const animationFrame = useRef<number | null>(null)
  const stream = useRef<MediaStream | null>(null)
  const transcriptRef = useRef('')
  const isListeningRef = useRef(false)
  const onSpeechEndRef = useRef(onSpeechEnd)
  const onInterruptRef = useRef(onInterrupt)
  const sttProviderRef = useRef(sttProvider)
  const silenceTimer = useRef<NodeJS.Timeout | null>(null)

  // Keep refs in sync
  useEffect(() => {
    onSpeechEndRef.current = onSpeechEnd
    onInterruptRef.current = onInterrupt
    sttProviderRef.current = sttProvider
  }, [onSpeechEnd, onInterrupt, sttProvider])

  useEffect(() => {
    isListeningRef.current = isListening
  }, [isListening])

  // Initialize browser speech recognition
  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SpeechRecognition) {
      console.warn('Speech recognition not supported')
      return
    }

    const recog = new SpeechRecognition()
    recog.continuous = true
    recog.interimResults = true
    recog.lang = 'en-US'

    recog.onresult = (event: any) => {
      let finalTranscript = ''

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i]
        if (result.isFinal) {
          finalTranscript += result[0].transcript
        }
      }

      if (finalTranscript) {
        transcriptRef.current = finalTranscript.trim()
        if (transcriptRef.current && onSpeechEndRef.current) {
          onSpeechEndRef.current(transcriptRef.current)
          transcriptRef.current = ''
        }
      }

      setVolume(0.5)
      setIsRecording(true)
    }

    recog.onspeechend = () => {
      setVolume(0)
      setIsRecording(false)
    }

    recog.onerror = (event: any) => {
      if (event.error !== 'no-speech' && event.error !== 'aborted') {
        console.error('Speech recognition error:', event.error)
      }
    }

    recog.onend = () => {
      if (isListeningRef.current && sttProviderRef.current === 'browser') {
        try {
          recog.start()
        } catch (e) {}
      }
    }

    recognition.current = recog

    return () => {
      try {
        recog.stop()
      } catch (e) {}
    }
  }, [])

  // Process audio with Whisper
  const processWithWhisper = useCallback(async (audioBlob: Blob) => {
    try {
      const formData = new FormData()
      formData.append('audio', audioBlob, 'recording.webm')

      const res = await fetch('/api/transcribe', {
        method: 'POST',
        body: formData,
      })

      if (res.ok) {
        const data = await res.json()
        const text = data.transcript || data.text
        if (text && onSpeechEndRef.current) {
          onSpeechEndRef.current(text)
        }
      }
    } catch (err) {
      console.error('Whisper transcription failed:', err)
    }
  }, [])

  const startListening = useCallback(async () => {
    if (isListeningRef.current) return

    try {
      stream.current = await navigator.mediaDevices.getUserMedia({ audio: true })
      audioContext.current = new AudioContext()
      analyser.current = audioContext.current.createAnalyser()
      analyser.current.fftSize = 512
      const source = audioContext.current.createMediaStreamSource(stream.current)
      source.connect(analyser.current)

      setIsListening(true)
      transcriptRef.current = ''

      if (sttProviderRef.current === 'browser' && recognition.current) {
        // Use browser speech recognition
        recognition.current.start()
      } else {
        // Use Whisper - set up MediaRecorder
        audioChunks.current = []
        mediaRecorder.current = new MediaRecorder(stream.current, {
          mimeType: 'audio/webm;codecs=opus'
        })

        mediaRecorder.current.ondataavailable = (e) => {
          if (e.data.size > 0) {
            audioChunks.current.push(e.data)
          }
        }

        mediaRecorder.current.onstop = async () => {
          if (audioChunks.current.length > 0) {
            const audioBlob = new Blob(audioChunks.current, { type: 'audio/webm' })
            await processWithWhisper(audioBlob)
          }
        }

        mediaRecorder.current.start(100) // Collect data every 100ms
      }

      // Volume monitoring and interrupt detection
      const checkVolume = () => {
        if (!analyser.current) return
        const dataArray = new Uint8Array(analyser.current.frequencyBinCount)
        analyser.current.getByteFrequencyData(dataArray)
        const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length / 255
        setVolume(avg)

        // Detect speech for Whisper mode
        if (sttProviderRef.current === 'whisper') {
          if (avg > 0.02) {
            setIsRecording(true)
            // Reset silence timer
            if (silenceTimer.current) {
              clearTimeout(silenceTimer.current)
            }
            silenceTimer.current = setTimeout(() => {
              // Silence detected - process audio
              if (mediaRecorder.current?.state === 'recording') {
                mediaRecorder.current.stop()
                setIsRecording(false)
                // Restart recording for continuous listening
                setTimeout(() => {
                  if (isListeningRef.current && stream.current) {
                    audioChunks.current = []
                    mediaRecorder.current = new MediaRecorder(stream.current, {
                      mimeType: 'audio/webm;codecs=opus'
                    })
                    mediaRecorder.current.ondataavailable = (e) => {
                      if (e.data.size > 0) {
                        audioChunks.current.push(e.data)
                      }
                    }
                    mediaRecorder.current.onstop = async () => {
                      if (audioChunks.current.length > 0) {
                        const audioBlob = new Blob(audioChunks.current, { type: 'audio/webm' })
                        await processWithWhisper(audioBlob)
                      }
                    }
                    mediaRecorder.current.start(100)
                  }
                }, 100)
              }
            }, 1500) // 1.5s silence = end of speech
          }
        }

        // Interrupt detection
        if (avg > 0.03 && currentAudio.current && !currentAudio.current.paused) {
          currentAudio.current.pause()
          currentAudio.current = null
          setIsPlaying(false)
          onInterruptRef.current?.()
        }

        animationFrame.current = requestAnimationFrame(checkVolume)
      }
      checkVolume()
    } catch (err) {
      console.error('Mic access denied:', err)
    }
  }, [processWithWhisper])

  const stopListening = useCallback(() => {
    setIsListening(false)
    setIsRecording(false)
    setVolume(0)

    if (silenceTimer.current) {
      clearTimeout(silenceTimer.current)
      silenceTimer.current = null
    }

    if (animationFrame.current) {
      cancelAnimationFrame(animationFrame.current)
      animationFrame.current = null
    }

    if (recognition.current) {
      try {
        recognition.current.stop()
      } catch (e) {}
    }

    if (mediaRecorder.current?.state === 'recording') {
      mediaRecorder.current.stop()
    }

    if (stream.current) {
      stream.current.getTracks().forEach(t => t.stop())
      stream.current = null
    }

    if (audioContext.current) {
      audioContext.current.close()
      audioContext.current = null
    }
  }, [])

  const startRecording = useCallback(async () => {
    await startListening()
  }, [startListening])

  const stopRecording = useCallback(async (): Promise<string> => {
    return new Promise((resolve) => {
      setTimeout(() => {
        const result = transcriptRef.current
        transcriptRef.current = ''
        stopListening()
        resolve(result)
      }, 500)
    })
  }, [stopListening])

  const speak = useCallback(async (text: string, provider: 'browser' | 'edge' | 'elevenlabs' = 'browser') => {
    if (!text) return

    if (currentAudio.current) {
      currentAudio.current.pause()
      currentAudio.current = null
    }
    speechSynthesis.cancel()

    setIsPlaying(true)

    // Browser TTS - instant, no network
    if (provider === 'browser') {
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.rate = 1.1
      utterance.onend = () => setIsPlaying(false)
      utterance.onerror = () => setIsPlaying(false)
      speechSynthesis.speak(utterance)
      return
    }

    // ElevenLabs - streaming, low latency, best quality
    if (provider === 'elevenlabs') {
      try {
        const res = await fetch('/api/tts/elevenlabs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        })

        if (res.ok && res.headers.get('content-type')?.includes('audio')) {
          const blob = await res.blob()
          const url = URL.createObjectURL(blob)
          currentAudio.current = new Audio(url)
          currentAudio.current.onended = () => {
            setIsPlaying(false)
            URL.revokeObjectURL(url)
          }
          currentAudio.current.onerror = () => {
            setIsPlaying(false)
            URL.revokeObjectURL(url)
          }
          await currentAudio.current.play()
          return
        }
      } catch (err) {
        console.log('ElevenLabs TTS failed, falling back to browser')
      }
      // Fallback to browser
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.onend = () => setIsPlaying(false)
      speechSynthesis.speak(utterance)
      return
    }

    // Edge TTS - neural voices, free
    try {
      const res = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })

      if (res.ok && res.headers.get('content-type')?.includes('audio')) {
        const blob = await res.blob()
        const url = URL.createObjectURL(blob)
        currentAudio.current = new Audio(url)
        currentAudio.current.onended = () => {
          setIsPlaying(false)
          URL.revokeObjectURL(url)
        }
        currentAudio.current.onerror = () => {
          setIsPlaying(false)
          URL.revokeObjectURL(url)
        }
        await currentAudio.current.play()
        return
      }
    } catch (err) {
      console.log('Edge TTS failed')
    }

    // Fallback to browser
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.onend = () => setIsPlaying(false)
    speechSynthesis.speak(utterance)
  }, [])

  const stopSpeaking = useCallback(() => {
    if (currentAudio.current) {
      currentAudio.current.pause()
      currentAudio.current = null
    }
    speechSynthesis.cancel()
    setIsPlaying(false)
  }, [])

  return {
    isListening,
    isRecording,
    isPlaying,
    volume,
    startListening,
    stopListening,
    startRecording,
    stopRecording,
    speak,
    stopSpeaking,
  }
}
