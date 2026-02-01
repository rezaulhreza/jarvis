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
  const [playbackVolume, setPlaybackVolume] = useState(0)

  const recognition = useRef<any>(null)
  const mediaRecorder = useRef<MediaRecorder | null>(null)
  const audioChunks = useRef<Blob[]>([])
  const currentAudio = useRef<HTMLAudioElement | null>(null)
  const audioContext = useRef<AudioContext | null>(null)
  const playbackContext = useRef<AudioContext | null>(null)
  const playbackAnalyser = useRef<AnalyserNode | null>(null)
  const playbackAnimFrame = useRef<number | null>(null)
  const analyser = useRef<AnalyserNode | null>(null)
  const animationFrame = useRef<number | null>(null)
  const stream = useRef<MediaStream | null>(null)
  const transcriptRef = useRef('')
  const isListeningRef = useRef(false)
  const isPlayingRef = useRef(false) // Track if TTS is playing (to ignore mic input)
  const onSpeechEndRef = useRef(onSpeechEnd)
  const onInterruptRef = useRef(onInterrupt)
  const sttProviderRef = useRef(sttProvider)
  const silenceTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const speakingRef = useRef(false)

  // Keep refs in sync
  useEffect(() => {
    onSpeechEndRef.current = onSpeechEnd
    onInterruptRef.current = onInterrupt
    sttProviderRef.current = sttProvider
  }, [onSpeechEnd, onInterrupt, sttProvider])

  useEffect(() => {
    isListeningRef.current = isListening
  }, [isListening])

  useEffect(() => {
    isPlayingRef.current = isPlaying
  }, [isPlaying])

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
      // IGNORE input while TTS is playing (prevents feedback loop)
      if (isPlayingRef.current) {
        return
      }

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
      // Only restart if we're supposed to be listening AND not playing audio
      if (isListeningRef.current && sttProviderRef.current === 'browser' && !isPlayingRef.current) {
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
    // Don't process if audio is playing (feedback prevention)
    if (isPlayingRef.current) return

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
        if (text && onSpeechEndRef.current && !isPlayingRef.current) {
          onSpeechEndRef.current(text)
        }
      }
    } catch (err) {
      console.error('Whisper transcription failed:', err)
    }
  }, [])

  const startListening = useCallback(async () => {
    if (isListeningRef.current) return
    // Don't start listening if audio is playing
    if (isPlayingRef.current) return

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
        recognition.current.start()
      } else {
        // Whisper mode - set up MediaRecorder
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
          if (audioChunks.current.length > 0 && !isPlayingRef.current) {
            const audioBlob = new Blob(audioChunks.current, { type: 'audio/webm' })
            await processWithWhisper(audioBlob)
          }
        }

        mediaRecorder.current.start(100)
      }

      // Volume monitoring
      const checkVolume = () => {
        if (!analyser.current) return
        const dataArray = new Uint8Array(analyser.current.frequencyBinCount)
        analyser.current.getByteFrequencyData(dataArray)
        const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length / 255

        // Only update volume if not playing (prevents feedback detection)
        if (!isPlayingRef.current) {
          setVolume(avg)
        }

        // Whisper mode silence detection - 3 seconds of silence
        if (sttProviderRef.current === 'whisper' && !isPlayingRef.current) {
          if (avg > 0.02) {
            setIsRecording(true)
            if (silenceTimer.current) {
              clearTimeout(silenceTimer.current)
            }
            silenceTimer.current = setTimeout(() => {
              if (mediaRecorder.current?.state === 'recording' && !isPlayingRef.current) {
                mediaRecorder.current.stop()
                setIsRecording(false)
                // Restart recording after processing
                setTimeout(() => {
                  if (isListeningRef.current && stream.current && !isPlayingRef.current) {
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
                      if (audioChunks.current.length > 0 && !isPlayingRef.current) {
                        const audioBlob = new Blob(audioChunks.current, { type: 'audio/webm' })
                        await processWithWhisper(audioBlob)
                      }
                    }
                    mediaRecorder.current.start(100)
                  }
                }, 100)
              }
            }, 3000) // 3 seconds silence before processing
          }
        }

        // Interrupt detection - only if user is LOUDLY speaking (not just ambient noise)
        if (avg > 0.1 && currentAudio.current && !currentAudio.current.paused) {
          currentAudio.current.pause()
          currentAudio.current = null
          setIsPlaying(false)
          isPlayingRef.current = false
          speakingRef.current = false
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

  // Stop playback volume animation
  const stopPlaybackAnalysis = useCallback(() => {
    if (playbackAnimFrame.current) {
      cancelAnimationFrame(playbackAnimFrame.current)
      playbackAnimFrame.current = null
    }
    if (playbackContext.current) {
      playbackContext.current.close().catch(() => {})
      playbackContext.current = null
      playbackAnalyser.current = null
    }
    setPlaybackVolume(0)
  }, [])

  // Analyze audio element output for visualization
  const startPlaybackAnalysis = useCallback((audioElement: HTMLAudioElement) => {
    try {
      playbackContext.current = new AudioContext()
      playbackAnalyser.current = playbackContext.current.createAnalyser()
      playbackAnalyser.current.fftSize = 256

      const source = playbackContext.current.createMediaElementSource(audioElement)
      source.connect(playbackAnalyser.current)
      playbackAnalyser.current.connect(playbackContext.current.destination)

      const analyzePlayback = () => {
        if (!playbackAnalyser.current || !isPlayingRef.current) {
          setPlaybackVolume(0)
          return
        }
        const dataArray = new Uint8Array(playbackAnalyser.current.frequencyBinCount)
        playbackAnalyser.current.getByteFrequencyData(dataArray)
        const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length / 255
        setPlaybackVolume(avg)
        playbackAnimFrame.current = requestAnimationFrame(analyzePlayback)
      }
      analyzePlayback()
    } catch (e) {
      // Fallback: simulate volume with a pulse
      const pulse = () => {
        if (!isPlayingRef.current) {
          setPlaybackVolume(0)
          return
        }
        setPlaybackVolume(0.3 + Math.random() * 0.4)
        playbackAnimFrame.current = requestAnimationFrame(pulse)
      }
      pulse()
    }
  }, [])

  // Stop any current audio/speech
  const stopCurrentAudio = useCallback(() => {
    stopPlaybackAnalysis()
    if (currentAudio.current) {
      currentAudio.current.pause()
      currentAudio.current.onended = null
      currentAudio.current.onerror = null
      currentAudio.current = null
    }
    speechSynthesis.cancel()
    setIsPlaying(false)
    isPlayingRef.current = false
    speakingRef.current = false
  }, [stopPlaybackAnalysis])

  // Pause recognition while speaking
  const pauseRecognition = useCallback(() => {
    if (recognition.current && sttProviderRef.current === 'browser') {
      try {
        recognition.current.stop()
      } catch (e) {}
    }
    if (mediaRecorder.current?.state === 'recording') {
      try {
        mediaRecorder.current.pause()
      } catch (e) {}
    }
  }, [])

  // Resume recognition after speaking
  const resumeRecognition = useCallback(() => {
    if (!isListeningRef.current) return

    if (sttProviderRef.current === 'browser' && recognition.current) {
      try {
        recognition.current.start()
      } catch (e) {}
    }
    if (mediaRecorder.current?.state === 'paused') {
      try {
        mediaRecorder.current.resume()
      } catch (e) {}
    }
  }, [])

  const speak = useCallback(async (text: string, provider: 'browser' | 'edge' | 'elevenlabs' = 'browser') => {
    if (!text) return

    // Stop any current audio first
    stopCurrentAudio()

    // Prevent concurrent speak calls
    if (speakingRef.current) {
      return
    }

    speakingRef.current = true
    setIsPlaying(true)
    isPlayingRef.current = true

    // Pause mic input while speaking
    pauseRecognition()

    const onFinished = () => {
      stopPlaybackAnalysis()
      setIsPlaying(false)
      isPlayingRef.current = false
      speakingRef.current = false
      // Resume listening after a short delay
      setTimeout(() => {
        if (isListeningRef.current) {
          resumeRecognition()
        }
      }, 300)
    }

    // Browser TTS - simulate volume pulse
    if (provider === 'browser') {
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.rate = 1.1
      utterance.onend = onFinished
      utterance.onerror = onFinished
      // Simulate volume animation for browser TTS
      const simulatePulse = () => {
        if (!isPlayingRef.current) {
          setPlaybackVolume(0)
          return
        }
        setPlaybackVolume(0.3 + Math.random() * 0.4)
        playbackAnimFrame.current = requestAnimationFrame(simulatePulse)
      }
      simulatePulse()
      speechSynthesis.speak(utterance)
      return
    }

    // ElevenLabs TTS
    if (provider === 'elevenlabs') {
      try {
        const res = await fetch('/api/tts/elevenlabs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        })

        const contentType = res.headers.get('content-type') || ''
        if (res.ok && contentType.includes('audio')) {
          const blob = await res.blob()
          const url = URL.createObjectURL(blob)
          currentAudio.current = new Audio(url)
          currentAudio.current.crossOrigin = 'anonymous'
          currentAudio.current.onended = () => {
            URL.revokeObjectURL(url)
            onFinished()
          }
          currentAudio.current.onerror = () => {
            URL.revokeObjectURL(url)
            onFinished()
          }
          startPlaybackAnalysis(currentAudio.current)
          await currentAudio.current.play()
          return
        } else {
          const data = await res.json().catch(() => ({}))
          console.error('ElevenLabs error:', data.error || 'Unknown error')
          onFinished()
          return
        }
      } catch (err) {
        console.error('ElevenLabs network error:', err)
        onFinished()
        return
      }
    }

    // Edge TTS
    if (provider === 'edge') {
      try {
        const res = await fetch('/api/tts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        })

        const contentType = res.headers.get('content-type') || ''
        if (res.ok && contentType.includes('audio')) {
          const blob = await res.blob()
          const url = URL.createObjectURL(blob)
          currentAudio.current = new Audio(url)
          currentAudio.current.crossOrigin = 'anonymous'
          currentAudio.current.onended = () => {
            URL.revokeObjectURL(url)
            onFinished()
          }
          currentAudio.current.onerror = () => {
            URL.revokeObjectURL(url)
            onFinished()
          }
          startPlaybackAnalysis(currentAudio.current)
          await currentAudio.current.play()
          return
        }
      } catch (err) {
        console.error('Edge TTS error:', err)
      }
      onFinished()
    }
  }, [stopCurrentAudio, pauseRecognition, resumeRecognition, startPlaybackAnalysis, stopPlaybackAnalysis])

  const stopSpeaking = useCallback(() => {
    stopCurrentAudio()
  }, [stopCurrentAudio])

  return {
    isListening,
    isRecording,
    isPlaying,
    volume,
    playbackVolume,
    startListening,
    stopListening,
    startRecording,
    stopRecording,
    speak,
    stopSpeaking,
  }
}
