import { useState, useRef, useCallback } from 'react'

export function useVoice() {
  const [isRecording, setIsRecording] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [transcript, setTranscript] = useState('')
  const mediaRecorder = useRef<MediaRecorder | null>(null)
  const audioChunks = useRef<Blob[]>([])
  const currentAudio = useRef<HTMLAudioElement | null>(null)

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      mediaRecorder.current = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      audioChunks.current = []

      mediaRecorder.current.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.current.push(e.data)
      }

      mediaRecorder.current.start()
      setIsRecording(true)
      setTranscript('')
    } catch (err) {
      console.error('Mic access denied:', err)
    }
  }, [])

  const stopRecording = useCallback(async (): Promise<string> => {
    return new Promise((resolve) => {
      if (!mediaRecorder.current) {
        resolve('')
        return
      }

      mediaRecorder.current.onstop = async () => {
        const stream = mediaRecorder.current?.stream
        stream?.getTracks().forEach((t) => t.stop())

        const audioBlob = new Blob(audioChunks.current, { type: 'audio/webm' })
        const formData = new FormData()
        formData.append('audio', audioBlob, 'recording.webm')

        try {
          const res = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData,
          })
          const data = await res.json()
          const text = data.transcript || ''
          setTranscript(text)
          resolve(text)
        } catch (err) {
          console.error('Transcription failed:', err)
          resolve('')
        }
      }

      mediaRecorder.current.stop()
      setIsRecording(false)
    })
  }, [])

  const speak = useCallback(async (text: string) => {
    if (!text) return

    // Stop any current audio
    if (currentAudio.current) {
      currentAudio.current.pause()
      currentAudio.current = null
    }

    setIsPlaying(true)

    try {
      // Try backend TTS first
      const res = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })

      if (res.ok && res.headers.get('content-type')?.includes('audio')) {
        const blob = await res.blob()
        const url = URL.createObjectURL(blob)
        currentAudio.current = new Audio(url)
        currentAudio.current.onended = () => setIsPlaying(false)
        currentAudio.current.play()
        return
      }
    } catch (err) {
      console.log('Backend TTS failed, using browser')
    }

    // Fallback to browser TTS
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
    isRecording,
    isPlaying,
    transcript,
    startRecording,
    stopRecording,
    speak,
    stopSpeaking,
  }
}
