import { useState, useRef, useEffect } from 'react'
import { useWebSocket } from './hooks/useWebSocket'
import { useVoice } from './hooks/useVoice'
import { cn } from './lib/utils'
import {
  Mic,
  MicOff,
  Send,
  Volume2,
  VolumeX,
  MessageSquare,
  Phone,
  Trash2,
  Loader2,
} from 'lucide-react'

type Mode = 'chat' | 'voice'

export default function App() {
  const [mode, setMode] = useState<Mode>('chat')
  const [chatMode, setChatMode] = useState(true)
  const [voiceOutput, setVoiceOutput] = useState(false)
  const [input, setInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const {
    connected,
    messages,
    streaming,
    isLoading,
    project,
    send,
    clear,
  } = useWebSocket()

  const {
    isRecording,
    isPlaying,
    startRecording,
    stopRecording,
    speak,
    stopSpeaking,
  } = useVoice()

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streaming])

  useEffect(() => {
    if (voiceOutput && messages.length > 0) {
      const last = messages[messages.length - 1]
      if (last.role === 'assistant') {
        speak(last.content)
      }
    }
  }, [messages, voiceOutput, speak])

  const handleSend = () => {
    if (!input.trim() || !connected) return
    send(input.trim(), chatMode)
    setInput('')
  }

  const handleVoiceToggle = async () => {
    if (isRecording) {
      const transcript = await stopRecording()
      if (transcript) {
        send(transcript, true)
      }
    } else {
      startRecording()
    }
  }

  // Voice mode - full screen voice conversation
  if (mode === 'voice') {
    return (
      <div className="flex flex-col h-full">
        <header className="flex items-center justify-between px-6 py-4 border-b border-[#2a2a3a]">
          <h1 className="text-xl font-semibold">Jarvis</h1>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setMode('chat')}
              className="p-2 rounded-lg hover:bg-[#1a1a24] text-[#71717a]"
            >
              <MessageSquare size={20} />
            </button>
            <span className={cn('w-2 h-2 rounded-full', connected ? 'bg-green-500' : 'bg-red-500')} />
          </div>
        </header>

        <div className="flex-1 flex flex-col items-center justify-center gap-8">
          <div className="text-center">
            {isRecording && <p className="text-blue-500 animate-pulse text-lg">Listening...</p>}
            {isLoading && <p className="text-[#71717a]">Thinking...</p>}
            {isPlaying && <p className="text-green-500">Speaking...</p>}
            {!isRecording && !isLoading && !isPlaying && (
              <p className="text-[#71717a]">Tap to talk</p>
            )}
          </div>

          <button
            onClick={handleVoiceToggle}
            disabled={isLoading}
            className={cn(
              'w-32 h-32 rounded-full flex items-center justify-center transition-all duration-200',
              isRecording
                ? 'bg-red-500 scale-110'
                : 'bg-blue-500 hover:scale-105 hover:bg-blue-600',
              isLoading && 'opacity-50 cursor-not-allowed'
            )}
            style={isRecording ? { animation: 'pulse 1.5s infinite' } : {}}
          >
            {isRecording ? <MicOff size={48} className="text-white" /> : <Mic size={48} className="text-white" />}
          </button>

          {streaming && (
            <div className="max-w-md text-center text-[#e4e4e7] px-4">{streaming}</div>
          )}

          {!streaming && messages.length > 0 && (
            <div className="max-w-md text-center text-[#71717a] px-4">
              {messages[messages.length - 1].content.slice(0, 200)}
              {messages[messages.length - 1].content.length > 200 && '...'}
            </div>
          )}
        </div>
      </div>
    )
  }

  // Chat mode
  return (
    <div className="flex flex-col h-full">
      <header className="flex items-center justify-between px-6 py-4 border-b border-[#2a2a3a]">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-semibold">Jarvis</h1>
          {project && (
            <span className="text-sm px-2 py-1 rounded bg-[#1a1a24] text-[#71717a]">{project}</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setChatMode(!chatMode)}
            className={cn(
              'px-3 py-1.5 rounded-lg text-sm transition-colors',
              chatMode ? 'bg-blue-500 text-white' : 'bg-[#1a1a24] text-[#71717a]'
            )}
          >
            {chatMode ? 'Chat' : 'Agent'}
          </button>

          <button
            onClick={() => { setVoiceOutput(!voiceOutput); if (voiceOutput) stopSpeaking() }}
            className={cn(
              'p-2 rounded-lg transition-colors',
              voiceOutput ? 'bg-green-500 text-white' : 'bg-[#1a1a24] text-[#71717a]'
            )}
          >
            {voiceOutput ? <Volume2 size={18} /> : <VolumeX size={18} />}
          </button>

          <button
            onClick={() => setMode('voice')}
            className="p-2 rounded-lg bg-[#1a1a24] text-[#71717a] hover:text-white"
          >
            <Phone size={18} />
          </button>

          <button onClick={clear} className="p-2 rounded-lg bg-[#1a1a24] text-[#71717a] hover:text-red-500">
            <Trash2 size={18} />
          </button>

          <span className={cn('w-2 h-2 rounded-full ml-2', connected ? 'bg-green-500' : 'bg-red-500')} />
        </div>
      </header>

      <main className="flex-1 overflow-y-auto p-4">
        <div className="max-w-3xl mx-auto space-y-4">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={cn(
                'p-4 rounded-xl',
                msg.role === 'user' ? 'bg-[#1a1a24] ml-12' :
                msg.role === 'system' ? 'bg-transparent text-center text-[#71717a] text-sm' :
                'bg-[#12121a] mr-12'
              )}
            >
              {msg.role !== 'system' && (
                <p className="text-xs text-[#71717a] mb-2">{msg.role === 'user' ? 'You' : 'Jarvis'}</p>
              )}
              <p className="whitespace-pre-wrap">{msg.content}</p>
            </div>
          ))}

          {streaming && (
            <div className="p-4 rounded-xl bg-[#12121a] mr-12">
              <p className="text-xs text-[#71717a] mb-2">Jarvis</p>
              <p className="whitespace-pre-wrap">{streaming}</p>
            </div>
          )}

          {isLoading && !streaming && (
            <div className="flex items-center gap-2 text-[#71717a]">
              <Loader2 size={16} className="animate-spin" />
              <span>Thinking...</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>

      <footer className="p-4 border-t border-[#2a2a3a]">
        <div className="max-w-3xl mx-auto flex gap-2">
          <button
            onClick={handleVoiceToggle}
            disabled={isLoading}
            className={cn(
              'p-3 rounded-xl transition-colors',
              isRecording ? 'bg-red-500 text-white' : 'bg-[#1a1a24] text-[#71717a] hover:text-white'
            )}
            style={isRecording ? { animation: 'pulse 1.5s infinite' } : {}}
          >
            {isRecording ? <MicOff size={20} /> : <Mic size={20} />}
          </button>

          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder={isRecording ? 'Listening...' : 'Type a message...'}
            disabled={isRecording}
            className="flex-1 px-4 py-3 rounded-xl bg-[#12121a] border border-[#2a2a3a] focus:border-blue-500 focus:outline-none"
          />

          <button
            onClick={handleSend}
            disabled={!input.trim() || !connected || isLoading}
            className="p-3 rounded-xl bg-blue-500 text-white disabled:opacity-50 hover:bg-blue-600"
          >
            <Send size={20} />
          </button>
        </div>
      </footer>
    </div>
  )
}
