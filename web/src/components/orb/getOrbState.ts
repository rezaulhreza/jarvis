import type { OrbState } from '../../types'

// Helper function to determine orb state from component state
export function getOrbState(props: {
  isPlaying: boolean
  isLoading: boolean
  isListening: boolean
  isRecording: boolean
}): OrbState {
  const { isPlaying, isLoading, isListening, isRecording } = props

  if (isPlaying) return 'speaking'
  if (isLoading) return 'thinking'
  if (isListening || isRecording) return 'listening'
  return 'idle'
}
