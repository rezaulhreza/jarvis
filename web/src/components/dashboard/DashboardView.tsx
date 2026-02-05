import { useState, useEffect } from 'react'
import { cn } from '../../lib/utils'
import { WidgetCard } from './WidgetCard'
import {
  TimeWidget,
  WeatherWidget,
  QuickCommandsWidget,
  MultiModelAnalysisWidget,
} from '../widgets'
import { MessageList } from '../chat'
import { UnifiedInput } from '../input'
import { Orb, getOrbState } from '../orb'
import {
  Settings,
  Home,
  MessageSquare,
  Calendar,
  Heart,
  Zap,
  Scale,
  Brain,
  Plus,
  ChevronRight,
  BrainCircuit,
} from 'lucide-react'
import type { Message, OrbState, UploadedFile, ReasoningLevel } from '../../types'
import type { AssistantConfig } from '../../types/dashboard'

// Calendar Event type
interface CalendarEvent {
  id: string
  title: string
  start: string
  end?: string
  allDay?: boolean
  location?: string
  description?: string
  source?: 'google' | 'apple'
}

interface DashboardViewProps {
  // Assistant config
  assistantConfig: AssistantConfig
  connected: boolean

  // Messages
  messages: Message[]
  streaming: string
  isLoading: boolean
  loadingText: string

  // Voice state
  isListening: boolean
  isRecording: boolean
  isPlaying: boolean
  volume: number
  playbackVolume: number
  isMuted: boolean
  wakeWordEnabled: boolean

  // Input state
  input: string
  files: UploadedFile[]

  // Reasoning level
  reasoningLevel: ReasoningLevel
  onReasoningLevelChange: (level: ReasoningLevel) => void

  // Handlers
  onInputChange: (value: string) => void
  onSend: () => void
  onVoiceToggle: () => void
  onFilesAdded: (files: FileList | File[]) => void
  onRemoveFile: (id: string) => void
  onOrbClick: () => void
  onToggleMute: () => void
  onOpenSettings: () => void
  onQuickCommand: (command: string) => void

  // View mode
  viewMode: 'desktop' | 'mobile'
}

export function DashboardView({
  assistantConfig,
  connected,
  messages,
  streaming,
  isLoading,
  loadingText,
  isListening,
  isRecording,
  isPlaying,
  volume,
  playbackVolume,
  isMuted,
  wakeWordEnabled,
  input,
  files,
  reasoningLevel,
  onReasoningLevelChange,
  onInputChange,
  onSend,
  onVoiceToggle,
  onFilesAdded,
  onRemoveFile,
  onOrbClick,
  onToggleMute,
  onOpenSettings,
  onQuickCommand,
  viewMode,
}: DashboardViewProps) {
  const orbState = getOrbState({ isPlaying, isLoading, isListening, isRecording })
  const assistantName = assistantConfig.name || 'Jarvis'

  // Calendar state for desktop
  const [calendarEvents, setCalendarEvents] = useState<CalendarEvent[]>([])
  const [calendarConnected, setCalendarConnected] = useState(false)

  // Fetch calendar events
  useEffect(() => {
    const fetchCalendar = async () => {
      try {
        const statusRes = await fetch('/api/calendar/status')
        if (statusRes.ok) {
          const status = await statusRes.json()
          setCalendarConnected(status.connected)

          if (status.connected) {
            const eventsRes = await fetch('/api/calendar/events?days=7')
            if (eventsRes.ok) {
              const data = await eventsRes.json()
              setCalendarEvents(data.events || [])
            }
          }
        }
      } catch (err) {
        console.error('Failed to fetch calendar:', err)
      }
    }
    fetchCalendar()
  }, [])

  const getTodayEvents = () => {
    const today = new Date().toDateString()
    return calendarEvents.filter(e => new Date(e.start).toDateString() === today)
  }

  const formatEventTime = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
  }

  // Mobile view
  if (viewMode === 'mobile') {
    return <MobileDashboard {...{
      assistantConfig,
      connected,
      messages,
      streaming,
      isLoading,
      loadingText,
      orbState,
      volume,
      playbackVolume,
      isMuted,
      wakeWordEnabled,
      input,
      files,
      reasoningLevel,
      onReasoningLevelChange,
      onInputChange,
      onSend,
      onVoiceToggle,
      onFilesAdded,
      onRemoveFile,
      onOrbClick,
      onToggleMute,
      onOpenSettings,
      onQuickCommand,
    }} />
  }

  // Desktop HUD view - Clean, aligned with mobile feel
  return (
    <div className="h-full flex flex-col bg-background overflow-hidden">
      {/* Top Bar */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-primary/10">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3">
            <div className={cn(
              'w-2.5 h-2.5 rounded-full',
              connected ? 'bg-success animate-pulse' : 'bg-error'
            )} />
            <h1 className="text-lg font-medium text-text">
              {assistantName}
            </h1>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <WeatherWidget compact className="opacity-70 hover:opacity-100 transition-opacity" />
          <div className="w-px h-6 bg-border/30" />
          <TimeWidget showDate={false} showSeconds={false} />
          <div className="w-px h-6 bg-border/30" />
          {/* Reasoning Level Selector */}
          <div className="flex items-center bg-surface/40 rounded-lg p-0.5 border border-border/10">
            {[
              { level: 'fast' as const, icon: Zap, color: 'warning', title: 'Fast' },
              { level: null, icon: Scale, color: 'primary', title: 'Auto' },
              { level: 'deep' as const, icon: Brain, color: 'listening', title: 'Deep' },
            ].map(({ level, icon: Icon, color, title }) => (
              <button
                key={title}
                onClick={() => onReasoningLevelChange(level === reasoningLevel ? null : level)}
                className={cn(
                  'p-1.5 rounded-md transition-all',
                  reasoningLevel === level
                    ? `bg-${color}/20 text-${color}`
                    : 'text-text-muted/60 hover:text-text-muted'
                )}
                title={title}
              >
                <Icon size={14} />
              </button>
            ))}
          </div>
          <button
            onClick={onOpenSettings}
            className="p-2 rounded-xl text-text-muted hover:text-text hover:bg-surface-2 transition-colors"
          >
            <Settings size={18} />
          </button>
        </div>
      </header>

      {/* Main Content - Clean two-column layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left - Assistant & Status */}
        <aside className="w-72 p-4 space-y-4 overflow-y-auto scrollbar-thin border-r border-border/10">
          {/* Orb */}
          <div className="flex flex-col items-center py-6">
            <Orb
              state={orbState}
              volume={volume}
              playbackVolume={playbackVolume}
              onClick={onOrbClick}
              size="md"
            />
            <p className="text-xs text-text-muted mt-4 text-center">
              {wakeWordEnabled ? (
                <>Say "<span className="text-primary">{assistantConfig.wakeWord || assistantName}</span>" to activate</>
              ) : (
                'Click orb to activate'
              )}
            </p>
            <button
              onClick={onToggleMute}
              className={cn(
                'mt-3 flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs transition-all',
                isMuted
                  ? 'bg-error/20 text-error'
                  : 'bg-surface-2 text-text-muted hover:text-text'
              )}
            >
              {isMuted ? 'Unmute' : 'Mute'}
            </button>
          </div>

        </aside>

        {/* Center - Chat */}
        <main className="flex-1 flex flex-col min-w-0">
          <MessageList
            messages={messages}
            streaming={streaming}
            isLoading={isLoading}
            loadingText={loadingText}
          />

          {/* Quick Commands */}
          <div className="px-4 py-3 border-t border-border/10">
            <QuickCommandsWidget onCommand={onQuickCommand} />
          </div>

          {/* Input */}
          <div className="px-4 pb-4">
            <UnifiedInput
              value={input}
              onChange={onInputChange}
              onSend={onSend}
              onVoiceToggle={onVoiceToggle}
              onFilesAdded={onFilesAdded}
              files={files}
              onRemoveFile={onRemoveFile}
              isRecording={isRecording}
              isLoading={isLoading}
              disabled={!connected}
            />
          </div>
        </main>

        {/* Right Sidebar - Schedule & Quick Actions */}
        <aside className="w-80 p-4 space-y-4 overflow-y-auto scrollbar-thin border-l border-border/10">
          {/* Today's Schedule */}
          <WidgetCard title="Today's Schedule" icon={<Calendar size={14} />}>
            {!calendarConnected ? (
              <button
                onClick={onOpenSettings}
                className="w-full py-4 flex flex-col items-center gap-2 text-text-muted hover:text-text transition-colors"
              >
                <Calendar className="w-6 h-6 opacity-40" />
                <span className="text-sm">Connect calendar</span>
              </button>
            ) : getTodayEvents().length === 0 ? (
              <p className="py-4 text-center text-text-muted text-sm">No events today</p>
            ) : (
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {getTodayEvents().map(event => (
                  <div key={event.id} className="flex items-center gap-2 p-2 rounded-lg bg-surface/50">
                    <div className={cn(
                      'w-1 h-8 rounded-full',
                      event.source === 'apple' ? 'bg-blue-400' : 'bg-purple-400'
                    )} />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{event.title}</p>
                      <p className="text-xs text-text-muted">
                        {event.allDay ? 'All day' : formatEventTime(event.start)}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </WidgetCard>

          {/* Quick Actions */}
          <WidgetCard title="Quick Actions">
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => {
                  onInputChange('Create a reminder for ')
                }}
                className="p-3 rounded-xl bg-surface hover:bg-surface-2 transition-colors text-left"
              >
                <Plus size={16} className="text-primary mb-1" />
                <p className="text-xs font-medium">Add Reminder</p>
              </button>
              <button
                onClick={() => {
                  onInputChange('Schedule a meeting for ')
                }}
                className="p-3 rounded-xl bg-surface hover:bg-surface-2 transition-colors text-left"
              >
                <Calendar size={16} className="text-purple-400 mb-1" />
                <p className="text-xs font-medium">New Event</p>
              </button>
              <button
                onClick={() => {
                  onInputChange("What's the weather like today?")
                  setTimeout(onSend, 100)
                }}
                className="p-3 rounded-xl bg-surface hover:bg-surface-2 transition-colors text-left"
              >
                <span className="text-base mb-1 block">🌤️</span>
                <p className="text-xs font-medium">Weather</p>
              </button>
              <button
                onClick={() => {
                  onInputChange('Search the web for ')
                }}
                className="p-3 rounded-xl bg-surface hover:bg-surface-2 transition-colors text-left"
              >
                <span className="text-base mb-1 block">🔍</span>
                <p className="text-xs font-medium">Web Search</p>
              </button>
            </div>
          </WidgetCard>

          {/* Weather */}
          <WidgetCard>
            <WeatherWidget />
          </WidgetCard>

          {/* Multi-Model Analysis */}
          <WidgetCard title="AI Analysis" icon={<BrainCircuit size={14} />}>
            <MultiModelAnalysisWidget compact />
          </WidgetCard>
        </aside>
      </div>
    </div>
  )
}

// Mobile Dashboard Component - Life Management Platform
function MobileDashboard({
  assistantConfig,
  connected,
  messages,
  streaming,
  isLoading,
  loadingText,
  orbState,
  volume,
  playbackVolume,
  isMuted: _isMuted,
  wakeWordEnabled: _wakeWordEnabled,
  input,
  files,
  reasoningLevel,
  onReasoningLevelChange,
  onInputChange,
  onSend,
  onVoiceToggle,
  onFilesAdded,
  onRemoveFile,
  onOrbClick,
  onToggleMute: _onToggleMute,
  onOpenSettings,
  onQuickCommand,
}: {
  assistantConfig: AssistantConfig
  connected: boolean
  messages: Message[]
  streaming: string
  isLoading: boolean
  loadingText: string
  orbState: OrbState
  volume: number
  playbackVolume: number
  isMuted: boolean
  wakeWordEnabled: boolean
  input: string
  files: UploadedFile[]
  reasoningLevel: ReasoningLevel
  onReasoningLevelChange: (level: ReasoningLevel) => void
  onInputChange: (value: string) => void
  onSend: () => void
  onVoiceToggle: () => void
  onFilesAdded: (files: FileList | File[]) => void
  onRemoveFile: (id: string) => void
  onOrbClick: () => void
  onToggleMute: () => void
  onOpenSettings: () => void
  onQuickCommand: (command: string) => void
}) {
  const [activeTab, setActiveTab] = useState<'home' | 'chat' | 'calendar' | 'health'>('home')
  const [calendarEvents, setCalendarEvents] = useState<CalendarEvent[]>([])
  const [calendarConnected, setCalendarConnected] = useState(false)
  const [calendarLoading, setCalendarLoading] = useState(true)
  const assistantName = assistantConfig.name || 'Jarvis'

  // Fetch calendar status and events
  useEffect(() => {
    const fetchCalendar = async () => {
      try {
        const statusRes = await fetch('/api/calendar/status')
        if (statusRes.ok) {
          const status = await statusRes.json()
          setCalendarConnected(status.connected)

          if (status.connected) {
            const eventsRes = await fetch('/api/calendar/events')
            if (eventsRes.ok) {
              const data = await eventsRes.json()
              setCalendarEvents(data.events || [])
            }
          }
        }
      } catch (err) {
        console.error('Failed to fetch calendar:', err)
      } finally {
        setCalendarLoading(false)
      }
    }
    fetchCalendar()
  }, [])

  const getGreeting = () => {
    const hour = new Date().getHours()
    if (hour < 12) return 'Good Morning'
    if (hour < 17) return 'Good Afternoon'
    return 'Good Evening'
  }

  const formatEventTime = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
  }

  const getTodayEvents = () => {
    const today = new Date().toDateString()
    return calendarEvents.filter(e => new Date(e.start).toDateString() === today)
  }

  const getUpcomingEvents = () => {
    const now = new Date()
    return calendarEvents
      .filter(e => new Date(e.start) > now)
      .slice(0, 5)
  }

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 safe-area-top border-b border-border/10">
        <div className="flex items-center gap-3">
          <Orb
            state={orbState}
            volume={volume}
            playbackVolume={playbackVolume}
            onClick={onOrbClick}
            size="sm"
            as="div"
          />
          <div>
            <h1 className="text-lg font-semibold text-text flex items-center gap-2">
              {assistantName}
              <span className={cn(
                'w-2 h-2 rounded-full',
                connected ? 'bg-success' : 'bg-error'
              )} />
            </h1>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center bg-surface/40 rounded-lg p-0.5 border border-border/10">
            {[
              { level: 'fast' as const, icon: Zap, color: 'text-warning', title: 'Fast' },
              { level: null, icon: Scale, color: 'text-primary', title: 'Auto' },
              { level: 'deep' as const, icon: Brain, color: 'text-purple-400', title: 'Deep' },
            ].map(({ level, icon: Icon, color, title }) => (
              <button
                key={title}
                onClick={() => onReasoningLevelChange(level === reasoningLevel ? null : level)}
                className={cn(
                  'p-1.5 rounded-md transition-all',
                  reasoningLevel === level
                    ? `bg-surface-2 ${color}`
                    : 'text-text-muted/60 hover:text-text-muted'
                )}
                title={title}
              >
                <Icon size={14} />
              </button>
            ))}
          </div>
          <button
            onClick={onOpenSettings}
            className="p-2 rounded-xl hover:bg-surface-2 transition-colors"
          >
            <Settings size={18} className="text-text-muted" />
          </button>
        </div>
      </header>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto">
        {/* Home Tab - Dashboard Overview */}
        {activeTab === 'home' && (
          <div className="p-4 space-y-4 pb-24">
            {/* Greeting */}
            <div className="py-2">
              <h2 className="text-2xl font-semibold text-text">{getGreeting()}!</h2>
              <p className="text-sm text-text-muted mt-1">
                {getTodayEvents().length > 0
                  ? `You have ${getTodayEvents().length} event${getTodayEvents().length > 1 ? 's' : ''} today`
                  : 'Your day is clear'}
              </p>
            </div>

            {/* Quick Stats Row */}
            <div className="grid grid-cols-2 gap-3">
              <WidgetCard className="bg-gradient-to-br from-blue-500/20 to-cyan-500/10">
                <WeatherWidget compact />
              </WidgetCard>
              <WidgetCard className="bg-gradient-to-br from-purple-500/20 to-pink-500/10">
                <div className="flex items-center gap-3">
                  <Calendar className="w-8 h-8 text-purple-400" />
                  <div>
                    <p className="text-2xl font-light">{getTodayEvents().length}</p>
                    <p className="text-xs text-text-muted">Today's Events</p>
                  </div>
                </div>
              </WidgetCard>
            </div>

            {/* Today's Schedule */}
            <WidgetCard title="Today's Schedule" icon={<Calendar size={14} />}>
              {calendarLoading ? (
                <div className="py-4 text-center text-text-muted text-sm">Loading...</div>
              ) : !calendarConnected ? (
                <button
                  onClick={onOpenSettings}
                  className="w-full py-4 flex flex-col items-center gap-2 text-text-muted hover:text-text transition-colors"
                >
                  <Calendar className="w-8 h-8 opacity-40" />
                  <span className="text-sm">Connect your calendar</span>
                  <span className="text-xs opacity-60">Tap to set up</span>
                </button>
              ) : getTodayEvents().length === 0 ? (
                <div className="py-4 text-center text-text-muted text-sm">
                  No events scheduled for today
                </div>
              ) : (
                <div className="space-y-2">
                  {getTodayEvents().map(event => (
                    <div key={event.id} className="flex items-center gap-3 p-2 rounded-lg bg-surface/50">
                      <div className="w-1 h-8 rounded-full bg-purple-400" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">{event.title}</p>
                        <p className="text-xs text-text-muted">
                          {event.allDay ? 'All day' : formatEventTime(event.start)}
                          {event.location && ` · ${event.location}`}
                        </p>
                      </div>
                      <ChevronRight size={16} className="text-text-muted/50" />
                    </div>
                  ))}
                </div>
              )}
            </WidgetCard>

            {/* Quick Actions */}
            <WidgetCard title="Quick Actions">
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => {
                    setActiveTab('chat')
                    onInputChange('Create a reminder for ')
                  }}
                  className="p-3 rounded-xl bg-surface hover:bg-surface-2 transition-colors text-left"
                >
                  <Plus size={18} className="text-primary mb-1" />
                  <p className="text-sm font-medium">Add Reminder</p>
                </button>
                <button
                  onClick={() => {
                    setActiveTab('chat')
                    onInputChange('Schedule a meeting ')
                  }}
                  className="p-3 rounded-xl bg-surface hover:bg-surface-2 transition-colors text-left"
                >
                  <Calendar size={18} className="text-purple-400 mb-1" />
                  <p className="text-sm font-medium">Schedule Event</p>
                </button>
                <button
                  onClick={() => {
                    setActiveTab('chat')
                    onInputChange("What's the weather like today?")
                    setTimeout(onSend, 100)
                  }}
                  className="p-3 rounded-xl bg-surface hover:bg-surface-2 transition-colors text-left"
                >
                  <span className="text-lg mb-1 block">🌤️</span>
                  <p className="text-sm font-medium">Weather</p>
                </button>
                <button
                  onClick={() => {
                    setActiveTab('chat')
                    onInputChange('Search the web for ')
                  }}
                  className="p-3 rounded-xl bg-surface hover:bg-surface-2 transition-colors text-left"
                >
                  <span className="text-lg mb-1 block">🔍</span>
                  <p className="text-sm font-medium">Web Search</p>
                </button>
              </div>
            </WidgetCard>

            {/* Multi-Model Analysis */}
            <WidgetCard title="AI Analysis" icon={<BrainCircuit size={14} />}>
              <MultiModelAnalysisWidget compact />
            </WidgetCard>

            {/* Health placeholder */}
            <WidgetCard title="Health & Wellness" icon={<Heart size={14} />}>
              <div className="py-4 flex flex-col items-center gap-2 text-text-muted">
                <Heart className="w-8 h-8 opacity-40" />
                <span className="text-sm">Coming soon</span>
                <span className="text-xs opacity-60">Apple Health integration</span>
              </div>
            </WidgetCard>
          </div>
        )}

        {/* Chat Tab */}
        {activeTab === 'chat' && (
          <div className="flex flex-col h-full">
            <div className="flex-1 overflow-y-auto pb-32">
              <MessageList
                messages={messages}
                streaming={streaming}
                isLoading={isLoading}
                loadingText={loadingText}
              />
            </div>
            <div className="px-4 py-2 border-t border-border/10 bg-background">
              <QuickCommandsWidget onCommand={onQuickCommand} />
            </div>
          </div>
        )}

        {/* Calendar Tab */}
        {activeTab === 'calendar' && (
          <div className="p-4 space-y-4 pb-24">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-text">Calendar</h2>
              {calendarConnected && (
                <button
                  onClick={() => {
                    setActiveTab('chat')
                    onInputChange('Create a new calendar event for ')
                  }}
                  className="p-2 rounded-lg bg-primary/20 text-primary"
                >
                  <Plus size={18} />
                </button>
              )}
            </div>

            {calendarLoading ? (
              <div className="py-8 text-center text-text-muted">Loading calendar...</div>
            ) : !calendarConnected ? (
              <WidgetCard>
                <div className="py-8 flex flex-col items-center gap-4">
                  <Calendar className="w-12 h-12 text-text-muted/40" />
                  <div className="text-center">
                    <p className="text-text font-medium">Connect Your Calendar</p>
                    <p className="text-sm text-text-muted mt-1">
                      Sync with Google Calendar to manage your schedule
                    </p>
                  </div>
                  <button
                    onClick={onOpenSettings}
                    className="px-4 py-2 rounded-lg bg-primary/20 text-primary text-sm font-medium"
                  >
                    Connect Calendar
                  </button>
                </div>
              </WidgetCard>
            ) : (
              <>
                {/* Today */}
                <WidgetCard title="Today">
                  {getTodayEvents().length === 0 ? (
                    <p className="py-4 text-center text-text-muted text-sm">No events today</p>
                  ) : (
                    <div className="space-y-2">
                      {getTodayEvents().map(event => (
                        <div key={event.id} className="flex items-start gap-3 p-3 rounded-lg bg-surface/50">
                          <div className="w-1 h-full min-h-[40px] rounded-full bg-purple-400" />
                          <div className="flex-1">
                            <p className="font-medium">{event.title}</p>
                            <p className="text-sm text-text-muted mt-1">
                              {event.allDay ? 'All day' : `${formatEventTime(event.start)}${event.end ? ` - ${formatEventTime(event.end)}` : ''}`}
                            </p>
                            {event.location && (
                              <p className="text-xs text-text-muted/70 mt-1">📍 {event.location}</p>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </WidgetCard>

                {/* Upcoming */}
                <WidgetCard title="Upcoming">
                  {getUpcomingEvents().length === 0 ? (
                    <p className="py-4 text-center text-text-muted text-sm">No upcoming events</p>
                  ) : (
                    <div className="space-y-2">
                      {getUpcomingEvents().map(event => (
                        <div key={event.id} className="flex items-center gap-3 p-2 rounded-lg hover:bg-surface/50">
                          <div className="text-center min-w-[40px]">
                            <p className="text-xs text-text-muted">
                              {new Date(event.start).toLocaleDateString('en-US', { month: 'short' })}
                            </p>
                            <p className="text-lg font-semibold">
                              {new Date(event.start).getDate()}
                            </p>
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium truncate">{event.title}</p>
                            <p className="text-xs text-text-muted">
                              {event.allDay ? 'All day' : formatEventTime(event.start)}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </WidgetCard>
              </>
            )}
          </div>
        )}

        {/* Health Tab */}
        {activeTab === 'health' && (
          <div className="p-4 space-y-4 pb-24">
            <h2 className="text-xl font-semibold text-text">Health & Wellness</h2>
            <WidgetCard>
              <div className="py-12 flex flex-col items-center gap-4">
                <Heart className="w-16 h-16 text-error/30" />
                <div className="text-center">
                  <p className="text-text font-medium">Coming Soon</p>
                  <p className="text-sm text-text-muted mt-2 max-w-[250px]">
                    Track your health metrics, sleep, activity, and more with Apple Health integration
                  </p>
                </div>
              </div>
            </WidgetCard>
          </div>
        )}
      </div>

      {/* Input (only on chat tab) */}
      {activeTab === 'chat' && (
        <div className="fixed bottom-[72px] left-0 right-0 px-4 pb-2 bg-background/95 backdrop-blur-xl border-t border-border/10 z-40">
          <UnifiedInput
            value={input}
            onChange={onInputChange}
            onSend={onSend}
            onVoiceToggle={onVoiceToggle}
            onFilesAdded={onFilesAdded}
            files={files}
            onRemoveFile={onRemoveFile}
            isRecording={false}
            isLoading={isLoading}
            disabled={!connected}
          />
        </div>
      )}

      {/* Bottom Navigation */}
      <nav className="fixed bottom-0 left-0 right-0 bg-background/95 backdrop-blur-xl border-t border-border/20 safe-area-bottom z-50">
        <div className="flex items-center justify-around py-2">
          {[
            { id: 'home' as const, icon: Home, label: 'Home' },
            { id: 'chat' as const, icon: MessageSquare, label: 'Chat' },
            { id: 'calendar' as const, icon: Calendar, label: 'Calendar' },
            { id: 'health' as const, icon: Heart, label: 'Health' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                'flex flex-col items-center gap-1 px-4 py-2 rounded-xl transition-colors',
                activeTab === tab.id ? 'text-primary' : 'text-text-muted'
              )}
            >
              <tab.icon size={20} />
              <span className="text-[10px]">{tab.label}</span>
            </button>
          ))}
        </div>
      </nav>
    </div>
  )
}
