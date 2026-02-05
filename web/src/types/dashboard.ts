// Dashboard & Widget Types for the AI Assistant Platform

// === Assistant Branding (Configurable) ===
export interface AssistantConfig {
  name: string              // "Jarvis", "Friday", "Alfred", etc.
  greeting: string          // "Good morning" / custom
  avatar?: string           // Custom avatar URL
  accentColor?: string      // Theme accent color
  wakeWord?: string         // Voice activation word
}

// === Widget System ===
export type WidgetSize = 'sm' | 'md' | 'lg' | 'xl'
export type WidgetCategory =
  | 'core'           // Voice, Chat, Quick Actions
  | 'productivity'   // Tasks, Calendar, Email
  | 'smart-home'     // IoT, Rooms, Devices
  | 'health'         // Fitness, Sleep, Nutrition
  | 'info'           // Weather, Time, System Status
  | 'communication'  // Messages, Notifications

export interface Widget {
  id: string
  type: string
  title: string
  category: WidgetCategory
  size: WidgetSize
  position: { row: number; col: number }
  enabled: boolean
  config?: Record<string, unknown>
}

// === Dashboard Layout ===
export interface DashboardLayout {
  id: string
  name: string
  description?: string
  widgets: Widget[]
  isDefault?: boolean
}

// === Quick Commands ===
export interface QuickCommand {
  id: string
  label: string
  icon: string
  command: string
  category?: string
}

// === Weather Widget ===
export interface WeatherData {
  temperature: number
  unit: 'C' | 'F'
  condition: string
  icon: string
  humidity: number
  wind: number
  visibility: number
  feelsLike: number
  location: string
  forecast?: WeatherForecast[]
}

export interface WeatherForecast {
  day: string
  high: number
  low: number
  condition: string
  icon: string
}

// === System Status Widget ===
export interface SystemStatus {
  cpu: number
  memory: number
  disk: number
  network: 'connected' | 'disconnected' | 'limited'
  uptime?: string
}

// === Task Widget ===
export interface Task {
  id: string
  title: string
  description?: string
  dueDate?: Date
  priority: 'low' | 'medium' | 'high'
  status: 'pending' | 'in_progress' | 'completed'
  tags?: string[]
  reminder?: Date
}

// === Calendar Widget ===
export interface CalendarEvent {
  id: string
  title: string
  start: Date
  end: Date
  location?: string
  description?: string
  color?: string
  isAllDay?: boolean
}

// === IoT Device Widget ===
export type DeviceType = 'light' | 'thermostat' | 'switch' | 'camera' | 'lock' | 'sensor' | 'speaker'
export type DeviceStatus = 'online' | 'offline' | 'unknown'

export interface IoTDevice {
  id: string
  name: string
  type: DeviceType
  room: string
  status: DeviceStatus
  state?: Record<string, unknown>  // Device-specific state
  lastSeen?: Date
}

export interface Room {
  id: string
  name: string
  icon?: string
  devices: IoTDevice[]
}

// === Health Widget ===
export interface HealthMetrics {
  steps: number
  stepsGoal: number
  calories: number
  caloriesGoal: number
  heartRate?: number
  sleep?: {
    duration: number  // minutes
    quality: 'poor' | 'fair' | 'good' | 'excellent'
  }
  water?: number  // ml
  waterGoal?: number
}

// === User Profile ===
export interface UserProfile {
  id: string
  name: string
  avatar?: string
  email?: string
  status: 'online' | 'away' | 'busy' | 'offline'
  role?: string
}

// === Notification ===
export interface Notification {
  id: string
  type: 'info' | 'warning' | 'error' | 'success'
  title: string
  message: string
  timestamp: Date
  read: boolean
  action?: {
    label: string
    handler: string
  }
}

// === Operating Modes ===
export type OperatingMode = 'normal' | 'focus' | 'night' | 'security' | 'fun'

export interface ModeConfig {
  id: OperatingMode
  name: string
  description: string
  icon: string
  settings: {
    voiceEnabled: boolean
    notificationsEnabled: boolean
    dimUI: boolean
    customPrompt?: string
  }
}

// === Integration Status ===
export interface IntegrationStatus {
  id: string
  name: string
  icon: string
  connected: boolean
  lastSync?: Date
  error?: string
}

// === Message Thread (for Telegram/WhatsApp/iMessage) ===
export interface MessageThread {
  id: string
  platform: 'telegram' | 'whatsapp' | 'imessage' | 'email'
  contact: {
    name: string
    avatar?: string
  }
  lastMessage: {
    content: string
    timestamp: Date
    isFromMe: boolean
  }
  unreadCount: number
}
