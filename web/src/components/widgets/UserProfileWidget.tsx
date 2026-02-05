import { cn } from '../../lib/utils'
import type { UserProfile } from '../../types/dashboard'

interface UserProfileWidgetProps {
  className?: string
  user?: UserProfile
}

const STATUS_COLORS = {
  online: 'bg-success',
  away: 'bg-warning',
  busy: 'bg-error',
  offline: 'bg-text-muted/50',
}

export function UserProfileWidget({ className = '', user }: UserProfileWidgetProps) {
  const defaultUser: UserProfile = {
    id: '1',
    name: 'User',
    status: 'online',
    role: 'Administrator',
  }

  const profile = user || defaultUser

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <div className="relative">
        {profile.avatar ? (
          <img
            src={profile.avatar}
            alt={profile.name}
            className="w-10 h-10 rounded-full object-cover"
          />
        ) : (
          <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center text-primary font-semibold">
            {profile.name.charAt(0).toUpperCase()}
          </div>
        )}
        <div
          className={cn(
            'absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full border-2 border-surface',
            STATUS_COLORS[profile.status]
          )}
        />
      </div>
      <div>
        <p className="text-sm font-medium text-text">{profile.name}</p>
        <p className="text-xs text-text-muted">
          {profile.role} · <span className="capitalize">{profile.status}</span>
        </p>
      </div>
    </div>
  )
}
