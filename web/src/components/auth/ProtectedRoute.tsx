import { Navigate } from 'react-router-dom'
import { useAuth } from '../../contexts/AuthContext'
import { Loader2 } from 'lucide-react'

export function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading, authEnabled } = useAuth()

  // Still checking auth status
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Loader2 size={32} className="animate-spin text-primary" />
      </div>
    )
  }

  // Auth not enabled — render app directly
  if (!authEnabled) {
    return <>{children}</>
  }

  // Auth enabled but not logged in — redirect to login
  if (!user) {
    return <Navigate to="/login" replace />
  }

  return <>{children}</>
}
