import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** Read the CSRF token from the cookie set by the backend. */
export function getCsrfToken(): string {
  const match = document.cookie.match(/jarvis_csrf=([^;]+)/)
  return match ? match[1] : ''
}

/** Wrapper around fetch that auto-includes CSRF header on mutating requests. */
export function apiFetch(url: string, init?: RequestInit): Promise<Response> {
  const method = (init?.method ?? 'GET').toUpperCase()
  if (['POST', 'PUT', 'DELETE', 'PATCH'].includes(method)) {
    const headers = new Headers(init?.headers)
    if (!headers.has('x-csrf-token')) {
      headers.set('x-csrf-token', getCsrfToken())
    }
    return fetch(url, { ...init, headers })
  }
  return fetch(url, init)
}
