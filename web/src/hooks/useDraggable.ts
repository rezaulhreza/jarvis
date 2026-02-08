import { useState, useRef, useCallback, useEffect } from 'react'

interface Position {
  x: number
  y: number
}

interface Size {
  width: number
  height: number
}

interface UseDraggableOptions {
  defaultPosition?: Position
  defaultSize?: Size
  minSize?: Size
  maxSize?: Size
  snapThreshold?: number
  storageKey?: string
}

type Corner = 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right'

export function useDraggable({
  defaultPosition = { x: window.innerWidth - 180, y: window.innerHeight - 180 },
  defaultSize = { width: 160, height: 120 },
  minSize = { width: 120, height: 90 },
  maxSize = { width: 480, height: 360 },
  snapThreshold = 20,
  storageKey,
}: UseDraggableOptions = {}) {
  // Load from localStorage if available
  const loadSaved = (): { position: Position; size: Size } | null => {
    if (!storageKey) return null
    try {
      const saved = localStorage.getItem(storageKey)
      if (saved) return JSON.parse(saved)
    } catch { /* ignore parse errors */ }
    return null
  }

  const saved = loadSaved()

  const [position, setPosition] = useState<Position>(saved?.position || defaultPosition)
  const [size, setSize] = useState<Size>(saved?.size || defaultSize)
  const [isDragging, setIsDragging] = useState(false)
  const [isResizing, setIsResizing] = useState(false)

  const dragStartPos = useRef<Position>({ x: 0, y: 0 })
  const dragStartOffset = useRef<Position>({ x: 0, y: 0 })
  const resizeStart = useRef<{ pos: Position; size: Size; corner: Corner }>({
    pos: { x: 0, y: 0 },
    size: { width: 0, height: 0 },
    corner: 'bottom-right',
  })

  // Persist to localStorage
  const persist = useCallback((pos: Position, sz: Size) => {
    if (!storageKey) return
    try {
      localStorage.setItem(storageKey, JSON.stringify({ position: pos, size: sz }))
    } catch { /* ignore storage errors */ }
  }, [storageKey])

  // Snap to nearest edge
  const snapToEdge = useCallback((pos: Position, sz: Size): Position => {
    const vw = window.innerWidth
    const vh = window.innerHeight
    const snapped = { ...pos }

    // Snap X
    if (pos.x < snapThreshold) snapped.x = 8
    else if (pos.x + sz.width > vw - snapThreshold) snapped.x = vw - sz.width - 8

    // Snap Y
    if (pos.y < snapThreshold) snapped.y = 8
    else if (pos.y + sz.height > vh - snapThreshold) snapped.y = vh - sz.height - 8

    return snapped
  }, [snapThreshold])

  // Clamp to viewport
  const clamp = useCallback((pos: Position, sz: Size): Position => ({
    x: Math.max(0, Math.min(pos.x, window.innerWidth - sz.width)),
    y: Math.max(0, Math.min(pos.y, window.innerHeight - sz.height)),
  }), [])

  // Drag handlers
  const onDragStart = useCallback((e: React.PointerEvent) => {
    e.preventDefault()
    e.stopPropagation()
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
    setIsDragging(true)
    dragStartPos.current = { x: e.clientX, y: e.clientY }
    dragStartOffset.current = { ...position }
  }, [position])

  const onDragMove = useCallback((e: React.PointerEvent) => {
    if (!isDragging) return
    const dx = e.clientX - dragStartPos.current.x
    const dy = e.clientY - dragStartPos.current.y
    const newPos = clamp({
      x: dragStartOffset.current.x + dx,
      y: dragStartOffset.current.y + dy,
    }, size)
    setPosition(newPos)
  }, [isDragging, size, clamp])

  const onDragEnd = useCallback((e: React.PointerEvent) => {
    if (!isDragging) return
    ;(e.target as HTMLElement).releasePointerCapture(e.pointerId)
    setIsDragging(false)
    const snapped = snapToEdge(position, size)
    setPosition(snapped)
    persist(snapped, size)
  }, [isDragging, position, size, snapToEdge, persist])

  // Resize handlers
  const onResizeStart = useCallback((corner: Corner) => (e: React.PointerEvent) => {
    e.preventDefault()
    e.stopPropagation()
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
    setIsResizing(true)
    resizeStart.current = {
      pos: { x: e.clientX, y: e.clientY },
      size: { ...size },
      corner,
    }
  }, [size])

  const onResizeMove = useCallback((e: React.PointerEvent) => {
    if (!isResizing) return
    const { pos: startPos, size: startSize, corner } = resizeStart.current
    const dx = e.clientX - startPos.x
    const dy = e.clientY - startPos.y

    let newWidth = startSize.width
    let newHeight = startSize.height
    let newX = position.x
    let newY = position.y

    if (corner.includes('right')) newWidth = startSize.width + dx
    if (corner.includes('left')) {
      newWidth = startSize.width - dx
      newX = position.x + dx
    }
    if (corner.includes('bottom')) newHeight = startSize.height + dy
    if (corner.includes('top')) {
      newHeight = startSize.height - dy
      newY = position.y + dy
    }

    // Clamp size
    newWidth = Math.max(minSize.width, Math.min(maxSize.width, newWidth))
    newHeight = Math.max(minSize.height, Math.min(maxSize.height, newHeight))

    // Maintain aspect ratio (4:3)
    newHeight = Math.round(newWidth * 0.75)
    newHeight = Math.max(minSize.height, Math.min(maxSize.height, newHeight))

    setSize({ width: newWidth, height: newHeight })
    if (corner.includes('left') || corner.includes('top')) {
      setPosition(clamp({ x: newX, y: newY }, { width: newWidth, height: newHeight }))
    }
  }, [isResizing, position, minSize, maxSize, clamp])

  const onResizeEnd = useCallback((e: React.PointerEvent) => {
    if (!isResizing) return
    ;(e.target as HTMLElement).releasePointerCapture(e.pointerId)
    setIsResizing(false)
    const snapped = snapToEdge(position, size)
    setPosition(snapped)
    persist(snapped, size)
  }, [isResizing, position, size, snapToEdge, persist])

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      setPosition(prev => clamp(prev, size))
    }
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [size, clamp])

  const dragHandleProps = {
    onPointerDown: onDragStart,
    onPointerMove: onDragMove,
    onPointerUp: onDragEnd,
    style: { cursor: isDragging ? 'grabbing' : 'grab', touchAction: 'none' as const },
  }

  const resizeHandleProps = (corner: Corner) => ({
    onPointerDown: onResizeStart(corner),
    onPointerMove: onResizeMove,
    onPointerUp: onResizeEnd,
    style: { touchAction: 'none' as const },
  })

  return {
    position,
    size,
    isDragging,
    isResizing,
    dragHandleProps,
    resizeHandleProps,
  }
}
