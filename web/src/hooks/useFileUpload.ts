import { useState, useCallback } from 'react'
import type { UploadedFile, FileType, UploadStatus } from '../types'

// File type detection
function getFileType(mimeType: string): FileType {
  if (mimeType.startsWith('image/')) return 'image'
  if (mimeType.startsWith('video/')) return 'video'
  if (mimeType.startsWith('audio/')) return 'audio'
  return 'document'
}

// Generate unique ID
function generateId(): string {
  return `file-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
}

// Max file size (10MB)
const MAX_FILE_SIZE = 10 * 1024 * 1024

// Supported file types
const SUPPORTED_TYPES = [
  'image/jpeg',
  'image/png',
  'image/gif',
  'image/webp',
  'image/svg+xml',
  'video/mp4',
  'video/webm',
  'audio/mpeg',
  'audio/wav',
  'audio/webm',
  'application/pdf',
  'text/plain',
  'text/markdown',
  'text/csv',
  'application/json',
  // Documents
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document', // .docx
  'application/msword', // .doc
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', // .xlsx
  'application/vnd.ms-excel', // .xls
]

interface UseFileUploadReturn {
  files: UploadedFile[]
  isUploading: boolean
  addFiles: (files: FileList | File[]) => Promise<void>
  removeFile: (id: string) => void
  clearFiles: () => void
  getAttachmentIds: () => string[]
}

export function useFileUpload(): UseFileUploadReturn {
  const [files, setFiles] = useState<UploadedFile[]>([])
  const [isUploading, setIsUploading] = useState(false)

  // Create preview for images
  const createPreview = useCallback((file: File): Promise<string | undefined> => {
    return new Promise((resolve) => {
      if (!file.type.startsWith('image/')) {
        resolve(undefined)
        return
      }

      const reader = new FileReader()
      reader.onload = (e) => resolve(e.target?.result as string)
      reader.onerror = () => resolve(undefined)
      reader.readAsDataURL(file)
    })
  }, [])

  // Upload a single file to the server
  const uploadFile = useCallback(async (file: File, id: string): Promise<string | null> => {
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`)
      }

      const data = await response.json()
      return data.id || id
    } catch (error) {
      console.error('Upload error:', error)
      return null
    }
  }, [])

  // Add files to the upload queue
  const addFiles = useCallback(async (fileList: FileList | File[]) => {
    const newFiles = Array.from(fileList)

    // Validate files
    const validFiles = newFiles.filter((file) => {
      if (file.size > MAX_FILE_SIZE) {
        console.warn(`File ${file.name} is too large (max ${MAX_FILE_SIZE / 1024 / 1024}MB)`)
        return false
      }
      if (!SUPPORTED_TYPES.includes(file.type)) {
        console.warn(`File type ${file.type} is not supported`)
        return false
      }
      return true
    })

    if (validFiles.length === 0) return

    setIsUploading(true)

    // Create pending entries with previews
    const pendingFiles: UploadedFile[] = await Promise.all(
      validFiles.map(async (file) => {
        const id = generateId()
        const preview = await createPreview(file)
        return {
          id,
          name: file.name,
          type: getFileType(file.type),
          mimeType: file.type,
          size: file.size,
          preview,
          status: 'pending' as UploadStatus,
        }
      })
    )

    // Add pending files to state
    setFiles((prev) => [...prev, ...pendingFiles])

    // Upload each file
    for (let i = 0; i < validFiles.length; i++) {
      const file = validFiles[i]
      const pendingFile = pendingFiles[i]

      // Update status to uploading
      setFiles((prev) =>
        prev.map((f) =>
          f.id === pendingFile.id ? { ...f, status: 'uploading' as UploadStatus } : f
        )
      )

      // Upload
      const uploadedId = await uploadFile(file, pendingFile.id)

      // Update status based on result
      setFiles((prev) =>
        prev.map((f) => {
          if (f.id === pendingFile.id) {
            if (uploadedId) {
              return { ...f, id: uploadedId, status: 'uploaded' as UploadStatus }
            } else {
              return { ...f, status: 'error' as UploadStatus, error: 'Upload failed' }
            }
          }
          return f
        })
      )
    }

    setIsUploading(false)
  }, [createPreview, uploadFile])

  // Remove a file
  const removeFile = useCallback((id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id))
  }, [])

  // Clear all files
  const clearFiles = useCallback(() => {
    setFiles([])
  }, [])

  // Get IDs of uploaded files for message attachments
  const getAttachmentIds = useCallback(() => {
    return files
      .filter((f) => f.status === 'uploaded')
      .map((f) => f.id)
  }, [files])

  return {
    files,
    isUploading,
    addFiles,
    removeFile,
    clearFiles,
    getAttachmentIds,
  }
}
