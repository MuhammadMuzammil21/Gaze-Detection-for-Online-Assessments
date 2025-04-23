import React, { useEffect, useRef, useState } from 'react'
import { io, Socket } from 'socket.io-client'

const TARGET_W = 320
const TARGET_H = 240
const QUALITY = 0.5

const App: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const viewRef = useRef<HTMLImageElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const processingRef = useRef<boolean>(false)
  const socketRef = useRef<Socket>()
  const [status, setStatus] = useState('initialising …')

  useEffect(() => {
    const canvas = document.createElement('canvas')
    canvasRef.current = canvas
    const ctx = canvas.getContext('2d')

    const socket = io()
    socketRef.current = socket
    socket.on('processed', ({ image, gaze }: { image: string; gaze: string }) => {
      if (viewRef.current) viewRef.current.src = image
      setStatus(`Gaze: ${gaze}`)
      processingRef.current = false
    })

    // start camera
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          videoRef.current.play()
          requestAnimationFrame(captureFrame)
        }
      })
      .catch(err => setStatus(`camera failure: ${err}`))

    function captureFrame() {
      const video = videoRef.current
      if (
        video &&
        ctx &&
        !processingRef.current &&
        video.readyState >= 2
      ) {
        processingRef.current = true

        
        canvas.width = TARGET_W
        canvas.height = TARGET_H
        ctx.save()
        ctx.translate(TARGET_W, 0)
        ctx.scale(-1, 1)
        ctx.drawImage(video, 0, 0, TARGET_W, TARGET_H)
        ctx.restore()

        const b64 = canvas.toDataURL('image/jpeg', QUALITY)
        socket.emit('frame', b64)
      }
      requestAnimationFrame(captureFrame)
    }

    return () => {
      socket.disconnect()
      // stop camera
      if (
        videoRef.current?.srcObject &&
        videoRef.current.srcObject instanceof MediaStream
      ) {
        videoRef.current.srcObject.getTracks().forEach(t => t.stop())
      }
    }
  }, [])

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-gray-100">
      <video
        ref={videoRef}
        className="hidden"
        playsInline
        autoPlay
      />
      <img
        ref={viewRef}
        className="max-w-full"
        alt="processed frame will appear here"
      />
      <div className="mt-4 text-lg">{status}</div>
    </div>
  )
}

export default App
