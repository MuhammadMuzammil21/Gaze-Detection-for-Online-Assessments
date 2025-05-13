// src/hooks/useGazeSocket.ts
import { useEffect, useRef } from "react"
import { io, Socket } from "socket.io-client"

const TARGET_W = 320
const TARGET_H = 240
const QUALITY = 0.5

export function useGazeSocket(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  onGazeAlert: (gaze: string) => void,
  onFrameProcessed?: (image: string) => void
) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const processingRef = useRef<boolean>(false)
  const socketRef = useRef<Socket | null>(null)

  useEffect(() => {
    const canvas = document.createElement("canvas")
    canvasRef.current = canvas
    const ctx = canvas.getContext("2d")

    const socket = io("http://localhost:4150") // or your ngrok HTTPS URL
    socketRef.current = socket

    // ✅ Receive processed image and distraction status
    socket.on("processed", ({ image, distracted }: { image: string; distracted: boolean }) => {
      if (onFrameProcessed) onFrameProcessed(image)
      if (distracted) onGazeAlert("distracted")
      processingRef.current = false
    })

    // ✅ Start webcam
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      const video = videoRef.current
      if (video) {
        video.srcObject = stream
        video.play()
        requestAnimationFrame(captureFrame)
      }
    })

    // ✅ Send frame to backend
    function captureFrame() {
      const video = videoRef.current
      if (video && ctx && !processingRef.current && video.readyState >= 2) {
        processingRef.current = true
        canvas.width = TARGET_W
        canvas.height = TARGET_H

        ctx.save()
        ctx.translate(TARGET_W, 0)
        ctx.scale(-1, 1)
        ctx.drawImage(video, 0, 0, TARGET_W, TARGET_H)
        ctx.restore()

        const b64 = canvas.toDataURL("image/jpeg", QUALITY)
        socket.emit("frame", b64)
      }
      requestAnimationFrame(captureFrame)
    }

    return () => {
      socket.disconnect()
      if (videoRef.current?.srcObject instanceof MediaStream) {
        videoRef.current.srcObject.getTracks().forEach((t) => t.stop())
      }
    }
  }, [videoRef, onGazeAlert, onFrameProcessed])
}
