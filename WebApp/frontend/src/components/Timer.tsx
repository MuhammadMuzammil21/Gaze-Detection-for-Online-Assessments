import { useEffect, useState } from "react"

interface TimerProps {
  initialMinutes?: number
  initialSeconds?: number
}

export default function Timer({ initialMinutes = 45, initialSeconds = 32 }: TimerProps) {
  const [minutes, setMinutes] = useState(initialMinutes)
  const [seconds, setSeconds] = useState(initialSeconds)

  useEffect(() => {
    const timer = setInterval(() => {
      if (seconds > 0) {
        setSeconds(seconds - 1)
      } else {
        if (minutes === 0) {
          clearInterval(timer)
        } else {
          setMinutes(minutes - 1)
          setSeconds(59)
        }
      }
    }, 1000)

    return () => clearInterval(timer)
  }, [minutes, seconds])

  const formatTime = (time: number) => time.toString().padStart(2, "0")

  return (
    <div className="text-lg font-medium text-gray-600">
      Timer: {formatTime(minutes)}:{formatTime(seconds)}
    </div>
  )
}
