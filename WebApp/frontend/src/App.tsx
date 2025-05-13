import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Loader2 } from "lucide-react"
import { toast } from "sonner"
import Timer from "@/components/Timer"
import WebcamPreview from "@/components/ui/WebcamPreview"
import { useGazeSocket } from "@/hooks/useGazeSocket"
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogHeader,
  AlertDialogTitle
} from "@/components/ui/alert-dialog"


export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const videoRef = useRef<HTMLVideoElement>(null)
  const [gazeImage, setGazeImage] = useState<string | null>(null)
  const [showDistractionAlert, setShowDistractionAlert] = useState(false)


  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      if (validateFileType(selectedFile)) {
        setFile(selectedFile)
      } else {
        toast.error("Invalid file type. Please upload a PDF, JPG, or PNG.")
      }
    }
  }

  const validateFileType = (file: File) => {
    const allowedTypes = ["application/pdf", "image/jpeg", "image/png"]
    return allowedTypes.includes(file.type)
  }

  const handleSubmit = async () => {
    if (!file) {
      toast.error("No file selected. Please upload your answer sheet before submitting.")
      return
    }

    setIsSubmitting(true)
    await new Promise((resolve) => setTimeout(resolve, 2000)) // Simulate API call
    setIsSubmitting(false)
    toast.success(`${file.name} has been submitted successfully! 🎉`)
    setFile(null)
    if (inputRef.current) inputRef.current.value = ""
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0]
      if (validateFileType(droppedFile)) {
        setFile(droppedFile)
      } else {
        toast.error("Invalid file type. Please upload a PDF, JPG, or PNG.")
      }
    }
  }

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const triggerGazeAlert = (gaze: string) => {
    if (gaze === "distracted") {
      setShowDistractionAlert(true)
    }
  }  

  useGazeSocket(videoRef, triggerGazeAlert, setGazeImage)

  return (
    <div className="w-screen min-h-screen flex flex-col bg-gray-50">
      <header className="w-full p-4 flex justify-between items-center bg-white shadow">
        <h1 className="text-2xl font-bold text-gray-800">FocusTest Portal</h1>
        <div className="text-lg font-medium text-gray-600">
          <Timer initialMinutes={45} initialSeconds={32} />
        </div>
      </header>

      <main className="flex flex-1 justify-center items-center p-6">
        <div className="w-full max-w-4xl px-4">
          <Card className="w-full shadow-lg">
            <CardHeader>
              <CardTitle>Upload Your Answer Sheet</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                className="w-full h-32 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center text-gray-400 cursor-pointer hover:border-gray-500 transition"
                onClick={() => inputRef.current?.click()}
              >
                {file ? (
                  <div className="text-gray-700 font-medium">{file.name}</div>
                ) : (
                  <div>Drag & Drop your file here or Click to Browse</div>
                )}
              </div>

              <Input
                type="file"
                accept=".pdf,.jpg,.jpeg,.png"
                onChange={handleFileChange}
                ref={inputRef}
                className="hidden"
              />

              <Button onClick={handleSubmit} className="w-full" disabled={isSubmitting}>
                {isSubmitting ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Submitting...
                  </>
                ) : (
                  "Submit"
                )}
              </Button>

              <AlertDialog open={showDistractionAlert} onOpenChange={setShowDistractionAlert}>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>⚠️ You seem distracted</AlertDialogTitle>
                    <AlertDialogDescription>
                      Please stay focused on the test window. This activity is being monitored.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                </AlertDialogContent>
              </AlertDialog>


              {/* {gazeImage && (
                <img src={gazeImage} alt="Processed frame" className="w-full max-w-md mx-auto rounded shadow" />
              )} */}
            </CardContent>
          </Card>
        </div>
      </main>

      <video ref={videoRef} className="hidden" playsInline autoPlay />
      <WebcamPreview />
    </div>
  )
}
