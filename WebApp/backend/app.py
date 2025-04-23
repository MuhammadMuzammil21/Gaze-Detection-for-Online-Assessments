# app.py
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
import cv2, numpy as np, base64, re
from utils.gaze_detector import process_frame 

app = Flask(__name__, static_folder="template")
socketio = SocketIO(app, cors_allowed_origins="*")

DATA_RE = re.compile(r"^data:image/\w+;base64,")

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@socketio.on('frame')
def handle_frame(b64):
    """
    Receives a base64 JPEG frame from the client,
    runs process_frame(), and emits back the gaze + new JPEG.
    """
    # strip header
    data = DATA_RE.sub("", b64)
    img_bytes = base64.b64decode(data)
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    jpeg, gaze, _ = process_frame(frame)

    # send back as JSON with base64 JPEG
    resp = {
      'gaze': gaze,
      'image': 'data:image/jpeg;base64,' + base64.b64encode(jpeg).decode()
    }
    emit('processed', resp)

if __name__ == "__main__":
    # eventlet server on 4150
    socketio.run(app, host="0.0.0.0", port=4150)
