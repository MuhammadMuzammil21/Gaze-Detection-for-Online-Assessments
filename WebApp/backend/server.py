from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
import cv2, numpy as np, base64, re
from utils.gaze_detector import process_frame
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__, static_folder="template")
socketio = SocketIO(app, cors_allowed_origins="*")
DATA_RE = re.compile(r"^data:image/\w+;base64,")

@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def serve_template(path):
    return send_from_directory(app.static_folder, path)

@socketio.on('frame')
def handle_frame(b64):
    try:
        data = DATA_RE.sub("", b64)
        img_bytes = base64.b64decode(data)
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        jpeg, gaze, distracted = process_frame(frame)

        resp = {
            'image': 'data:image/jpeg;base64,' + base64.b64encode(jpeg).decode(),
            'gaze': gaze,
            'distracted': distracted
        }
        emit('processed', resp)
    except Exception as e:
        print(f"[ERROR] processing frame: {e}")

# ✅ Proper server startup block
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    socketio.run(app, host="0.0.0.0", port=4150, debug=True)
