import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from filterpy.kalman import KalmanFilter
import joblib

# --- 1. Load models & scalers (inference only) ---
reg_model = load_model('gaze_regressor.h5', compile=False)
cls_model = load_model('focus_classifier.h5', compile=False)

# Optional: compile if further training is needed
# reg_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# cls_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

scaler_reg = joblib.load('scaler_reg.pkl')
scaler_cls = joblib.load('scaler_cls.pkl')

# --- 2. MediaPipe FaceMesh & Kalman filter ---
mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5
)
kf = KalmanFilter(dim_x=2, dim_z=2)
kf.F = np.eye(2); kf.H = np.eye(2)
kf.x = np.zeros(2); kf.P *= 1000
kf.R = np.diag([0.1, 0.1]); kf.Q = np.eye(2)*0.01

def get_head_pos(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mp_face.process(rgb)
    if not res.multi_face_landmarks: return None
    lm = res.multi_face_landmarks[0].landmark[1]
    return [lm.x, lm.y]

# --- 3. Inference loop ---
def infer_and_display():
    cap = cv2.VideoCapture(0)
    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret: break

        head = get_head_pos(frame)
        if head:
            h_scaled = scaler_reg.transform([head])
            pred = reg_model.predict(h_scaled, verbose=0)[0]
            kf.predict(); kf.update(pred)
            gaze = kf.x

            prob = cls_model.predict(scaler_cls.transform([gaze]), verbose=0)[0][0]
            status = "Looking at Screen" if prob > 0.5 else "Away"
            color = (0,255,0) if status=="Looking at Screen" else (0,0,255)
            cv2.putText(frame, status, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Gaze Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    infer_and_display()
