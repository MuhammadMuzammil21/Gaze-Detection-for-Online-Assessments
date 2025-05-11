# ================== Load existing models without recompiling ==================
from tensorflow.keras.models import load_model
reg_model = load_model('gaze_regressor.h5', compile=False)    # :contentReference[oaicite:5]{index=5}
cls_model = load_model('focus_classifier.h5', compile=False)  # :contentReference[oaicite:6]{index=6}
# ==============================================================================

import os
import numpy as np
import cv2
import mediapipe as mp
from sklearn.preprocessing import StandardScaler                 # :contentReference[oaicite:7]{index=7}
from filterpy.kalman import KalmanFilter                         # :contentReference[oaicite:8]{index=8}
from datasets import load_dataset                                # :contentReference[oaicite:9]{index=9}

# Paths for optional scaler persistence (we'll fit on the fly)
# scaler_reg_path = 'scaler_reg.pkl'
# scaler_cls_path = 'scaler_cls.pkl'

# =============================================================================
# # OPTIONAL TRAINING BLOCK — COMMENTED OUT TO AVOID OVERWRITING YOUR .h5 FILES
# 
# if not os.path.exists('gaze_regressor.h5') or not os.path.exists('focus_classifier.h5'):
#     # 1. Load and parse Gaze360 for quick fitting
#     data = load_dataset("Morning5/Gaze360", split="train")
#     Xr, yr, Xc, yc = [], [], [], []
#     for row in data['text']:
#         _, _, _, hx, hy, _, gx, gy = row.split(',')
#         head = [float(hx), float(hy)]
#         gaze = [float(gx), float(gy)]
#         Xr.append(head); yr.append(gaze)
#         focus = int(0.4 <= gaze[0] <= 0.6 and 0.4 <= gaze[1] <= 0.6)
#         Xc.append(gaze); yc.append(focus)
#     Xr, yr = np.array(Xr), np.array(yr)
#     Xc, yc = np.array(Xc), np.array(yc)
# 
#     # 2. Scale features
#     scaler_reg = StandardScaler().fit(Xr)
#     scaler_cls = StandardScaler().fit(Xc)
#     Xr_t = scaler_reg.transform(Xr); Xc_t = scaler_cls.transform(Xc)
# 
#     # 3. Build & train for a single epoch
#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.layers import Dense, Dropout
#     reg_model = Sequential([Dense(64, activation='relu', input_shape=(2,)),
#                              Dropout(0.2), Dense(64, activation='relu'),
#                              Dropout(0.2), Dense(2)])
#     reg_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     reg_model.fit(Xr_t, yr, epochs=1, batch_size=32, verbose=1)
# 
#     cls_model = Sequential([Dense(32, activation='relu', input_shape=(2,)),
#                              Dropout(0.2), Dense(16, activation='relu'),
#                              Dropout(0.2), Dense(1, activation='sigmoid')])
#     cls_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     cls_model.fit(Xc_t, yc, epochs=1, batch_size=32, verbose=1)
# 
#     # 4. Save models and scalers (commented to protect your .h5)
#     # reg_model.save('gaze_regressor.h5')
#     # cls_model.save('focus_classifier.h5')
#     # import joblib
#     # joblib.dump(scaler_reg, scaler_reg_path)
#     # joblib.dump(scaler_cls, scaler_cls_path)
# =============================================================================

# 5. Compute scalers on-the-fly (since .pkl files are missing)
#    This avoids further disk writes but ensures proper normalization.
data = load_dataset("Morning5/Gaze360", split="train")           # :contentReference[oaicite:10]{index=10}
Xr, yr, Xc, yc = [], [], [], []
for row in data['text']:
    _, _, _, hx, hy, _, gx, gy = row.split(',')
    Xr.append([float(hx), float(hy)])
    Xc.append([float(gx), float(gy)])
scaler_reg = StandardScaler().fit(np.array(Xr))                 # :contentReference[oaicite:11]{index=11}
scaler_cls = StandardScaler().fit(np.array(Xc))                 # :contentReference[oaicite:12]{index=12}

# 6. Setup MediaPipe FaceMesh & Kalman Filter
mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5
)                                                                # :contentReference[oaicite:13]{index=13}

kf = KalmanFilter(dim_x=2, dim_z=2)
kf.F = np.eye(2); kf.H = np.eye(2)
kf.x = np.zeros(2); kf.P *= 1000
kf.R = np.diag([0.1, 0.1]); kf.Q = np.eye(2) * 0.01             # :contentReference[oaicite:14]{index=14}

def get_head_pos(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mp_face.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark[1]  # nose tip
    return [lm.x, lm.y]

# 7. Real-time inference loop
def infer_and_display():
    cap = cv2.VideoCapture(0)                                # :contentReference[oaicite:15]{index=15}
    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        head = get_head_pos(frame)
        if head:
            scaled_h = scaler_reg.transform([head])
            pred = reg_model.predict(scaled_h, verbose=0)[0]
            kf.predict(); kf.update(pred)
            gaze = kf.x
            prob = cls_model.predict(scaler_cls.transform([gaze]), verbose=0)[0][0]
            status = "Looking at Screen" if prob > 0.5 else "Away"
            color = (0,255,0) if status == "Looking at Screen" else (0,0,255)
            cv2.putText(frame, status, (30,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Gaze Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    infer_and_display()
