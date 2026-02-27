import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer

# --- إعدادات الصفحة ---
st.set_page_config(page_title="وصال AI - بث مباشر", layout="centered")
st.title("wesal-AI live")

# --- تحميل الموديل (1530 نقطة) ---
actions = np.array(['HELP', 'TOILET', 'KF_GATE', 'THANKS', 'WELCOME'])

@st.cache_resource
def load_my_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(30, 1530))) 
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(actions), activation='softmax'))
    model.load_weights(r"C:\Users\ASUS\Downloads\action_model.h5") # تأكدي من وجود الملف في نفس المجلد
    return model

model = load_my_model()

# --- معالج الفيديو (هنا السحر!) ---
class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.sequence = []
        self.output_text = "في انتظار الإشارة..."

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # معالجة الفريم بـ Mediapipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_holistic.process(img_rgb)
        
        # استخراج النقاط (1530 نقطة)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        keypoints = np.concatenate([face, lh, rh])
        
        # Sliding Window
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:]
        
        if len(self.sequence) == 30:
            res = model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
            if res[np.argmax(res)] > 0.8: # Threshold
                self.output_text = actions[np.argmax(res)]
        
        # كتابة النص على الفيديو
        cv2.putText(img, self.output_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        return img

# --- تشغيل البث في الموقع ---
webrtc_streamer(key="wesal-stream", video_transformer_factory=SignLanguageTransformer)

st.write("developed by dalia , tala")