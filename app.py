import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
import os

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="وصال AI", page_icon="🤟", layout="centered")

st.title("Wesal-AI")
st.subheader("مترجم لغة الإشارة الفوري (نسخة الويب)")
st.markdown("---")

# --- 2. دالة بناء الموديل (مطابق لكودك السابق 1530) ---
@st.cache_resource # لتسريع الموقع وعدم تحميل الموديل في كل مرة
def load_wesal_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(30, 1530))) 
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(actions), activation='softmax'))
    
    # تأكدي من وضع ملف الـ h5 في نفس مجلد الكود
    weights_path = 'action_model.h5' 
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    return model

# --- 3. استخراج النقاط (Mediapipe) ---
mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([face, lh, rh])

# --- 4. تشغيل الواجهة ---
actions = np.array(['أريد مساعدة', 'بوابة الملك فهد', 'شكراً', 'دورة مياه', 'عفواً'])
model = load_wesal_model()

st.sidebar.info("هذا النظام يستخدم الذكاء الاصطناعي لترجمة الإشارات الحركية إلى نصوص وصوت.")
st.sidebar.image("https://img.icons8.com/fluency/100/sign-language.png")

# إطار الكاميرا
img_file_buffer = st.camera_input("وجه الكاميرا وقم بعمل الإشارة ثم التقط الصورة")

if img_file_buffer is not None:
    # تحويل الصورة الملتقطة
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # معالجة الصورة بـ Mediapipe
    with mp_holistic.Holistic(min_detection_confidence=0.5) as holistic:
        image_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        keypoints = extract_keypoints(results)
        
        # محاكاة الـ 30 فريم (لأن الموقع يأخذ صورة واحدة حالياً)
        # في النسخة المتقدمة يتم تجميع الفيديو
        sequence = [keypoints] * 30 
        
        # التوقع
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        detected_action = actions[np.argmax(res)]
        confidence = res[np.argmax(res)]

        # عرض النتيجة
        st.markdown(f"### النتيجة: **{detected_action}**")
        st.progress(float(confidence))
        st.write(f"نسبة التأكد: {confidence*100:.2f}%")
        
        if confidence > 0.7:
            st.success(f"تم التعرف على الإشارة: {detected_action}")
            # ملاحظة: النطق الصوتي يحتاج إعدادات إضافية في السيرفرات
        else:
            st.warning("الإشارة غير واضحة تماماً، حاول مرة أخرى.")