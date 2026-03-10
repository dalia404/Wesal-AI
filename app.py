from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
import os

app = Flask(__name__)
CORS(app) # ضروري للسماح للمتصفح بالاتصال بالسيرفر

# 1. إعدادات الموديل
actions = np.array(['HELP', 'TOILET', 'KF_GATE', 'THANKS', 'WELCOME'])

def build_and_load_model(path, num_classes):
    model = Sequential()
    model.add(InputLayer(input_shape=(30, 1530)))
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # تأكدي من صحة المسار هنا
    if os.path.exists(path):
        model.load_weights(path)
    return model

# تحميل الموديل مرة واحدة عند التشغيل
model = build_and_load_model("wesal-ai-ui/action_model.h5", 5)

# 2. إعدادات MediaPipe (خارج الدالة لضمان الاستقرار)
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=True, 
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=False
)

sequence = []

def extract_keypoints(results):
    # استخراج النقاط مع التأكد من تعبئة الأصفار إذا لم توجد يد (للحفاظ على حجم 1530)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([face, lh, rh])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global sequence
    try:
        file = request.files.get("frame")
        if not file:
            return jsonify({"error": "No frame received"}), 400

        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # معالجة الصورة
        results = holistic.process(image)
        keypoints = extract_keypoints(results)

        sequence.append(keypoints)
        sequence = sequence[-30:] # الاحتفاظ بآخر 30 إطار

        if len(sequence) == 30:
            # استخدمي هذا:
            input_data = tf.convert_to_tensor(np.expand_dims(sequence, axis=0), dtype=tf.float32)
            res = model(input_data, training=False).numpy()[0]
            if res[np.argmax(res)] > 0.5:
                 action = actions[np.argmax(res)]
                 return jsonify({"prediction": action})

        return jsonify({"prediction": "جارِ التحليل..."})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
  
   port = int(os.environ.get("PORT",5000))
   app.run(host='0.0.0.0', port=port)




