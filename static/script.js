document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('video');
    const mainTranslation = document.getElementById('main-translation');
    const historyContainer = document.getElementById('history-container');
    const statusDisplay = document.getElementById('status');
    let lastPrediction = ""; // لمنع التكرار المزعج لنفس الكلمة

    // 1. تشغيل الكاميرا
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => { 
            video.srcObject = stream;
            statusDisplay.innerText = "الكاميرا متصلة - ابدأ الإشارة";
        })
        .catch(err => {
            statusDisplay.innerText = "خطأ: لم يتم الوصول للكاميرا";
            console.error(err);
        });

    // 2. دالة لإرسال الصور للسيرفر
    async function sendFrame() {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('frame', blob);

            try {
                const response = await fetch('http://127.0.0.1:5000/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                if (data.prediction && data.prediction !== "..." && data.prediction !== lastPrediction) {
                    updateUI(data.prediction);
                    lastPrediction = data.prediction;
                }
            } catch (error) {
                console.log("السيرفر غير متصل حالياً");
            }
        }, 'image/jpeg');
    }

    // 3. تحديث الواجهة وإضافة فقاعة كلام
    function updateUI(text) {
        // تحديث النص الكبير في الأعلى
        mainTranslation.innerText = text;

        // إضافة فقاعة جديدة للسجل
        const now = new Date();
        const timeStr = now.getHours() + ":" + (now.getMinutes() < 10 ? '0' : '') + now.getMinutes();

        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble current-bubble';
        bubble.innerHTML = `
            <span class="time-stamp">${timeStr}</span>
            <p>${text}</p>
        `;

        historyContainer.prepend(bubble); // إضافة الأحدث في الأعلى
    }
    // إرسال صورة كل نصف ثانية
    setInterval(sendFrame, 500);
});