from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app)

model = load_model("best_model.h5")
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)


def video_stream():
    while True:
        ret, frame = cap.read()
        if ret: 
            _, buffer = cv2.imencode('.jpg', frame)
            emit('video', buffer.tobytes())

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/connect')
def handleConnect():
    emit('video_settings', {'width': cap.get(3), 'height': cap.get(4)})
    socketio.start_background_task(video_stream)



@app.route('/analyze', methods=['POST'])
def analyze():

    if not cap.isOpened():
        return Response('Camera not available', status=500)

    pre_emotions = []

    while len(pre_emotions) < 20:
        ret, test_img = cap.read()
        if not ret:
            continue
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)

            # find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]
            
            pre_emotions.append(predicted_emotion)

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis ', resized_img)

    

        if cv2.waitKey(100) == ord('q'):  # wait until 'q' key is pressed
            break

    if len(pre_emotions) > 0:
       final_prediction = max(set(pre_emotions), key=pre_emotions.count)
    else:
       final_prediction = "Unknown"

    cap.release()
    cv2.destroyAllWindows()

    return f"Final Prediction: {final_prediction}"

    return jsonify({
        'status': 'success',
        'message': 'Analysis completed successfully.',
        'final_prediction': final_prediction })


if __name__ == "__main__":
    socketio.start_background_task(video_stream)
    socketio.run(app, debug=True)

    