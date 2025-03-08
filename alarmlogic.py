from flask import Flask, render_template, Response
import cv2
import urllib.request
import numpy as np
import pygame

app = Flask(__name__)

# ESP32-CAM URL
url = 'http://192.168.72.34/cam-hi.jpg'

# Initialize pygame mixer for playing alarm
pygame.mixer.init()
alarm_sound = "alarm.wav"

# Load class names
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

scissors_detected = False  # Track if scissors are in the frame

def play_alarm():
    """Plays the alarm sound."""
    pygame.mixer.music.load(alarm_sound)
    pygame.mixer.music.play()

def generate_frames():
    global scissors_detected
    while True:
        try:
            imgResponse = urllib.request.urlopen(url)
            imgNp = np.array(bytearray(imgResponse.read()), dtype=np.uint8)
            img = cv2.imdecode(imgNp, -1)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            # Object detection
            classIds, confs, bbox = net.detect(img, confThreshold=0.5)
            detected_scissors = False

            if len(classIds) != 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    label = classNames[classId - 1]

                    # Draw bounding box and label
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
                    cv2.putText(img, label, (box[0] + 10, box[1] + 30), 
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                    # Check if scissors are detected
                    if label.lower() == "scissors":
                        detected_scissors = True
                        if not scissors_detected:
                            play_alarm()  # Play alarm only once when scissors appear
                        scissors_detected = True

            # If scissors are detected, overlay a translucent red effect
            if scissors_detected:
                overlay = img.copy()
                red_filter = np.zeros_like(img, dtype=np.uint8)
                red_filter[:, :, 2] = 150  # Red intensity
                cv2.addWeighted(red_filter, 0.5, overlay, 0.5, 0, overlay)
                img = overlay

            # If no scissors are detected in the current frame, stop the alarm
            if not detected_scissors and scissors_detected:
                pygame.mixer.music.stop()
                scissors_detected = False

            _, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
