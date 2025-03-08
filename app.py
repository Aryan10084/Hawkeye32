from flask import Flask, render_template, Response
import cv2
import urllib.request
import numpy as np

app = Flask(__name__)

# ESP32-CAM URL
url = 'http://192.168.72.34/cam-hi.jpg'

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

def generate_frames():
    while True:
        try:
            imgResponse = urllib.request.urlopen(url)
            imgNp = np.array(bytearray(imgResponse.read()), dtype=np.uint8)
            img = cv2.imdecode(imgNp, -1)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            
            # Object detection
            classIds, confs, bbox = net.detect(img, confThreshold=0.5)
            if len(classIds) != 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
                    cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 30), 
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            
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
