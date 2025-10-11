import cv2
from flask import Flask, render_template, Response
from demo_detector import detect, check_person_on_object
import time

app = Flask(__name__)

fall_detected_time = None  # stores when fall first happened

def gen_flame():
    cap = cv2.VideoCapture("fall.mp4")
    global fall_detected_time

    while True:
        success, frame = cap.read()
        if not success:
            break

        fall, info, new_frame = detect(frame)

        # If a fall is detected
        if fall:
            if fall_detected_time is None:
                fall_detected_time = time.time()  # mark the start time
        else:
            fall_detected_time = None  # reset if no fall

        # Check if person has remained fallen for 5 seconds
        if fall_detected_time is not None:
            elapsed = time.time() - fall_detected_time
            if elapsed >= 5:
                print("⚠️ Person still fallen after 5 seconds!")
                # optional: trigger alert once and reset timer
                fall_detected_time = None

        ret, buffer = cv2.imencode('.jpg', new_frame)
        new_frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + new_frame + b'\r\n')
            
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_flame(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)