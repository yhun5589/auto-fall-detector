import cv2
import math
from ultralytics import YOLO

# Load YOLOv12 model
model = YOLO('yolo12n.pt')

# Load class names
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

def detect(frame):
    info = {}  # store all detected objects
    results = model(frame, verbose=False)
    fall = False
    new_frame = frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_detect = classnames[class_id]
            conf = math.ceil(confidence * 100)

            if conf > 60 and class_detect in ['person', 'sofa', 'bed', 'chair']:
                width, height = x2 - x1, y2 - y1
                new_frame = frame
                # store detections
                if class_detect not in info:
                    info[class_detect] = []
                if class_detect != "person":
                    info[class_detect].append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "width": width, "height": height
                    })
                if class_detect == 'person':
                    ratio = height / width
                    if ratio < 1.2:
                        info[class_detect].append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "width": width, "height": height
                        })
                        fall = True
                        new_frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                new_frame = frame
    return fall, info, new_frame


# 🧠 NEW FUNCTION
def check_person_on_object(info):
    fall_on_floor = False
    if 'person' not in info:
        print("❌ No person detected.")
        return False

    for person in info['person']:
        px1, py1, px2, py2 = person['x1'], person['y1'], person['x2'], person['y2']
        pw, ph = person['width'], person['height']

        on_any_object = False
        fall_on_floor = True
        for obj_name in ['bed', 'sofa', 'chair']:
            if obj_name in info:
                for obj in info[obj_name]:
                    ox1, oy1, ox2, oy2 = obj['x1'], obj['y1'], obj['x2'], obj['y2']
                    ow, oh = obj['width'], obj['height']

                    # --- Check vertical alignment (bottom of person near top of object) ---
                    horizontal_condition= px1,px2 >= oy1 and px1,px2 <= ox2 * 1.2  # allow small margin

                    # --- Check horizontal alignment (person within object width) ---
                    vertical_condition = py1 >= oy2*0.8 and py1 < oy2 * 1.2# small margin

                    if vertical_condition and horizontal_condition:
                        on_any_object = True
                        fall_on_floor = False
                        break

    return fall_on_floor


if __name__ == "__main__":
    image = cv2.imread('hq720.jpg')
    image = cv2.resize(image, (640, 480))
    fall, detections, new_frame = detect(image)
    cv2.imshow("Detection", new_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Detections:", detections)
    print("Fall detected:", fall)
    status = check_person_on_object(detections)
    print("Status:", status)
