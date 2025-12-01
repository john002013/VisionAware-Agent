"This is a computer vison tools for agentic AI"

from langchain.tools import Tool
import cv2
from ultralytics import YOLO
import time
import math
from collections import defaultdict
import subprocess



def main(_: str = "") -> str:
    model = YOLO("yolov8n.pt").to("cpu")
    classnames = model.names
    cap = cv2.VideoCapture('/dev/video0') 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output.mp4", fourcc, 10.0, (640, 480))

    object_counts = defaultdict(int)
    start_time = time.time()
    timeout = 40  # seconds

    while True:
        success, frame = cap.read()

        results = model(frame)
        frame_object = defaultdict(int)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classnames[cls]
                frame_object[currentClass] += 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{currentClass} {conf}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
        last_frame = frame_object
        out.write(frame)

        if time.time() - start_time > timeout:
            #time.sleep(2)
            break

    cap.release()
    out.release()

    time.sleep(1)
    subprocess.run(["ffplay", "-autoexit", "-loglevel", "quiet", "output.mp4"])


    if last_frame:
        for obj, count in last_frame.items():
            return f" {count} {obj}{'s' if count > 1 else ''}"
    else:
        print("No objects detected in the last frame.")




visual_tool = Tool(
    name = "visual_tool",
    func = main,
    description = "Opens webcam feed and uses YOLOv8n to detect and display real-time bounding boxes. Returns a list of all objects seen"
)

"Note: The video Captured was saved first before being displayed and this is due to some limitations as regarding the PC I used to program this but this can be modified to dsiplay while the model is running"
