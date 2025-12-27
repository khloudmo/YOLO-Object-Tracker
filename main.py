import cv2
import torch
from tracker.deepsort import DeepSORT
from utils.visualization import draw_tracks
from utils.heatmap import Heatmap




# Load YOLOv5

device = torch.device('cpu') 
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4

cap = cv2.VideoCapture('data/video.mp4')

tracker = DeepSORT()
heatmap = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if heatmap is None:
        heatmap = Heatmap(frame.shape)

    results = model(frame)
    detections = []

    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        detections.append([x1, y1, x2, y2])
        heatmap.update([x1, y1, x2, y2])

    tracks = tracker.update(detections)
    frame = draw_tracks(frame, tracks)
    frame = heatmap.draw(frame)

    cv2.imshow("YOLO Object Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
