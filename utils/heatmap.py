import numpy as np
import cv2

class Heatmap:
    def __init__(self, shape):
        self.map = np.zeros(shape[:2], dtype=np.float32)

    def update(self, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        self.map[cy, cx] += 1

    def draw(self, frame):
        heatmap = cv2.normalize(self.map, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = heatmap.astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
