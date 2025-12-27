import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import kalmanFilter

class Track:
    def __init__(self,bbox,track_id):
        self.kf=kalmanFilter(dim_x=7, dim_z=4)
        self.kf.F=np.eye(7)
        self.kf.H=np.eye(4, 7)
        self.kf.x[:4] = bbox.reshape((4,1))
        self.id= track_id
        self.hits = 1
        self.no_losses = 0

    def predict(self):
        self.kf.predict()
        return self.kf.x[:4]

    def update(self, bbox):
        self.kf.update(bbox)
        self.hits += 1
        self.no_losses = 0


class DeepSORT:
    def __init__(self):
        self.tracks = []
        self.track_id = 0

    def update(self, detections):
        updated_tracks = []
        for det in detections:
            matched = False
            for track in self.tracks:
                track.update(det)
                updated_tracks.append(track)
                matched = True
                break
            if not matched:
                self.track_id += 1
                updated_tracks.append(Track(det, self.track_id))

        self.tracks = updated_tracks
        return self.tracks
        
        
        