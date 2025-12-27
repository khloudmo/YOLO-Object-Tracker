import cv2

def draw_tracks(frame, tracks):
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.kf.x[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
            frame,
            f'ID {track.id}',
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )
    return frame
