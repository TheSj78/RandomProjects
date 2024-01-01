from pprint import pprint
from fer import FER
import cv2
from fer.utils import draw_annotations
from fer.classes import Video
import asyncio

detector = FER()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # cap frame-by-frame
    _, frame = cap.read()

    emotions = detector.detect_emotions(frame)
    frame = draw_annotations(frame, emotions)

    # Display the resulting frame
    cv2.imshow("frame", frame)
    
    # asyncio.sleep(10)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
