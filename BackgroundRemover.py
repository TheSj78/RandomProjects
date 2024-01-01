import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
segmentor = SelfiSegmentation()

while True:
    ret, frame = cap.read()
    
    imgOut = segmentor.removeBG(frame, (255, 255, 255), threshold=0.8) #edit threshold parameter to change

    imgStacked = cvzone.stackImages([frame, imgOut], 2, 1)
    cv2.imshow("Stacked", imgStacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
