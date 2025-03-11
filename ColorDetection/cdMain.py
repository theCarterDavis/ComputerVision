import cv2
from Utils import getLimits
from PIL import Image

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()

    hsvF = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # input a bgr color into function
    color = [225,0,0]
    ll, ul =getLimits(color)

    mask = cv2.inRange(hsvF, ll, ul)

    masks = Image.fromarray(mask)

    bbox = masks.getbbox()

    print(bbox)

    if bbox is not None:
        x1,y1,x2,y2 = bbox
        frame = cv2.rectangle(frame, (x1 , y1), (x2 , y2), color, 2)
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()