import cv2
#capturing the video feef from my laptop
cam = cv2.VideoCapture(0)#0 is the default camera

while True:
    ret, frame = cam.read()
    cv2.imshow('Webcam', frame)
    #Looping the campera feed/displaying it untill and interupt or q is peressed
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()