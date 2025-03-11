import cv2
import mediapipe as mp
#capturing video feed
cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    #Message snet if the feed is not being recieved
    if not ret or img is None:
        print("Failed to capture image from webcam")
        continue  # Skip the rest of the loop and try agai

    #Flipping the immage to look mirroed
    img = cv2.flip(img, 1)

    H, W, x = img.shape
    #using mediapipe to locate the faces that may be in the immage
    face = mp.solutions.face_detection

    # Model Selection 0 is for close up, 1 is for far away
    with face.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = face_detection.process(img_rgb)

        if out.detections is not None:
            for detection in out.detections:#looping through each detection made
                loc = detection.location_data

                bb = loc.relative_bounding_box
                #Retreving the bounding boxes that are fromthe detections that are returned
                x1, y1, w, h = bb.xmin, bb.ymin, bb.width, bb.height

                #getting the locations so that the fave can be blured propery
                x1 = int(x1 * W)
                y1 = int(y1 * H)
                w = int(W * w)
                h = int(H * h)
                # img = cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,255,0),2)

                #applying a bluring affect to the bounding boxes that were retrieved
                img[y1:y1 + h, x1:x1 + w] = cv2.blur(img[y1:y1 + h, x1:x1 + w], (35, 35))

    cv2.imshow('Webcam', img)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()