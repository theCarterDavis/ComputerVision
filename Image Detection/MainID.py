import cv2
import numpy as np
from functions import getSpotsBB, emptyOrNot

#Method to calculate the difference betwene two images using their numpy arrays
def calcDiff(img1, img2):
    return np.abs(np.mean((img1) - np.mean(img2)))


path = "/Users/carterdavis/PycharmProjects/opencvpractive/Image Detection/Data/parking_1920_1080_loop.mp4"#video path
masksP = "/Users/carterdavis/PycharmProjects/opencvpractive/Image Detection/Data/Parking Space Counter Mask 1920x1080.png"#masks path
cap = cv2.VideoCapture(path)
masks = cv2.imread(masksP,0)
ret = True

#using connected compants to use the masks to map onto the spots
connectedComponants = cv2.connectedComponentsWithStats(masks, 4, cv2.CV_32S)

spots = getSpotsBB(connectedComponants)

spotsStatus = [None for j in spots]
diffs = [None for j in spots]
prevFrame = None

frameNum = 0
while ret:
    ret, frame = cap.read()

    #calling the calculate differneces method to save wheather a spot has changed since the last time that it was looked at
    if frameNum % 30 == 0 and prevFrame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot

            spotCrop = frame[y1:y1 + h, x1:x1 + w, :]

            diffs[spot_indx] = calcDiff(spotCrop, prevFrame[y1:y1 + h, x1:x1 + w, :])

    #looping through the detected differences once envery thirtyframes to update wheather the sopt is open now and saving the result
    if frameNum % 30 == 0:
        if prevFrame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
        for spotIndex in arr_:
            spot = spots[spotIndex]
            x1,y1,w,h = spot

            spotCrop = frame[y1:y1+h,x1:x1+w]

            spotStatus = emptyOrNot(spotCrop)
            spotsStatus[spotIndex] = spotStatus

    #saving the frames every 30 frames so that they can be compared to check for differnences
    if frameNum % 30 == 0:
        prevFrame = frame.copy()

    #drawing a green rect around the spots that are open, and red around the taken
    for spotIndx, spot in enumerate(spots):
        spot_status = spotsStatus[spotIndx]
        x1, y1, w, h = spots[spotIndx]

        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    frameNum += 1

    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)#Black box to put the text counter on
    #Summing the number of total and availab le spots and displaying the number of each
    cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(spotsStatus)), str(len(spotsStatus))), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('frame', frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
