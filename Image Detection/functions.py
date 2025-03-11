import pickle

from skimage.transform import resize
import numpy as np
import cv2



#Pre trained model used to detect wheather a parking space is ocupied or not
MODEL = pickle.load(open("model.p", "rb"))
def getSpotsBB(componants):
    (totalLables, labelID, values, centroid) = componants

    slots = []
    coef = 1
    for i in range(1,totalLables):

        x1 = int(values[i,cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i,cv2.CC_STAT_TOP] * coef)
        w = int(values[i,cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i,cv2.CC_STAT_HEIGHT] * coef)

        slots.append((x1,y1,w,h))
    return slots

def emptyOrNot(spot_bgr):

    flat_data = []

    img_resized = resize(spot_bgr, (15, 15, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    y_output = MODEL.predict(flat_data)

    if y_output == 0:
        return True
    else:
        return False


