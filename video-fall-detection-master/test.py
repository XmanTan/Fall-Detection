import numpy as np
import cv2 as cv

img = cv.imread('tzj.jpg')
fgbg = cv.createBackgroundSubtractorMOG2() # Help BG
while(1):
    ret,thresh = cv.threshold(img,127,255,0)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray) #Remove background
    contours, _ = cv.findContours(fgmask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # source img, contour retrivial mode, contour approximation method
    if contours:
        # List to hold all areas
        areas = []

        for contour in contours:
            ar = cv.contourArea(contour)
            areas.append(ar)

        max_area = max(areas, default = 0)

        max_area_index = areas.index(max_area)

        cnt = contours[max_area_index]

        M = cv.moments(cnt)

        x, y, w, h = cv.boundingRect(cnt)

        cv.drawContours(fgmask, [cnt], 0, (255,255,255), 3, maxLevel = 0)
        cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    cv.imshow('image',img)
    if cv.waitKey(33) & 0xFF == 27:
        break
cv.destroyAllWindows()