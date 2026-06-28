#!/usr/bin/env python

import cv2   as cv
import numpy as np
from umucv.stream import mkStream
from umucv.util import putText
from os import system


def focus(v):
    cmd = f'v4l2-ctl -d /dev/video{DEV} -c focus_absolute={v}'
    system(cmd)

def exposure(v):
    cmd = f'v4l2-ctl -d /dev/video{DEV} -c exposure_time_absolute={v}'
    system(cmd)

def dev(v):
    global DEV
    DEV = v
    system(f'v4l2-ctl -d /dev/video{DEV} -c focus_automatic_continuous=0')
    system(f'v4l2-ctl -d /dev/video{DEV} -c auto_exposure=1')


D = 0

dev(D)

cv.namedWindow("control")
cv.createTrackbar("focus", "control", 0, 255, focus)
cv.createTrackbar("exposure", "control", 250, 2047, exposure)


while True:
    key = cv.waitKey(100) & 0xFF
    if key == 27 or key == ord('q'):
        break


    if ord("0") <= key <= ord("9"):
        D = key-ord("0")
        dev(D)

    img = np.zeros((70,500),np.uint8)
    putText(img,f"Camera {D}")
    cv.imshow("control", img)
