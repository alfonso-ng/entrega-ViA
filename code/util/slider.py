#!/usr/bin/env python

import cv2   as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import Slider, putText


cv.namedWindow("binary")
h = Slider("umbral", "binary", 0.5, 0, 1, 0.02)

for key, frame in autoStream():
    cv.imshow("original", frame)
    # convertimos a niveles de gris cogiendo el canal G
    # y lo ponemos en modo float (entre cero y 1)
    gray = frame[:,:,1] / 255
    cv.imshow("gray",gray)
    
    mask = (gray > h.value).astype(float)
    
    putText(mask, f"umbral = {h.value:.2f}")
    cv.imshow('binary', mask )
