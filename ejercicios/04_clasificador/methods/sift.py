# SIFT para objetos con textura. Distancia = 1 / nº de coincidencias (ratio test).

import cv2 as cv
import numpy as np

NAME = "sift"

_sift = cv.SIFT_create(nfeatures=500)
_matcher = cv.BFMatcher()

RATIO = 0.75


def describe(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, des = _sift.detectAndCompute(gray, None)
    if des is None or len(des) < 2:
        return None
    return des


def _good_matches(d1, d2):
    matches = _matcher.knnMatch(d1, d2, k=2)
    good = 0
    for m in matches:
        if len(m) == 2:
            best, second = m
            if best.distance < RATIO * second.distance:
                good += 1
    return good


def distance(d1, d2):
    good = _good_matches(d1, d2)
    if good == 0:
        return float('inf')
    return 1 / good


def info(d, dist):
    good = round(1 / dist) if np.isfinite(dist) else 0
    return f"{good} coincidencias"
