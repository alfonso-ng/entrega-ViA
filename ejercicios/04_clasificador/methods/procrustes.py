# Gestos de mano con landmarks de mediapipe, centrados y normalizados.
# Distancia = residuo de Procrustes ortogonal (invariante a tamaño y rotación).

import cv2 as cv
import numpy as np
import mediapipe as mp

NAME = "procrustes"

_hands = mp.solutions.hands.Hands(max_num_hands=1)


def describe(frame):
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = _hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    lm = result.multi_hand_landmarks[0]
    pts = np.array([[p.x, p.y] for p in lm.landmark])
    pts = pts - pts.mean(axis=0)
    norm = np.linalg.norm(pts)
    if norm < 1e-9:
        return None
    return pts / norm


def distance(d1, d2):
    u, _, vt = np.linalg.svd(d1.T @ d2)
    r = u @ vt
    return float(np.linalg.norm(d1 @ r - d2))


def info(d, dist):
    return f"residuo {dist:.3f}"
