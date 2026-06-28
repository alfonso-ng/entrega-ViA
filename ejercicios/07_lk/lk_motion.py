#!/usr/bin/env python

# Ampliación de lk_track.py: clasifica el movimiento de la cámara
# (LEFT/RIGHT/UP/DOWN/FORWARD/BACKWARD/ROTACION) a partir del flujo óptico.
# Tecla d: muestra/oculta las trayectorias.

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import putText
from collections import deque
import time

tracks = []
track_len = 20
detect_interval = 5

corners_params = dict( maxCorners = 500,
                       qualityLevel= 0.1,
                       minDistance = 10,
                       blockSize = 7)

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

MINFLOW = 0.5

show_tracks = True
prev_t = time.time()

for n, (key, frame) in enumerate(autoStream()):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    h, w = gray.shape
    center = np.array([w/2, h/2])
    R = np.hypot(w, h) / 2

    now = time.time()
    dt = now - prev_t
    prev_t = now
    fps = 1/dt if dt > 0 else 0

    info = "sin movimiento detectado"

    if tracks:
        p0 = np.float32( [t[-1] for t in tracks] )
        p1,  _, _ =  cv.calcOpticalFlowPyrLK(prevgray, gray, p0, None, **lk_params)
        p0r, _, _ =  cv.calcOpticalFlowPyrLK(gray, prevgray, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1,2).max(axis=1)
        good = d < 1

        new_tracks = []
        flows = []
        for t, p_old, p_new, ok in zip(tracks, p0, p1.reshape(-1,2), good):
            if not ok:
                continue
            t.append( p_new )
            new_tracks.append(t)
            flows.append((p_old, p_new - p_old))

        tracks = new_tracks

        if show_tracks:
            cv.polylines(frame, [ np.int32(t) for t in tracks ], isClosed=False, color=(0,0,255))
            for t in tracks:
                point = np.int32(t[-1])
                cv.circle(frame, center=point, radius=2, color=(0, 0, 255), thickness=-1)

        if len(flows) >= 10:
            p0s = np.array([p for p,_ in flows])
            vs  = np.array([v for _,v in flows])

            mean_flow = vs.mean(axis=0)

            r = p0s - center
            rn = r / np.linalg.norm(r, axis=1, keepdims=True).clip(1e-3)

            expansion = np.mean(np.sum(vs * rn, axis=1))

            cross = r[:,0]*vs[:,1] - r[:,1]*vs[:,0]
            r2 = np.sum(r*r, axis=1).clip(1.0)
            omega = np.mean(cross / r2)  # rad/frame

            mag_trans = np.linalg.norm(mean_flow)
            mag_exp   = abs(expansion)
            mag_rot   = abs(omega) * R

            mags = {'trans': mag_trans, 'exp': mag_exp, 'rot': mag_rot}
            best = max(mags, key=mags.get)

            if mags[best] < MINFLOW:
                info = "sin movimiento detectado"
            elif best == 'rot':
                deg_s = np.degrees(omega) * fps
                sentido = "horario" if omega > 0 else "antihorario"
                info = f"ROTACION ({sentido}, {abs(deg_s):.1f} grados/s)"
            elif best == 'exp':
                info = "FORWARD" if expansion > 0 else "BACKWARD"
            else:
                dx, dy = -mean_flow  # la camara se mueve al contrario que el fondo
                if abs(dx) > abs(dy):
                    info = "RIGHT" if dx > 0 else "LEFT"
                else:
                    info = "DOWN" if dy > 0 else "UP"

            cv.arrowedLine(frame, tuple(center.astype(int)),
                            tuple((center - mean_flow*10).astype(int)),
                            (0,255,255), 2)

    if n % detect_interval == 0:
        mask = np.zeros_like(gray)
        mask[:] = 255
        for x,y in [np.int32(t[-1]) for t in tracks]:
            cv.circle(mask, (x,y), 5, 0, -1)
        corners = cv.goodFeaturesToTrack(gray, mask=mask, **corners_params)
        if corners is not None:
            for [pt] in np.float32(corners):
                tracks.append( deque([pt], maxlen=track_len) )

    putText(frame, f'{len(tracks)} corners, {fps:.0f}fps', orig=(5,16))
    putText(frame, info, orig=(5,36))
    cv.imshow('input', frame)
    prevgray = gray

    if key == ord('d'):
        show_tracks = not show_tracks
