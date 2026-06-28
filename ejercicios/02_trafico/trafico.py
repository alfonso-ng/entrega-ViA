#!/usr/bin/env python

# Cuenta vehículos cruzando una línea vertical (sentido R/L).
# MOG2 + apertura + dilatación + findContours. MINAREA bajo porque la cámara está lejos.
# Un cruce solo cuenta si el track lleva MINHITS frames y se desplazó MINMOVE px
# (filtra ruido que parpadea sin avanzar). ROI con sliders para ignorar pájaros, etc.
# Registra los cruces en trafico.csv.
#
# Uso: ./trafico.py --dev=carretera   Tecla m: máscara

import datetime
import os

import numpy as np
import cv2   as cv

from umucv.stream import autoStream
from umucv.util    import putText, Slider

MINAREA = 300    # px²
MAXDIST = 60     # px
MAXAGE  = 10     # frames
MINHITS = 5      # frames seguidos para confirmar
MINMOVE = 20     # px mínimos en el sentido del cruce

LOGFILE = 'trafico.csv'

bgsub = cv.createBackgroundSubtractorMOG2(500, 16, False)
openkernel  = np.ones((3,3), np.uint8)
dilatekernel = np.ones((5,5), np.uint8)

tracks = []
count  = {'R':0, 'L':0}

newfile = not os.path.exists(LOGFILE)
log = open(LOGFILE, 'a')
if newfile:
    log.write('timestamp,sentido\n')

show_mask = False

stream = autoStream()
HEIGHT, WIDTH = next(stream)[1].shape[:2]

x0sl = Slider('x0', 'trafico', 0,      start=0, stop=WIDTH,  step=1)
x1sl = Slider('x1', 'trafico', WIDTH,  start=0, stop=WIDTH,  step=1)
y0sl = Slider('y0', 'trafico', 0,      start=0, stop=HEIGHT, step=1)
y1sl = Slider('y1', 'trafico', HEIGHT, start=0, stop=HEIGHT, step=1)

for key, frame in stream:
    if key == ord('m'):
        show_mask = not show_mask

    H, W = frame.shape[:2]
    line_x = W // 2

    x0, x1 = sorted((x0sl.value, x1sl.value))
    y0, y1 = sorted((y0sl.value, y1sl.value))

    fgmask = bgsub.apply(frame)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, openkernel, iterations=1)
    fgmask = cv.dilate(fgmask, dilatekernel, iterations=2)

    roi = np.zeros_like(fgmask)
    roi[y0:y1, x0:x1] = 255
    fgmask = cv.bitwise_and(fgmask, roi)

    contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    detections = []
    for c in contours:
        if cv.contourArea(c) < MINAREA:
            continue
        x,y,w,h = cv.boundingRect(c)
        detections.append((x+w/2, y+h/2))
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 1)

    used = [False]*len(detections)
    for t in tracks:
        best, bestd = None, MAXDIST
        for i,(cx,cy) in enumerate(detections):
            if used[i]:
                continue
            d = np.hypot(cx-t['pos'][0], cy-t['pos'][1])
            if d < bestd:
                best, bestd = i, d
        if best is None:
            t['age'] += 1
            continue
        used[best] = True
        newpos = detections[best]
        t['hits'] += 1
        if not t['counted'] and t['hits'] >= MINHITS:
            moved = newpos[0] - t['start_x']
            if t['pos'][0] < line_x <= newpos[0] and moved >= MINMOVE:
                count['R'] += 1
                t['counted'] = True
                log.write(f'{datetime.datetime.now().isoformat()},R\n')
                log.flush()
            elif t['pos'][0] > line_x >= newpos[0] and -moved >= MINMOVE:
                count['L'] += 1
                t['counted'] = True
                log.write(f'{datetime.datetime.now().isoformat()},L\n')
                log.flush()
        t['pos'] = newpos
        t['age'] = 0

    for i,(cx,cy) in enumerate(detections):
        if not used[i]:
            tracks.append({'pos':(cx,cy), 'start_x':cx, 'age':0, 'hits':1, 'counted':False})

    tracks = [t for t in tracks if t['age'] <= MAXAGE]

    cv.rectangle(frame, (x0,y0), (x1,y1), (255,0,0), 1)
    cv.line(frame, (line_x,max(y0,0)), (line_x,min(y1,H)), (0,0,255), 2)
    putText(frame, f'-> {count["R"]}   <- {count["L"]}')
    cv.imshow('trafico', frame)
    if show_mask:
        cv.imshow('mask', fgmask)

log.close()
cv.destroyAllWindows()
