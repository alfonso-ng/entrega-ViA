#!/usr/bin/env python

# Apartado 3 de calibración: cuadrícula de medida sobre la imagen de la cámara.
# Sliders: fov (campo visual), Z (distancia al plano, cm), A (altura cámara, dm), X (offset cm).
# El fov inicial se lee de calib.txt si existe.

import numpy as np
import cv2   as cv

from umucv.stream import autoStream
from umucv.htrans  import Kfov
from umucv.util    import Slider, putText, showCalib, lineType

CELL = 50   # tamaño de la celda de la cuadrícula, en cm (0.5 m)
NX   = 10   # celdas a cada lado del eje óptico
NY   = 10   # celdas desde el suelo hacia arriba

try:
    K0 = np.loadtxt('calib.txt')[:9].reshape(3,3)
    w0, h0 = 2*K0[0,2], 2*K0[1,2]
    fov0 = int(round(2*np.degrees(np.arctan2(w0/2, K0[0,0]))))
except (OSError, ValueError):
    fov0 = 60

stream = autoStream()
HEIGHT, WIDTH = next(stream)[1].shape[:2]
size = WIDTH, HEIGHT

fovsl = Slider('fov', 'grid', fov0, start=0, stop=80,  step=1)
zsl   = Slider('Z',   'grid', 100,  start=0, stop=290, step=1)
asl   = Slider('A',   'grid', 8,    start=0, stop=20,  step=1)
xsl   = Slider('X',   'grid', 0,    start=0, stop=100, step=1)


def project(K, X, Y, Z):
    x, y, _ = K @ np.array([X, Y, Z])
    return int(round(x/Z)), int(round(y/Z))


for key, frame in stream:
    fov = fovsl.value
    Z   = max(zsl.value, CELL)
    H   = asl.value * 10
    Xoff = xsl.value

    K = Kfov(size, fov)

    for gx in range(-NX, NX+1):
        Xw = gx*CELL + Xoff
        p1 = project(K, Xw, H,           Z)
        p2 = project(K, Xw, H-NY*CELL,   Z)
        cv.line(frame, p1, p2, (0,255,255), 1, lineType)

    for gy in range(0, NY+1):
        Yw = H - gy*CELL
        p1 = project(K, -NX*CELL+Xoff, Yw, Z)
        p2 = project(K,  NX*CELL+Xoff, Yw, Z)
        thick = 2 if gy == 0 else 1
        cv.line(frame, p1, p2, (0,255,255), thick, lineType)
        px,py = project(K, Xoff, Yw, Z)
        putText(frame, f'{gy*CELL/100:.1f}m', (px+2,py-2), color=(0,255,255), div=1)

    # cuadrícula del plano del suelo (Y = H), en perspectiva hacia la cámara
    Zfar  = Z
    Znear = max(Z - NY*CELL, CELL)

    for gx in range(-NX, NX+1):
        Xw = gx*CELL + Xoff
        p1 = project(K, Xw, H, Znear)
        p2 = project(K, Xw, H, Zfar)
        cv.line(frame, p1, p2, (0,255,255), 1, lineType)

    for gy in range(0, NY+1):
        Zd = Zfar - gy*(Zfar-Znear)/NY
        p1 = project(K, -NX*CELL+Xoff, H, Zd)
        p2 = project(K,  NX*CELL+Xoff, H, Zd)
        thick = 2 if gy == 0 else 1
        cv.line(frame, p1, p2, (0,255,255), thick, lineType)

    showCalib(K, frame)
    putText(frame, f'fov={fov}  Z={Z}cm  A={asl.value} (H={H}cm)  X={Xoff}cm')
    cv.imshow('grid', frame)
