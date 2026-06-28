#!/usr/bin/env python

# Ejercicio RECTIFICACIÓN
#
# Deshace la deformación de perspectiva de un plano a partir de puntos de
# referencia con coordenadas reales conocidas, para medir distancias sobre
# ese plano marcando puntos con el ratón en la imagen ORIGINAL.
#
# El archivo de referencias (--ref) contiene una línea por punto:
#
#   px py X Y
#
# con (px,py) las coordenadas en píxeles del punto en la imagen y (X,Y) sus
# coordenadas reales en cm sobre el plano. Se necesitan al menos 4 puntos
# (no alineados) para calcular la homografía con cv.findHomography.
#
# Teclas:
#   click izq.  marca un punto (los dos últimos definen el segmento medido)
#   x           borra el último punto marcado
#   r           muestra/oculta la imagen rectificada (para comprobar la
#               homografía)
#   ESC         salir
#
# Uso:
#   ./rectifica.py --dev=../../images/coins.png --ref=coins.txt
#   ./rectifica.py --dev=mesa.jpg --ref=mesa.txt

import argparse
from collections import deque

import cv2 as cv
import numpy as np

from umucv.htrans import htrans, desp, scale
from umucv.util import putText

ap = argparse.ArgumentParser()
ap.add_argument('--dev', required=True, help="imagen a rectificar")
ap.add_argument('--ref', required=True, help="archivo de referencias: px py X Y por línea")
args, _ = ap.parse_known_args()

img = cv.imread(args.dev)

ref = np.loadtxt(args.ref)
imgpts = ref[:, :2]
worldpts = ref[:, 2:]

# H transforma puntos de la imagen a coordenadas reales (cm) sobre el plano
H, _ = cv.findHomography(imgpts, worldpts)

# transformación para previsualizar el plano rectificado: encaja el
# rectángulo que contiene los puntos de referencia en una imagen de tamaño
# manejable, con un margen alrededor
wpts = htrans(H, imgpts)
margin = 0.1 * max(wpts.max(0) - wpts.min(0))
spc = 600 / (wpts.max(0) - wpts.min(0) + 2 * margin).max()  # px / cm
T = scale([spc, spc]) @ desp(margin - wpts.min(0)) @ H
size = tuple((spc * (wpts.max(0) - wpts.min(0) + 2 * margin)).astype(int))

points = deque(maxlen=2)


def fun(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x, y))


cv.namedWindow("rectificacion")
cv.setMouseCallback("rectificacion", fun)

show_rectif = False

while True:
    frame = img.copy()

    for px, py in imgpts.astype(int):
        cv.drawMarker(frame, (px, py), (0, 255, 255), cv.MARKER_CROSS, 12, 1)

    for p in points:
        cv.circle(frame, p, 4, (0, 0, 255), -1)

    if len(points) == 2:
        cv.line(frame, points[0], points[1], (0, 0, 255), 1)
        w = htrans(H, np.array(points, dtype=float))
        d = np.linalg.norm(w[1] - w[0])
        c = tuple(np.mean(points, axis=0).astype(int))
        putText(frame, f'{d:.1f} cm', c)

    cv.imshow('rectificacion', frame)

    if show_rectif:
        rectif = cv.warpPerspective(img, T, size)
        cv.imshow('rectificada', rectif)

    key = cv.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('x') and points:
        points.pop()
    elif key == ord('r'):
        show_rectif = not show_rectif
        if not show_rectif:
            cv.destroyWindow('rectificada')

cv.destroyAllWindows()
