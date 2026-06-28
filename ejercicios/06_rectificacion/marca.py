#!/usr/bin/env python

# Herramienta auxiliar para crear el archivo de referencias que necesita
# rectifica.py.
#
# Muestra la imagen y permite marcar puntos con el ratón sobre un objeto de
# medidas conocidas (p.ej. las esquinas de la caja de un juego, una hoja de
# papel, baldosas, etc.). Por cada click se pide por consola las coordenadas
# reales (X,Y) en cm de ese punto, y se va guardando la línea
#
#   px py X Y
#
# en el archivo indicado en --out. Se necesitan al menos 4 puntos no
# alineados.
#
# Teclas:
#   click izq.  marca un punto y pide sus coordenadas reales por consola
#   x           borra el último punto marcado
#   ESC         termina y guarda --out
#
# Uso:
#   ./marca.py --dev=20260613-182406.png --out=mesa.txt

import argparse

import cv2 as cv
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--dev', required=True, help="imagen sobre la que marcar los puntos")
ap.add_argument('--out', required=True, help="archivo de referencias a generar (px py X Y)")
args, _ = ap.parse_known_args()

img = cv.imread(args.dev)

points = []  # [(px,py,X,Y), ...]


def fun(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(f"punto en ({x},{y}) -> coordenadas reales 'X Y' (cm): ", end='', flush=True)
        try:
            X, Y = map(float, input().split())
        except (ValueError, EOFError):
            print("entrada no válida, punto descartado")
            return
        points.append((x, y, X, Y))


cv.namedWindow("marca")
cv.setMouseCallback("marca", fun)

while True:
    frame = img.copy()
    for px, py, X, Y in points:
        cv.drawMarker(frame, (px, py), (0, 255, 255), cv.MARKER_CROSS, 12, 1)
        cv.putText(frame, f"({X:g},{Y:g})", (px + 6, py - 6),
                    cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv.LINE_AA)

    cv.imshow('marca', frame)

    key = cv.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('x') and points:
        points.pop()

cv.destroyAllWindows()

if points:
    np.savetxt(args.out, points, fmt='%g')
    print(f"guardados {len(points)} puntos en {args.out}")
else:
    print("ningún punto marcado, no se ha guardado nada")
