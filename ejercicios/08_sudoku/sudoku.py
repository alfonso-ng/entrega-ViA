#!/usr/bin/env python

# Detecta el tablero de un sudoku en vivo, lee los dígitos con una CNN,
# resuelve el sudoku y dibuja la solución en perspectiva sobre la imagen.
# Teclas: g (tablero rectificado + binarización), t (umbral de detección)

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import putText

import reader
from solver import solve

N = reader.N
CELL = reader.CELL

STABLE_FRAMES = 1

last_board = None
stable_board = None
stable_count = 0
solution = None

show_grid = False
show_thresh = False

for key, frame in autoStream():

    if show_thresh:
        cv.imshow('umbral', reader.threshold_debug(frame))

    corners = reader.find_grid(frame)
    info = "tablero no detectado"

    if corners is not None:
        Hm = reader.grid_homography(corners)
        warp = cv.warpPerspective(frame, Hm, (N, N))
        board = reader.read_board(warp)

        if np.array_equal(board, last_board):
            stable_count += 1
        else:
            stable_count = 1
        last_board = board

        if stable_count == STABLE_FRAMES and not np.array_equal(board, stable_board):
            stable_board = board
            # mínimo 17 pistas para evitar backtracking sobre tablero casi vacío
            if (board != 0).sum() >= 17:
                solution = solve(board)
            else:
                solution = None

        if solution is None:
            info = f"{(board != 0).sum()} dígitos leídos, sin solución"
        elif not np.array_equal(board, stable_board):
            info = "tablero cambiando..."
        else:
            info = "resuelto"

            canvas = np.zeros((N, N, 3), np.uint8)
            for r in range(9):
                for c in range(9):
                    if board[r, c] == 0:
                        x = c*CELL + CELL//4
                        y = r*CELL + 3*CELL//4
                        cv.putText(canvas, str(solution[r, c]), (x, y),
                                   cv.FONT_HERSHEY_SIMPLEX, CELL/40, (0, 255, 0), 2, cv.LINE_AA)

            invH = np.linalg.inv(Hm)
            overlay = cv.warpPerspective(canvas, invH, (frame.shape[1], frame.shape[0]))
            mask = overlay.any(axis=2)
            frame[mask] = overlay[mask]

            if show_grid:
                for r in range(9):
                    for c in range(9):
                        if board[r, c] == 0:
                            x = c*CELL + CELL//4
                            y = r*CELL + 3*CELL//4
                            cv.putText(warp, str(solution[r, c]), (x, y),
                                       cv.FONT_HERSHEY_SIMPLEX, CELL/40, (0, 255, 0), 2, cv.LINE_AA)

        cv.polylines(frame, [corners.astype(int)], True, (0, 0, 255), 2)

        if show_grid:
            cv.imshow('tablero', warp)
            cv.imshow('binario', reader.binarize(warp))
    elif show_grid:
        cv.destroyWindow('tablero')
        cv.destroyWindow('binario')

    putText(frame, info, orig=(5, 16))
    cv.imshow('sudoku', frame)

    if key == ord('g'):
        show_grid = not show_grid
        if not show_grid:
            cv.destroyWindow('tablero')
            cv.destroyWindow('binario')

    if key == ord('t'):
        show_thresh = not show_thresh
        if not show_thresh:
            cv.destroyWindow('umbral')
