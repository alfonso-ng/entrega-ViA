# Detección del tablero de sudoku en una imagen y lectura de sus dígitos.
#
# find_grid(img)   -> las 4 esquinas del contorno cuadrado más grande de la
#                      imagen (o None si no se encuentra), en el orden
#                      [arriba, derecha, abajo, izquierda] empezando por la
#                      esquina más alta de la imagen.
# read_board(warp) -> matriz 9x9 con los dígitos reconocidos (0 = vacía)

import os
from itertools import combinations

os.environ.setdefault("KERAS_BACKEND", "torch")

import cv2 as cv
import numpy as np
import keras

from umucv.util import check_and_download

check_and_download(
    os.path.join(os.path.dirname(__file__), "digits.keras"),
    "https://umubox.um.es/index.php/s/ge3h1OdotGtx6RQ/download")

_model = keras.models.load_model(os.path.join(os.path.dirname(__file__), "digits.keras"))

N = 450       # tamaño del tablero rectificado
CELL = N // 9


def _signed_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return np.sum(x*np.roll(y, -1) - np.roll(x, -1)*y)


def _order_corners(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    if _signed_area(pts) < 0:        # aseguramos un sentido de recorrido fijo
        pts = pts[::-1]
    i0 = np.argmin(pts[:, 1])        # empezamos por la esquina más alta
    return np.roll(pts, -i0, axis=0)


def threshold_debug(img, blockSize=25):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    g = cv.GaussianBlur(gray, (5, 5), 0)
    return cv.adaptiveThreshold(g, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, blockSize, 4)


def _squarest_quad(pts):
    n = len(pts)
    best = None
    for idx in combinations(range(n), 4):
        quad = pts[list(idx)]
        if not cv.isContourConvex(quad.reshape(4, 1, 2).astype(np.int32)):
            continue
        sides = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
        ratio = sides.max() / sides.min()
        if best is None or ratio < best[0]:
            best = (ratio, quad)
    return best


def _grid_score(img, corners):
    Hm = grid_homography(corners)
    warp = cv.warpPerspective(img, Hm, (N, N))
    b = binarize(warp)
    rows = np.sum(b.mean(axis=1)/255 > 0.4)
    cols = np.sum(b.mean(axis=0)/255 > 0.4)
    return rows + cols


def find_grid(img, minfrac=0.05):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    g = cv.GaussianBlur(gray, (5, 5), 0)
    H, W = gray.shape

    candidates = []
    for blockSize in (11, 15, 25, 35):
        th = cv.adaptiveThreshold(g, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, blockSize, 4)
        cs, _ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cs:
            area = cv.contourArea(c)
            if area < minfrac*W*H or area > 0.9*W*H:
                continue
            if best is not None and area <= best[0]:
                continue
            peri = cv.arcLength(c, True)
            for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
                approx = cv.approxPolyDP(c, eps*peri, True)
                if len(approx) < 4 or len(approx) > 8:
                    continue
                if len(approx) == 4 and not cv.isContourConvex(approx):
                    continue
                pts = approx.reshape(-1, 2).astype(np.float32)
                if len(pts) == 4:
                    sides = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
                    ratio = sides.max() / sides.min()
                    quad = pts
                else:
                    # más de 4 vértices: buscamos el subconjunto más cuadrado
                    found = _squarest_quad(pts)
                    if found is None:
                        break
                    ratio, quad = found
                quad_area = abs(cv.contourArea(quad))
                # cuadrado: lados similares y sin encogerse demasiado
                if ratio < 1.5 and quad_area > 0.5*area:
                    best = (quad_area, quad.reshape(4, 1, 2))
                break

        if best is not None:
            candidates.append(_order_corners(best[1]))

    if not candidates:
        return None

    return max(candidates, key=lambda corners: _grid_score(img, corners))


def grid_homography(corners):
    dst = np.array([[0, 0], [N, 0], [N, N], [0, N]], dtype=np.float32)
    return cv.getPerspectiveTransform(corners, dst)


def _adaptsize(x):
    h, w = x.shape
    s = max(h, w)
    h2, w2 = (s-h)//2, (s-w)//2
    y = x
    if w2 > 0:
        z1 = np.zeros([s, w2], np.uint8)
        z2 = np.zeros([s, s-w-w2], np.uint8)
        y = np.hstack([z1, x, z2])
    if h2 > 0:
        z1 = np.zeros([h2, s], np.uint8)
        z2 = np.zeros([s-h-h2, s], np.uint8)
        y = np.vstack([z1, x, z2])
    y = cv.resize(y, (20, 20))
    M = cv.moments(y)
    mx, my = (M['m10']/M['m00'], M['m01']/M['m00']) if M['m00'] > 0 else (10, 10)
    A = np.array([[1., 0, 4-(mx-9.5)], [0, 1, 4-(my-9.5)]])
    return cv.warpAffine(y, A, (28, 28)) / 255


# margen por lado para no coger las líneas de la rejilla
MARGIN = CELL // 6
MINAREA = 0.03 * (CELL - 2*MARGIN)**2


def _extract_digit(cell_bin):
    n, cc, st, _ = cv.connectedComponentsWithStats(cell_bin)
    best = None
    for i in range(1, n):
        area = st[i][cv.CC_STAT_AREA]
        if area < MINAREA:
            continue
        if best is None or area > st[best][cv.CC_STAT_AREA]:
            best = i
    if best is None:
        return None
    x1, y1, w, h = st[best][cv.CC_STAT_LEFT], st[best][cv.CC_STAT_TOP], st[best][cv.CC_STAT_WIDTH], st[best][cv.CC_STAT_HEIGHT]
    return (cc[y1:y1+h, x1:x1+w] == best).astype(np.uint8) * 255


def binarize(warp):
    # adaptativo sobre el tablero completo, más estable que Otsu celda a celda
    gray = cv.cvtColor(warp, cv.COLOR_BGR2GRAY)
    return cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 25, 10)


def read_board(warp):
    b = binarize(warp)

    board = np.zeros((9, 9), int)
    digits = []
    positions = []
    for r in range(9):
        for c in range(9):
            cell = b[r*CELL+MARGIN:(r+1)*CELL-MARGIN, c*CELL+MARGIN:(c+1)*CELL-MARGIN]
            digit = _extract_digit(cell)
            if digit is not None:
                digits.append(_adaptsize(digit))
                positions.append((r, c))
    if digits:
        x = np.array(digits).reshape(-1, 28, 28, 1)
        p = _model.predict(x, verbose=False)
        labels = np.argmax(p, axis=1)
        for (r, c), label in zip(positions, labels):
            board[r, c] = label
    return board
