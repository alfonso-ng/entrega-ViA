#!/usr/bin/env python

# Compara cada fotograma con imágenes "modelo" de un directorio usando un
# método intercambiable (embedding/sift/procrustes en methods/).
# Los descriptores se cachean en .pkl. Tecla c: nuevo modelo.
#
# Uso: ./clasificador.py --models=<dir> --method=<método>

import argparse
import glob
import importlib
import os
import pickle

import cv2 as cv

from umucv.stream import autoStream
from umucv.util import putText

ap = argparse.ArgumentParser()
ap.add_argument('--models', default='models', help="directorio con las imágenes modelo")
ap.add_argument('--method', default='embedding', help="módulo de methods/ a utilizar")
ap.add_argument('--top', type=int, default=5, help="número de modelos a mostrar en el ranking")
args, _ = ap.parse_known_args()

method = importlib.import_module(f'methods.{args.method}')

os.makedirs(args.models, exist_ok=True)

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')


def cache_path(imgfile):
    base, _ = os.path.splitext(imgfile)
    return f"{base}.{args.method}.pkl"


def load_descriptor(imgfile):
    cfile = cache_path(imgfile)
    if os.path.exists(cfile) and os.path.getmtime(cfile) >= os.path.getmtime(imgfile):
        with open(cfile, 'rb') as f:
            return pickle.load(f)
    img = cv.imread(imgfile)
    desc = method.describe(img) if img is not None else None
    with open(cfile, 'wb') as f:
        pickle.dump(desc, f)
    return desc


def load_models():
    models = []
    for imgfile in sorted(glob.glob(os.path.join(args.models, '*'))):
        if os.path.splitext(imgfile)[1].lower() not in IMG_EXTS:
            continue
        desc = load_descriptor(imgfile)
        if desc is None:
            print(f"aviso: no se ha podido calcular el descriptor de {imgfile}")
            continue
        label = os.path.splitext(os.path.basename(imgfile))[0]
        models.append((label, desc))
    return models


models = load_models()
print(f"método: {method.NAME}   modelos cargados: {[m[0] for m in models]}")


for key, frame in autoStream():

    desc = method.describe(frame)

    if key == ord('c'):
        name = input("nombre del nuevo modelo: ").strip()
        if name:
            imgfile = os.path.join(args.models, name + '.png')
            cv.imwrite(imgfile, frame)
            d = method.describe(frame)
            with open(cache_path(imgfile), 'wb') as f:
                pickle.dump(d, f)
            if d is None:
                print(f"aviso: modelo '{name}' guardado pero sin descriptor calculable")
            else:
                models.append((name, d))
                print(f"modelo '{name}' guardado en {imgfile}")
        else:
            print("nombre vacío, no se ha guardado nada")

    y = 20
    if desc is None:
        putText(frame, f"[{method.NAME}] sin descriptor en este fotograma", orig=(5, y))
    elif not models:
        putText(frame, f"[{method.NAME}] sin modelos: pulsa 'c' para añadir uno", orig=(5, y))
    else:
        ranking = sorted(((method.distance(desc, d), label) for label, d in models))
        best_dist, best_label = ranking[0]
        putText(frame, f"[{method.NAME}] mejor: {best_label}  ({method.info(desc, best_dist)})", orig=(5, y))
        for dist, label in ranking[:args.top]:
            y += 18
            putText(frame, f"  {label}: {dist:.3f}  {method.info(desc, dist)}", orig=(5, y))

    cv.imshow('clasificador', frame)
