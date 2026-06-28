# Embedding con mobilenet_v3_small de mediapipe. Distancia = 1 - similitud coseno
# (embeddings normalizados, así que la similitud es el producto escalar).

import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from umucv.util import check_and_download

NAME = "embedding"

check_and_download(
    "embedder.tflite",
    "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite")

_embedder = vision.ImageEmbedder.create_from_options(
    vision.ImageEmbedderOptions(
        base_options=python.BaseOptions(model_asset_path='embedder.tflite'),
        l2_normalize=True, quantize=False))


def describe(frame):
    mpimage = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    return np.array(_embedder.embed(mpimage).embeddings[0].embedding)


def distance(d1, d2):
    return float(1 - d1 @ d2)


def info(d, dist):
    return f"similitud {1 - dist:.3f}"
