#!/usr/bin/env python

# Detector de movimiento en ROI marcada con el ratón. Cuando hay movimiento
# sostenido graba un vídeo (con preroll de 1s). Clasifica con EfficientDet-Lite;
# si aparece alguna categoría de --target manda un correo. Anónimiza caras.
# Registra cada evento en actividad.csv.
#
# Uso: ./actividad.py --dev=0 --target=bird,cat
# Teclas: c (borrar ROI), m (máscara)

import argparse
import csv
import datetime
import os
import time
from collections import deque

import numpy as np
import cv2 as cv

from umucv.stream import autoStream
from umucv.util import putText, ROI, check_and_download

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import correo

ap = argparse.ArgumentParser()
ap.add_argument('--target', default='cell phone',
                help="categorías de interés separadas por comas (disparan la notificación)")
ap.add_argument('--conf', type=float, default=0.5,
                help="confianza mínima del detector de objetos")
args, _ = ap.parse_known_args()
TARGETS = {c.strip() for c in args.target.split(',') if c.strip()}

WARMUP_FRAMES  = 30    # frames iniciales para que MOG2 aprenda el fondo
MOTION_FRAC    = 0.01  # fracción mínima del área de la ROI en movimiento
MOTION_FRAMES  = 3     # frames consecutivos con movimiento para abrir evento
PREROLL_SEC    = 1.0   # segundos de preroll
IDLE_SEC       = 1.0   # segundos sin movimiento para cerrar el evento
MAX_EVENT_SEC  = 6.0   # duración máxima de seguridad
DETECT_EVERY   = 5     # cada cuántos frames se llama al clasificador

LOGFILE = 'actividad.csv'

check_and_download(
    "efficientdet.tflite",
    "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite")

check_and_download(
    "detector.tflite",
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite")

object_detector = vision.ObjectDetector.create_from_options(
    vision.ObjectDetectorOptions(
        base_options=python.BaseOptions(model_asset_path='efficientdet.tflite'),
        score_threshold=args.conf))

face_detector = vision.FaceDetector.create_from_options(
    vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path='detector.tflite')))


def detect_categories(crop):
    if crop.size == 0:
        return set()
    mpimage = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(crop, cv.COLOR_BGR2RGB))
    result = object_detector.detect(mpimage)
    return {d.categories[0].category_name for d in result.detections}


def anonymize(frame):
    mpimage = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    result = face_detector.detect(mpimage)
    for d in result.detections:
        b = d.bounding_box
        x, y, w, h = b.origin_x, b.origin_y, b.width, b.height
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)
    return frame


newfile = not os.path.exists(LOGFILE)
log = open(LOGFILE, 'a', newline='')
logwriter = csv.writer(log)
if newfile:
    logwriter.writerow(['timestamp', 'categorias', 'video', 'notificado'])


def finalize_event(frames, categories, fps_now):
    ts = datetime.datetime.now()

    if 'person' in categories:
        frames = [anonymize(f.copy()) for f in frames]

    fname = ts.strftime("%Y%m%d-%H%M%S") + "_evento.mp4"
    h, w = frames[0].shape[:2]
    out = cv.VideoWriter(fname, cv.VideoWriter_fourcc(*'mp4v'), max(1, fps_now), (w, h))
    for f in frames:
        out.write(f)
    out.release()

    notified = False
    if categories & TARGETS:
        try:
            correo.send_event_video(fname, categories, ts)
            notified = True
        except Exception as e:
            print("No se pudo enviar la notificación:", e)

    logwriter.writerow([ts.isoformat(), ' '.join(sorted(categories)), fname, notified])
    log.flush()
    print(f"evento guardado en {fname}, categorías: {sorted(categories)}, notificado: {notified}")


bgsub = cv.createBackgroundSubtractorMOG2(500, 16, False)
openkernel = np.ones((3, 3), np.uint8)
dilatekernel = np.ones((9, 9), np.uint8)

roi = ROI('actividad')

times = deque(maxlen=30)
preroll = deque(maxlen=120)

state = 'IDLE'
frame_count = 0
motion_run = 0
idle_run = 0
event_frames = []
event_categories = set()

show_mask = False

for key, frame in autoStream():
    times.append(time.time())
    frame_count += 1

    if len(times) >= 2:
        deltas = np.diff(times)
        fps = 1 / np.mean(deltas) if np.mean(deltas) > 0 else 15
    else:
        fps = 15

    if key == ord('m'):
        show_mask = not show_mask

    fgmask = bgsub.apply(frame)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, openkernel, iterations=1)
    fgmask = cv.dilate(fgmask, dilatekernel, iterations=1)

    if key == ord('c'):
        roi.roi = []
        state = 'IDLE'
        motion_run = 0
        preroll.clear()

    have_roi = roi.roi and (roi.roi[2] - roi.roi[0]) * (roi.roi[3] - roi.roi[1]) >= 100

    if not have_roi:
        putText(frame, "arrastra el raton para marcar la ROI ('c' para borrarla)")
    else:
        x1, y1, x2, y2 = roi.roi
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        if frame_count > WARMUP_FRAMES:
            roimask = fgmask[y1:y2 + 1, x1:x2 + 1]
            roiarea = (x2 - x1 + 1) * (y2 - y1 + 1)
            moving = cv.countNonZero(roimask) > MOTION_FRAC * roiarea

            if state == 'IDLE':
                preroll.append(frame.copy())
                motion_run = motion_run + 1 if moving else 0
                if motion_run >= MOTION_FRAMES:
                    state = 'EVENT'
                    npre = max(1, round(fps * PREROLL_SEC))
                    event_frames = list(preroll)[-npre:]
                    event_categories = set()
                    idle_run = 0
                    motion_run = 0

            elif state == 'EVENT':
                event_frames.append(frame.copy())
                idle_run = 0 if moving else idle_run + 1

                if len(event_frames) % DETECT_EVERY == 0:
                    crop = frame[y1:y2 + 1, x1:x2 + 1]
                    event_categories |= detect_categories(crop)

                idle_frames = round(fps * IDLE_SEC)
                max_frames = round(fps * (MAX_EVENT_SEC + PREROLL_SEC))
                if idle_run >= idle_frames or len(event_frames) >= max_frames:
                    finalize_event(event_frames, event_categories, fps)
                    state = 'IDLE'
                    preroll.clear()
                    motion_run = 0

    if state == 'EVENT':
        cv.circle(frame, (15, 15), 6, (0, 0, 255), -1)

    cv.imshow('actividad', frame)
    if show_mask:
        cv.imshow('mask', fgmask)

log.close()
cv.destroyAllWindows()
