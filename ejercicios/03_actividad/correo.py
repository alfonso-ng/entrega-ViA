#!/usr/bin/env python

# Envío por email del vídeo de un evento detectado por actividad.py.
#
# La configuración (servidor SMTP, credenciales y destinatario) se lee de
# correo.env, que no se versiona (está en .gitignore como *.env). Copia
# correo.env.example a correo.env y rellena tus datos.

import os
import smtplib
from email.message import EmailMessage


def _load_env(path):
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'correo.env'))


def send_event_video(video_path, categories, timestamp):
    host = os.environ['SMTP_HOST']
    port = int(os.environ.get('SMTP_PORT', 465))
    user = os.environ['SMTP_USER']
    password = os.environ['SMTP_PASS']
    to = os.environ['NOTIFY_TO']

    msg = EmailMessage()
    msg['Subject'] = f"Actividad detectada: {', '.join(sorted(categories))}"
    msg['From'] = user
    msg['To'] = to
    msg.set_content(
        f"Evento detectado el {timestamp.isoformat()}.\n"
        f"Categorías: {', '.join(sorted(categories))}")

    with open(video_path, 'rb') as f:
        msg.add_attachment(f.read(), maintype='video', subtype='mp4',
                            filename=os.path.basename(video_path))

    with smtplib.SMTP_SSL(host, port) as s:
        s.login(user, password)
        s.send_message(msg)
