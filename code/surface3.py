import numpy as np
import cv2 as cv
from umucv.stream import autoStream

from vispy import app, scene
from vispy import app
from vispy.app import Timer

# Sacamos las imágenes del stream con next.
# Necesitamos una al principio para mirar su tamaño.

stream = autoStream()
_, img = next(stream)
data = img[:,:,1]/255
# Grid dimensions
rows, cols = data.shape
x = np.arange(cols)/cols*2
y = np.arange(rows)/cols*2
x, y = np.meshgrid(x, y[::-1])

# Create vertices
z = data  # Use data for heights
vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

# Create faces
faces = []
for i in range(rows - 1):
    for j in range(cols - 1):
        v0 = i * cols + j
        v1 = v0 + 1
        v2 = (i + 1) * cols + j
        v3 = v2 + 1
        faces.append([v0, v2, v3])
        faces.append([v0, v3, v1])
faces = np.array(faces)

# Create mesh
mesh = scene.visuals.Mesh(vertices=vertices, faces=faces, color=(.5, .7, .5, 1))

# admite imagen monocroma entre 0 y 1
def update_surface(frame):
    vertices[:,2] = z = frame.flatten()
    colors = np.vstack([z,z,z]).T
    mesh.set_data(vertices = vertices, faces=faces, vertex_colors=colors)

def capture(timer_event):
    _, frame = next(stream)
    result = work(frame)
    update_surface(result)

tim = Timer("0.03", start=True, connect=capture)

canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
view = canvas.central_widget.add_view()
view.add(mesh)
view.camera = 'turntable'

@canvas.events.key_press.connect
def on_key_press(event):
    if event.key == 'Q':
        exit(0)

########################################################################

# simplemente mostramos la imagen
def work(frame):
    return frame[:,:,1]/255

########################################################################

app.run()




