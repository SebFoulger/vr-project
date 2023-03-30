import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import pandas as pd
from matplotlib import pyplot as plt

import configparser

from plottable_objects import *

config = configparser.RawConfigParser()
config.read(filenames = './animation_config')

fps = 1000 # Frames per second for animation

skip_frames = 1 # We keep one in every skip_frames frames. i.e. if it's 3 we keep 1/3 frames. The idea is to be able to simulate faster without visibly losing data

delta = 0.01

total_frames = 10000

file_location = config.get('main', 'FILE_LOCATION')

df = pd.read_csv(file_location)[::skip_frames].reset_index()

time_passed=df[' timeExp']

def str2bool(v):
  return v.lower().strip() in ("yes", "true", "t", "1")

grabbed=np.vectorize(str2bool)(df['grabbed'])


fig = plt.figure()
ax = p3.Axes3D(fig)

plots = {}

plots["head_eye"] = Scatter(np.array([df['head_x'],df['eye_x']]),np.array([df['head_y'],df['eye_y']]),np.array([df['head_z'],df['eye_z']]),ax,".","--","head_eye")

plots["controller"] = Arrow(df[' controller_x'],df[' controller_y'],df[' controller_z'],df['controller_yaw'],df[' controller_pitch'],df[' controller_roll'],0.1,"black", "controller",ax,delta)

txt = fig.suptitle('')

def update_points(t):

    txt.set_text('time passed={:d} seconds'.format(int(time_passed[t])))

    plots["head_eye"].update(t)

    if grabbed[t]:
        plots["controller"].update(t,color="orange")
    else:
        plots["controller"].update(t,color="black")



ani=animation.FuncAnimation(fig, update_points, frames=total_frames, interval = int(1000/fps))

ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
