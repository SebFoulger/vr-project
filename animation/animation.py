import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import pandas as pd
from matplotlib import pyplot as plt
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git

from plottable_objects import *

repo = git.Repo('.', search_parent_directories=True)
repo.working_tree_dir

fps = 1000 # Frames per second for animation

skip_frames = 1 # We keep one in every skip_frames frames. i.e. if it's 3 we keep 1/3 frames. The idea is to be able to simulate faster without visibly losing data

delta = 0.01

file_location = repo.working_tree_dir+"/input_data/test1.csv"

df = pd.read_csv(file_location)[::skip_frames].sort_values(by=['frame']).reset_index()

total_frames = len(df)

time_passed=df[' timeExp']

def str2bool(v):
  return v.lower().strip() in ("yes", "true", "t", "1")

grabbed=np.vectorize(str2bool)(df['grabbed'])

is_target=df[' isTarget']

subject_no = df[' sub'][0]

session_no = df[' session'][0]

practice_bool = True if df['practice'][0]=='practice' else False

window_title="Subject: "+str(subject_no)+" | Session: "+str(session_no)+" | Practice: "+str(practice_bool)

fig = plt.figure(window_title)
ax = p3.Axes3D(fig)

plots = {}

plots["head_eye"] = Scatter(np.array([df['head_x'],df['eye_x']]),np.array([df['head_y'],df['eye_y']]),np.array([df['head_z'],df['eye_z']]),ax,".","--","head_eye")

plots["controller"] = Arrow(df[' controller_x'],df[' controller_y'],df[' controller_z'],df['controller_yaw'],df[' controller_pitch'],df[' controller_roll'],0.1,"black", "controller",ax,delta)

max_x, min_x, max_y, min_y, max_z, min_z = 0, 0, 0, 0, 0, 0

max_x = max(max(df['head_x']),max(df['eye_x']),max(df[' controller_x']))
max_y = max(max(df['head_y']),max(df['eye_y']),max(df[' controller_y']))
max_z = max(max(df['head_z']),max(df['eye_z']),max(df[' controller_z']))

min_x = min(min(df['head_x']),min(df['eye_x']),min(df[' controller_x']))
min_y = min(min(df['head_y']),min(df['eye_y']),min(df[' controller_y']))
min_z = min(min(df['head_z']),min(df['eye_z']),min(df[' controller_z']))

txt = fig.suptitle('')

def update_points(t):

    txt.set_text('time passed={:d} seconds'.format(int(time_passed[t])))

    plots["head_eye"].update(t)

    if grabbed[t] and is_target[t]:
        plots["controller"].update(t,color="orange")
    elif grabbed[t]:
        plots["controller"].update(t,color="red")
    else:
        plots["controller"].update(t,color="black")


def on_press(event):
    if event.key.isspace():
        if ani.running:
            ani.event_source.stop()
        else:
            ani.event_source.start()
        ani.running ^= True

fig.canvas.mpl_connect('key_press_event', on_press)
ani=animation.FuncAnimation(fig, update_points, frames=total_frames, interval = int(1000/fps))
ani.running = True

ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_xlim(min_x,max_x)
ax.set_ylim(min_y,max_y)
ax.set_zlim(min_z,max_z)
plt.show()
