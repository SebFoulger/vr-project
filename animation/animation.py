import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import pandas as pd
from matplotlib import pyplot as plt
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git
import time

from plottable_objects import *

class AniObject:
    def __init__(self, df):
        self.df = df.sort_values(by=['frame']).reset_index(drop=True)

        self.subject_no = df[' sub'][0]

        self.session_no = df[' session'][0]

        practice_bool = True if df['practice'][0]=='practice' else False

        window_title="Subject: "+str(self.subject_no)+" | Session: "+str(self.session_no)
        window_title+=" | Practice: "+str(practice_bool)

        self.fig = plt.figure(window_title)

        self.ax = p3.Axes3D(self.fig)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        

    def animate(self, fps=10, skip_frames = 1, fps_bool = False, speed=1, arrow_length=0.3):

        delta = 0.01

        ani_df = self.df[::skip_frames].reset_index(drop=True)
        total_frames = len(ani_df)

        ani_df['frame_time'] = ani_df[' timeExp'].diff()

        avg_frame_interval = 1000*sum(ani_df[(ani_df['frame_time'].notnull()) 
                                        & (ani_df['frame_time']<skip_frames*0.02)]['frame_time'])/total_frames

        interval = round(avg_frame_interval/speed)

        self.time_passed=ani_df[' timeExp']

        self.grabbed=np.vectorize(lambda v: v.lower().strip() in ("yes", "true", "t", "1"))(ani_df['grabbed'])

        self.is_target=ani_df[' isTarget']

        self.plots = {}

        self.plots["head_eye"] = Scatter(xss=np.array([ani_df['head_x'],ani_df['eye_x']]),
                                         yss=np.array([ani_df['head_y'],ani_df['eye_y']]),
                                         zss=np.array([ani_df['head_z'],ani_df['eye_z']]),
                                         ax=self.ax,init_marker=".",init_linestyle="--",init_label="head_eye")

        self.plots["controller"] = Arrow(xs=ani_df[' controller_x'],ys=ani_df[' controller_y'],
                                         zs=ani_df[' controller_z'],yaws=ani_df['controller_yaw'],
                                         pitches=ani_df[' controller_pitch'],rolls=ani_df[' controller_roll'],
                                         init_length=arrow_length,init_color="black", init_label="controller",
                                         ax=self.ax,delta=delta)
        
        max_x = max(max(ani_df['head_x']),max(ani_df['eye_x']),max(ani_df[' controller_x']))
        max_y = max(max(ani_df['head_y']),max(ani_df['eye_y']),max(ani_df[' controller_y']))
        max_z = max(max(ani_df['head_z']),max(ani_df['eye_z']),max(ani_df[' controller_z']))

        min_x = min(min(ani_df['head_x']),min(ani_df['eye_x']),min(ani_df[' controller_x']))
        min_y = min(min(ani_df['head_y']),min(ani_df['eye_y']),min(ani_df[' controller_y']))
        min_z = min(min(ani_df['head_z']),min(ani_df['eye_z']),min(ani_df[' controller_z']))

        self.txt = self.fig.suptitle('')

        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.start = time.time()
        self.start1 = self.time_passed[0]
        if fps_bool:
            self.ani=animation.FuncAnimation(self.fig, self.update_points, frames=total_frames, 
                                                                                            interval = int(1000/fps))
        else:
            self.ani=animation.FuncAnimation(self.fig, self.update_points, frames=total_frames, interval = interval)
        self.ani.running = True

        self.ax.legend()
        self.ax.set_xlim(min_x,max_x)
        self.ax.set_ylim(min_y,max_y)
        self.ax.set_zlim(min_z,max_z)
        plt.show()

    def update_points(self,t):
        
        self.txt.set_text(f'''time passed={int(self.time_passed[t])} seconds,
                            actual time passed={int(time.time()-self.start)}''')

        self.plots["head_eye"].update(t)

        if self.grabbed[t] and self.is_target[t]:
            self.plots["controller"].update(t,color="orange")
        elif self.grabbed[t]:
            self.plots["controller"].update(t,color="red")
        else:
            self.plots["controller"].update(t,color="black")


    def on_press(self,event):
        if event.key.isspace():
            if self.ani.running:
                self.ani.event_source.stop()
            else:
                self.ani.event_source.start()
            self.ani.running ^= True

repo = git.Repo('.', search_parent_directories=True)
repo.working_tree_dir

file_location = repo.working_tree_dir+"/input_data/test1.csv"
df = pd.read_csv(file_location)

obj = AniObject(df=df)
obj.animate()
