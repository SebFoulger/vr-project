import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import pandas as pd
from matplotlib import pyplot as plt
import os
import git
import time

from plottable_objects import *

os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

class AniObject:
    """
    Object for animation runs of the task
    """    
    def __init__(self, df: pd.DataFrame, subject_no: int, week_no: int, session_no: int):
        """Initialiser function for class

        Args:
            df (pd.DataFrame): dataframe with data for animation.
        """        
        self.df = df.reset_index(drop=True).copy()

        self.df['head_y'], self.df['head_z'] = self.df['head_z'], self.df['head_y']
        self.df['controller_y'], self.df['controller_z'] = self.df['controller_z'], self.df['controller_y']
        self.df['hit_y'], self.df['hit_z'] = self.df['hit_z'], self.df['hit_y']

        self.subject_no = subject_no

        self.session_no = session_no

        self.week_no = week_no

        practice_bool = df['practice'][0]=='practice'

        window_title='Subject: '+str(self.subject_no)+' | Week: '+str(self.week_no)+' | Session: '+str(self.session_no)
        window_title+=' | Practice: '+str(practice_bool)

        self.fig = plt.figure(window_title)

        self.ax = p3.Axes3D(self.fig)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        #self.ax.view_init(elev=96, azim=-90)

    """
    Public Functions
    """

    def animate(self, fps: int = 10, skip_frames: int = 1, fps_bool: bool = False, 
                speed: int = 1, arrow_length: float = 0.3):
        """Function to call for animation.

        Args:
            fps (int, optional): frames per second for animation. Defaults to 10.
            skip_frames (int, optional): one in every skip_frames frames are kept. Only used in fps_bool is True. 
            Defaults to 1.
            fps_bool (bool, optional): set to True if speed of animation should be based on FPS. Set to False if speed
            should be in accordance with real time. Defaults to False.
            speed (int, optional): speed of animation multiplier. Only used if fps_bool is False. Defaults to 1.
            arrow_length (float, optional): length of arrow. Defaults to 0.3.
        """        

        delta = 0.01

        ani_df = self.df[::skip_frames].reset_index(drop=True).copy()
        total_frames = len(ani_df)

        frame_time = ani_df['timeExp'].diff()
        ani_df['frame_time'] = frame_time

        if fps_bool:
            _index = (frame_time.notnull()) & (frame_time<skip_frames*0.02)
            summand = ani_df[_index]['frame_time']
            avg_frame_interval = 1000*sum(summand)/total_frames

            interval = round(avg_frame_interval/speed)
        else:
            interval = int(1000/fps)

        self.time_passed=ani_df['timeExp']

        is_grabbing = lambda v: v.lower().strip() in ('yes', 'true', 't', '1')

        self.grabbed=np.vectorize(is_grabbing)(ani_df['grabbed'])

        self.is_target=ani_df['isTarget']

        self.plots = {}

        self.plots['head_eye'] = Scatter(xss=np.array([ani_df['head_x'],ani_df['hit_x']]),
                                         yss=np.array([ani_df['head_y'],ani_df['hit_y']]),
                                         zss=np.array([ani_df['head_z'],ani_df['hit_z']]),
                                         ax=self.ax,init_marker='.',init_linestyle='--',init_label='head_eye')


        self.plots['controller'] = Arrow(xs=ani_df['controller_x'],ys=ani_df['controller_y'],
                                         zs=ani_df['controller_z'],yaws=ani_df['controller_yaw'],
                                         pitches=ani_df['controller_pitch'],rolls=ani_df['controller_roll'],
                                         init_length=0.1, init_color='black', init_label='controller',
                                         ax=self.ax, delta=delta)
        
        workspace_x_mid, workspace_y_mid, workspace_z_mid = -0.05, 2, -2.25
        resource_x_mid, resource_y_mid, resource_z_mid = 1, -0.618, -0.188
        model_x_mid, model_y_mid, model_z_mid = -0.841,-1.621, 0

        self.plots['workspace'] = self.ax.plot_surface(*self._generate_plane(workspace_x_mid,
                                    workspace_y_mid,workspace_z_mid,dimensions = (1,1), orientation='z'), color='green')

        self.plots['resource'] = self.ax.plot_surface(*self._generate_plane(resource_x_mid,
                                    resource_y_mid,resource_z_mid,dimensions = (1,1), orientation='z'), color='red')
        
        self.plots['model'] = self.ax.plot_surface(*self._generate_plane(model_x_mid,
                                    model_y_mid,model_z_mid,dimensions = (1,1), orientation='y'), color='blue')


        max_x = max(max(ani_df['head_x']),max(ani_df['hit_x']),max(ani_df['controller_x']))
        max_y = max(max(ani_df['head_y']),max(ani_df['hit_y']),max(ani_df['controller_y']))
        max_z = max(max(ani_df['head_z']),max(ani_df['hit_z']),max(ani_df['controller_z']))

        min_x = min(min(ani_df['head_x']),min(ani_df['hit_x']),min(ani_df['controller_x']))
        min_y = min(min(ani_df['head_y']),min(ani_df['hit_y']),min(ani_df['controller_y']))
        min_z = min(min(ani_df['head_z']),min(ani_df['hit_z']),min(ani_df['controller_z']))

        self.txt = self.fig.suptitle('', x=0.25)

        self.fig.canvas.mpl_connect('key_press_event', self._on_press)
        self.start = time.time()
        self.start1 = self.time_passed[0]

        self.ani=animation.FuncAnimation(self.fig, self._update_animation, frames=total_frames, interval = interval)
        self.ani.running = True

        self.ax.legend()
        self.ax.set_xlim(min_x,max_x)
        self.ax.set_ylim(min_y,max_y)
        self.ax.set_zlim(min_z,max_z)

        plt.show()

    """
    Private Functions
    """

    def _update_animation(self,t: int):
        """Function to call for animation update

        Args:
            t (int): frame of animation
        """        
        self.txt.set_text(f'''study time={int(self.time_passed[t])} seconds,
                            actual time passed={int(time.time()-self.start)}''')

        self.plots['head_eye'].update(t)

        if self.grabbed[t] and self.is_target[t]:
            color = 'orange'
        elif self.grabbed[t]:
            color = 'red'
        else:
            color = 'black'

        self.plots['controller'].update(t,color=color)


    def _on_press(self,event):   
        if event.key.isspace():
            if self.ani.running:
                self.ani.event_source.stop()
            else:
                self.ani.event_source.start()
            self.ani.running ^= True

    def _generate_plane(self, x_mid: float, y_mid: float, z_mid: float, dimensions: tuple, orientation: str):
        """Function to generate coordintes for plane plotting

        Args:
            x_mid (float): middle x-coordinate.
            y_mid (float): middle y-coordinate.
            z_mid (float): middle z-coordinate.
            dimensions (tuple): 2-tuple with dimensions for the two non-flat directions.
            orientation (str): the flat coordinate, i.e. if orientation='z' then the plane will be flat in the 'z'
            plane.

        Returns:
            tuple: x, y, z coordinates for plane.
        """        
        if orientation=='x':
            one_mid=y_mid
            two_mid=z_mid
            three_mid=x_mid
        elif orientation=='y':
            one_mid=x_mid
            two_mid=z_mid
            three_mid=y_mid
        else:
            one_mid=x_mid
            two_mid=y_mid
            three_mid=z_mid
        one = [[one_mid-dimensions[0]/2,one_mid+dimensions[0]/2],[one_mid-dimensions[0]/2,one_mid+dimensions[0]/2]]

        two = [[two_mid-dimensions[1]/2,two_mid-dimensions[1]/2],[two_mid+dimensions[1]/2,two_mid+dimensions[1]/2]]

        three = [[three_mid,three_mid],[three_mid,three_mid]]
        
        if orientation=='x':
            return three,one,np.array(two)
        elif orientation=='y':
            return one,three,np.array(two)
        else:
            return one,two,np.array(three)

repo = git.Repo('.', search_parent_directories=True)
repo.working_tree_dir

file_location = repo.working_tree_dir+'/input_data/raw/1_0_1.csv'
df = pd.read_csv(file_location)

obj = AniObject(df, 1, 1, 1)
obj.animate(fps = 90, fps_bool=True, skip_frames=2)
