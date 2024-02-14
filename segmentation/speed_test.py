import git
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
from linear_approach import LinearSegmentation
from summarize_segments import summarize

def plot_prediction(time: pd.Series,
                    y: pd.Series,
                    cut_time: float = 1,
                    plot_breaks: bool = False,
                    beta_bool: bool = True,
                    sig_level: float = 0.01,
                    break_line_color: str = 'black',
                    prediction_line_color: str = 'orange',
                    prediction_line_label: str = 'prediction',
                    init_segment_size: int = 10,
                    window_size: int = 10,
                    step: int = 1,
                    force_left_intersection: bool = False,
                    force_right_intersection: bool = False):
    all_breakpoints = []
    prev_break = 0
    for _break in list(time.loc[time.diff()>cut_time].index)+[len(time)]:
        cur_time = time[prev_break:_break-1]
        cur_y = y[prev_break:_break-1]
        
        segmentation = LinearSegmentation(x=cur_time,y=cur_y)
        predictions, breakpoints = segmentation.segment(init_segment_size=init_segment_size, 
                                                        window_size=window_size,
                                                        step=step,
                                                        sig_level=sig_level,
                                                        beta_bool=beta_bool,
                                                        force_left_intersection=force_left_intersection,
                                                        force_right_intersection=force_right_intersection)
        all_breakpoints += [prev_break]+list(map(lambda x: x+prev_break,breakpoints))+[prev_break+len(predictions)]
        if plot_breaks:
            for small_break in breakpoints:
                plt.plot([time[small_break],time[small_break]],[min(y),max(y)], color=break_line_color)
        plt.plot(time[prev_break:prev_break+len(predictions)],predictions, color=prediction_line_color, label=prediction_line_label)
        prev_break = _break 
    return all_breakpoints

repo = git.Repo('.', search_parent_directories=True)
repo.working_tree_dir

file_location = repo.working_tree_dir+'/input_data/test1.csv'
df = pd.read_csv(file_location)[:5000]

df=df[['frame',' timeExp','head_x','head_y','head_z',' controller_x',' controller_y',' controller_z']]
df=df.rename(columns={' controller_x': 'controller_x',' controller_y': 'controller_y',' controller_z': 'controller_z',
                      ' timeExp': 'time'})
time_exp = df['time'][1:]

df_diff = df.diff().dropna()

df_diff['head_dist'] = np.sqrt(df_diff['head_x']**2+df_diff['head_y']**2+df_diff['head_z']**2)
df_diff['controller_dist'] = np.sqrt(df_diff['controller_x']**2+df_diff['controller_y']**2+df_diff['controller_z']**2)

df_dist = df_diff[['time','head_dist','controller_dist']]   
df_dist['time_exp'] = time_exp

df_speed = df_dist[['time']].copy()
df_speed['head_speed'] = df_dist['head_dist']/df_dist['time']
df_speed['controller_speed'] = df_dist['controller_dist']/df_dist['time']
df_speed['time_exp'] = time_exp
df_speed = df_speed.reset_index(drop=True)
plt.plot(df_speed['time_exp'],df_speed['controller_speed'], label='head')

start = time.time()
#all_breakpoints = plot_prediction(time=df_speed['time_exp'],y=df_speed['controller_speed'], prediction_line_color='red',sig_level=0.05, beta_bool=True, force_left_intersection=True, force_right_intersection=True, init_segment_size=20, window_size=20)
plot_prediction(time=df_speed['time_exp'],y=df_speed['controller_speed'], prediction_line_color='orange', beta_bool=True, force_left_intersection=True, force_right_intersection=True, init_segment_size=10, window_size=10, sig_level = 10**(-2))


print(time.time()-start)
plt.legend()
plt.xlabel('Time (seconds)')
plt.ylabel('Speed (unit/second)')
plt.title('Speed')
plt.show()

