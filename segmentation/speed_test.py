import git
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
from linear_approach import LinearSegmentation

repo = git.Repo('.', search_parent_directories=True)
repo.working_tree_dir

file_location = repo.working_tree_dir+'/input_data/test1.csv'
df = pd.read_csv(file_location)

df=df[['frame',' timeExp','head_x','head_y','head_z',' controller_x',' controller_y',' controller_z']]
df=df.rename(columns={' controller_x': 'controller_x',' controller_y': 'controller_y',' controller_z': 'controller_z',
                      ' timeExp': 'time'})
time_exp = df['time'][1:]

df_diff = df.diff().dropna()

df_diff['head_dist'] = np.sqrt(df_diff['head_x']**2+df_diff['head_y']**2+df_diff['head_z']**2)
df_diff['controller_dist'] = np.sqrt(df_diff['controller_x']**2+df_diff['controller_y']**2+df_diff['controller_z']**2)

df_dist = df_diff[['time','head_dist','controller_dist']]
df_dist['time_exp'] = time_exp


df_speed = df_dist[['time']]
df_speed['head_speed'] = df_dist['head_dist']/df_dist['time']
df_speed['controller_speed'] = df_dist['controller_dist']/df_dist['time']
df_speed['time_exp'] = time_exp
df_speed = df_speed[:5000].reset_index()
prev_break = 0
cut_time = 1
plt.plot(df_speed['time_exp'],df_speed['head_speed'], label='head')
start = time.time()
for _break in list(df_speed['time_exp'].loc[df_speed['time_exp'].diff()>cut_time].index)+[len(df_speed)]:
    cur_df = df_speed[prev_break:_break-1]

    segmentation = LinearSegmentation(x=cur_df['time_exp'],y=cur_df['head_speed'])
    predictions, breakpoints = segmentation.segment()

    plt.plot(df_speed['time_exp'][prev_break:prev_break+len(predictions)],predictions, label='prediction')
    prev_break = _break
print(time.time()-start)

plt.title('Speed')
plt.show()