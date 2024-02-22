import os
import git
import sys
import pandas as pd
import numpy as np
from hampel import hampel

def clean_outliers(time: pd.Series,
                    y: pd.Series,
                    cut_time: float = 1):
    
    prev_break = 0
    clean_y = []
    # Loop over all larger sections of data
    for _break in list(time.loc[time.diff() > cut_time].index) + [len(time)]:
        cur_y = y[prev_break:_break].reset_index(drop=True)
        clean_y = np.append(clean_y, hampel(cur_y, window_size=3, n=2, imputation=True))
        prev_break = _break 
    
    return clean_y

# Processing data
def preprocess(files: list = None):
    """Preprocesses files inputted.

    Args:
        files (list, optional): List of file. Set to None if preprocessing all files in raw_data is desired. Defaults to
        None.
    """    
    repo = git.Repo('.', search_parent_directories=True)
    raw_file_path = os.path.join(repo.working_tree_dir, 'raw_data')
    save_path = os.path.join(repo.working_tree_dir, 'input_data')

    if len(files) == 0:
        files = os.listdir(raw_file_path)

    for file in files:
        df = pd.read_csv(os.path.join(raw_file_path, file))

        rename_dict = {}
        for col in df.columns:
            rename_dict[col] = col.replace(' ', '')

        df=df.rename(columns=rename_dict)

        sub = str(df['sub'][0])
        session = str(df['session'][0])
        file_name = f'{sub}_{session}.csv'

        df = df.drop(columns=['frame', 'sub', 'subID', 'timepoint', 'session'])
        df.to_csv(os.path.join(save_path, 'raw', file_name), index = False)

        df = df[['timeExp','head_x','head_y','head_z','controller_x','controller_y','controller_z']].copy()

        df['head_dist'] = np.sqrt(df['head_x']**2+df['head_y']**2+df['head_z']**2)
        df['controller_dist'] = np.sqrt(df['controller_x']**2+df['controller_y']**2+df['controller_z']**2)

        df['head_dist_clean'] = clean_outliers(df['timeExp'], df['head_dist'])
        df['controller_dist_clean'] = clean_outliers(df['timeExp'], df['controller_dist'])

        df[['timeExp', 'head_dist', 'head_dist_clean']].to_csv(
                                            os.path.join(save_path, 'head', 'dist', file_name), index = False)
        df[['timeExp', 'controller_dist', 'controller_dist_clean']].to_csv(
                                            os.path.join(save_path, 'controller', 'dist', file_name), index = False)

        df_diff = df.diff().dropna()
        df_diff = df_diff.rename(columns = {'timeExp': 'timeFrame'})

        df_diff['head_disp'] = np.sqrt(df_diff['head_x']**2+df_diff['head_y']**2+df_diff['head_z']**2)
        controller_disp = np.sqrt(df_diff['controller_x']**2+df_diff['controller_y']**2+df_diff['controller_z']**2)
        df_diff['controller_disp'] = controller_disp

        df_disp = df_diff[['timeFrame','head_disp','controller_disp']]   
        df_disp['timeExp'] = df['timeExp'][1:]

        df_disp['head_disp_clean'] = clean_outliers(df_disp['timeExp'], df_disp['head_disp'])
        df_disp['controller_disp_clean'] = clean_outliers(df_disp['timeExp'], df_disp['controller_disp'])

        df_disp[['timeExp', 'head_disp', 'head_disp_clean']].to_csv(
                                            os.path.join(save_path, 'head', 'disp', file_name), index = False)
        df_disp[['timeExp', 'controller_disp', 'controller_disp_clean']].to_csv(
                                            os.path.join(save_path, 'controller', 'disp', file_name), index = False)

        df_speed = df_disp[['timeExp']].copy()
        df_speed['head_speed'] = df_disp['head_disp']/df_disp['timeFrame']
        df_speed['controller_speed'] = df_disp['controller_disp']/df_disp['timeFrame']

        df_speed['head_speed_clean'] = clean_outliers(df_speed['timeExp'], df_speed['head_speed'])
        df_speed['controller_speed_clean'] = clean_outliers(df_speed['timeExp'], df_speed['controller_speed'])

        df_speed[['timeExp', 'head_speed', 'head_speed_clean']].to_csv(
                                            os.path.join(save_path, 'head', 'speed', file_name), index = False)
        df_speed[['timeExp', 'controller_speed', 'controller_speed_clean']].to_csv(
                                            os.path.join(save_path, 'controller', 'speed', file_name), index = False)
        
        df_accel = df_speed.diff().dropna().rename(columns = {'timeExp': 'timeFrame', 'head_speed': 'head_accel',
                                                            'controller_speed': 'controller_accel'})
        df_accel['head_accel'] = df_accel['head_accel'] / df_accel['timeFrame']
        df_accel['controller_accel'] = df_accel['controller_accel'] / df_accel['timeFrame']
        df_accel['timeExp'] = df['timeExp'][2:]

        df_accel['head_accel_clean'] = clean_outliers(df_accel['timeExp'], df_accel['head_accel'])
        df_accel['controller_accel_clean'] = clean_outliers(df_accel['timeExp'], df_accel['controller_accel'])

        df_accel[['timeExp', 'head_accel', 'head_accel_clean']].to_csv(
                                            os.path.join(save_path, 'head', 'accel', file_name), index = False)
        df_accel[['timeExp', 'controller_accel', 'controller_accel_clean']].to_csv(
                                            os.path.join(save_path, 'controller', 'accel', file_name), index = False)

        print(file+' done')

if __name__ == "__main__":
    preprocess(sys.argv[1:])