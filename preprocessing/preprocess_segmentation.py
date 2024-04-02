import os
import git
import sys
import pandas as pd
import numpy as np
from hampel import hampel
import json
import time

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
def preprocess(files: list = None,
               save_disp: bool = False,
               save_accel_clean: bool = False):
    """Preprocesses files inputted.

    Args:
        files (list, optional): List of files. Set to None if preprocessing all files in raw_data is desired. Defaults 
        to None.
        save_disp (bool, optional): Indicate whether to save the displacement data. Defaults to False.
        save_accel_clean (bool, optional): Indicate whether to save the cleaned acceleration data. Defaults to False.
    """    
    start = time.time()
    local_arg_dict = locals()
    
    repo = git.Repo('.', search_parent_directories=True)
    raw_data_file_path = os.path.join(repo.working_tree_dir, 'raw_data')

    raw_file_path = os.path.join(raw_data_file_path, 'raw')
    processed_file_path = os.path.join(raw_data_file_path, 'processed')
    meta_file_path = os.path.join(raw_data_file_path, 'metadata')

    save_path = os.path.join(repo.working_tree_dir, 'input_data')

    file_names = []

    if len(files) == 0:
        files = os.listdir(raw_file_path)

    for file in files:
        temp_start = time.time()
        df = pd.read_csv(os.path.join(raw_file_path, file))

        rename_dict = {}
        for col in df.columns:
            rename_dict[col] = col.replace(' ', '')

        df=df.rename(columns=rename_dict)

        sub = str(df['sub'][0])
        session = str(df['session'][0])
        time_point = str(df['timepoint'][0])
        file_name = f'{sub}_{time_point}_{session}.csv'

        file_names.append(file_name)

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

        if save_disp:
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

        if save_accel_clean: 
            df_accel['head_accel_clean'] = clean_outliers(df_accel['timeExp'], df_accel['head_accel'])
            df_accel['controller_accel_clean'] = clean_outliers(df_accel['timeExp'], df_accel['controller_accel'])

            df_accel[['timeExp', 'head_accel', 'head_accel_clean']].to_csv(
                                            os.path.join(save_path, 'head', 'accel', file_name), index = False)
            df_accel[['timeExp', 'controller_accel', 'controller_accel_clean']].to_csv(
                                            os.path.join(save_path, 'controller', 'accel', file_name), index = False)
        else:
            df_accel[['timeExp', 'head_accel']].to_csv(
                                            os.path.join(save_path, 'head', 'accel', file_name), index = False)
            df_accel[['timeExp', 'controller_accel']].to_csv(
                                            os.path.join(save_path, 'controller', 'accel', file_name), index = False)
            
        
        os.rename(os.path.join(raw_file_path, file), os.path.join(processed_file_path, file))
        print(file+' done in', time.time()-temp_start)

    if len(files)!=0:
        runs_file = open(os.path.join(meta_file_path, 'runs.txt'), 'r')
        run_no = int(runs_file.read())
        runs_file.close()
        runs_file = open(os.path.join(meta_file_path, 'runs.txt'), 'w')
        runs_file.write(str(run_no + 1))
        runs_file.close()

        meta_dict = {'args': local_arg_dict, 'files': files, 'file_names': file_names}

        with open(os.path.join(meta_file_path, f'{run_no + 1}_meta.json'), 'w') as meta_file:
            json.dump(meta_dict, meta_file)
    print('Time taken:', time.time() - start)

def correct_naming():
    """Corrects a naming error that some of the timepoints are 0 when they should be 1.
    """
    repo = git.Repo('.', search_parent_directories = True)
    input_file_path = os.path.join(repo.working_tree_dir, 'input_data')

    for var in ['controller', 'head']:
        for var_type in ['accel', 'disp', 'dist', 'speed']:
            cur_file_path = os.path.join(input_file_path, var, var_type)
            for file in os.listdir(cur_file_path):
                if file.split('_')[1] == '0':
                    new_file = file.split('_')
                    new_file[1] = '1'
                    os.rename(os.path.join(cur_file_path, file), os.path.join(cur_file_path, '_'.join(new_file)))

if __name__ == "__main__":
    preprocess(sys.argv[1:])
    correct_naming()