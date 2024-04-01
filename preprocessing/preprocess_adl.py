import pandas as pd
import numpy as np
import os
import git
from loess.loess_1d import loess_1d
from scipy.signal import find_peaks

def preprocess(files, fs: float = 120):
    if files == []:
        return None
    repo = git.Repo('.', search_parent_directories=True)
    raw_file_path = os.path.join(repo.working_tree_dir, 'raw_data', 'raw_adl')

    save_path = os.path.join(repo.working_tree_dir, 'input_data', 'adl')

    files = os.listdir(raw_file_path)
    for file_name in files:
        file = os.path.join(raw_file_path, file_name)
        df = pd.read_csv(file, header=None).T.rename(columns = {0: 'x', 1: 'y', 2: 'z'}).drop(columns = [3])
        df = df.dropna().reset_index(drop=True)

        df_diff = df.diff().dropna().reset_index(drop=True)
        df_vel = df_diff * fs
        
        df_speed = pd.DataFrame({'speed': np.sqrt(df_vel['x']**2 + df_vel['y']**2 + df_vel['z']**2)})
        df_speed['speed_clean'] = loess_1d(np.array([i / fs for i in range(len(df)-1)]), np.array(df_speed['speed']), 
                                           npoints = 12)[1]
        if file_name[0] == 'C':
            df_speed.to_csv(os.path.join(save_path, 'old', file_name[1:3]+'.csv'), index=False)
        else:
            df_speed.to_csv(os.path.join(save_path, 'young', file_name[1:3]+'.csv'), index=False)

def calculate_jerk(fs: float = 120):
    repo = git.Repo('.', search_parent_directories=True)
    file_path = os.path.join(repo.working_tree_dir, 'input_data', 'adl')
    for file_name in os.listdir(os.path.join(file_path, 'old')):
        file = os.path.join(file_path, 'old', file_name)
        df = pd.read_csv(file)
        df['accel'] = df['speed'].diff()*fs
        df['jerk'] = df['accel'].diff()*fs
        df['accel_clean'] = df['speed_clean'].diff()*fs
        df['jerk_clean'] = df['accel_clean'].diff()*fs

        df.to_csv(file, index=False)
        
    for file_name in os.listdir(os.path.join(file_path, 'young')):
        file = os.path.join(file_path, 'young', file_name)
        df = pd.read_csv(file)
        df['accel'] = df['speed'].diff()*fs
        df['jerk'] = df['accel'].diff()*fs
        df['accel_clean'] = df['speed_clean'].diff()*fs
        df['jerk_clean'] = df['accel_clean'].diff()*fs

        df.to_csv(file, index=False)

def split_files():
    repo = git.Repo('.', search_parent_directories=True)
    file_path = os.path.join(repo.working_tree_dir, 'input_data', 'adl')
    for file_name in os.listdir(os.path.join(file_path, 'old')):
        file = os.path.join(file_path, 'old', file_name)
        df = pd.read_csv(file)
        split = int(len(df)/2)
        df_1, df_2 = df[:split], df[split:].reset_index(drop=True)

        df_1.to_csv(os.path.join(file_path, 'old', 'halves', file_name[:2]+'_1.csv'))
        df_2.to_csv(os.path.join(file_path, 'old', 'halves', file_name[:2]+'_2.csv'))

    for file_name in os.listdir(os.path.join(file_path, 'young')):
        file = os.path.join(file_path, 'young', file_name)
        df = pd.read_csv(file)
        split = int(len(df)/2)
        df_1, df_2 = df[:split], df[split:].reset_index(drop=True)

        df_1.to_csv(os.path.join(file_path, 'young', 'halves', file_name[:2]+'_1.csv'))
        df_2.to_csv(os.path.join(file_path, 'young', 'halves', file_name[:2]+'_2.csv'))

def parameter_values(fs: float = 120):
    # Path length, mean peak velocity, trial duration, relative activity
    parameter_df = pd.DataFrame(columns = ['id', 'duration', 'mean_peak', 'path_length', 'rel_activity'])
    repo = git.Repo('.', search_parent_directories=True)
    file_path = os.path.join(repo.working_tree_dir, 'input_data', 'adl')
    for file_name in os.listdir(os.path.join(file_path, 'old')):
        if file_name != 'halves':
            file = os.path.join(file_path, 'old', file_name)
            df = pd.read_csv(file)
            peaks = df.iloc[find_peaks(df['speed'], prominence=0.05)[0]]['speed']
            rest = len(df[df['speed']<0.05])

            cur_dict = {'id': file_name[:2]+'_o', 'duration': len(df), 'mean_peak': np.mean(peaks), 
                        'path_length': sum(df['speed']/fs), 'rel_activity': rest/len(df)}

            parameter_df = parameter_df.append(cur_dict, ignore_index=True)


    for file_name in os.listdir(os.path.join(file_path, 'young')):
        if file_name != 'halves':
            file = os.path.join(file_path, 'young', file_name)
            df = pd.read_csv(file)

            peaks = df.iloc[find_peaks(df['speed'], prominence=0.05)[0]]['speed']
            rest = len(df[df['speed']<0.05])

            cur_dict = {'id': file_name[:2]+'_y', 'duration': len(df), 'mean_peak': np.mean(peaks), 
                        'path_length': sum(df['speed']/fs), 'rel_activity': rest/len(df)}

            parameter_df = parameter_df.append(cur_dict, ignore_index=True)
    
    parameter_df.to_csv(os.path.join(repo.working_tree_dir, 'outputs', 'adl_parameters.csv'), index=False)

def temp(fs = 120):
    repo = git.Repo('.', search_parent_directories=True)
    file_path = os.path.join(repo.working_tree_dir, 'input_data', 'adl')
    """
    for file_name in os.listdir(os.path.join(file_path, 'old')):
        print(file_name)
        file = os.path.join(file_path, 'old', file_name)
        df = pd.read_csv(file)
        df['jerk_clean_2'] =  np.append([np.nan, np.nan], loess_1d(np.array([i / fs for i in range(len(df)-2)]), 
                                                                   np.array(df.dropna()['jerk']), npoints = 12)[1])
        df.to_csv(file, index=False)
    """

    for file_name in os.listdir(os.path.join(file_path, 'young')):
        print(file_name)
        file = os.path.join(file_path, 'young', file_name)
        df = pd.read_csv(file)

        df['jerk_clean_2'] =  np.append([np.nan, np.nan], loess_1d(np.array([i / fs for i in range(len(df)-2)]), 
                                                                   np.array(df.dropna()['jerk']), npoints = 12)[1])
        
        df.to_csv(file, index=False)

if __name__ == "__main__":
    #preprocess(files = [])
    #calculate_jerk()
    #split_files()

    #parameter_values()
    temp()
