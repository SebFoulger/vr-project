import pandas as pd
import numpy as np
import os
import git
from loess.loess_1d import loess_1d
from scipy.signal import find_peaks

def preprocess():

    repo = git.Repo('.', search_parent_directories=True)
    raw_file_path = os.path.join(repo.working_tree_dir, 'raw_data', 'clean_adl')

    save_path = os.path.join(repo.working_tree_dir, 'input_data', 'adl')

    files = ['C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C18', 'C19', 'H01', 'H02', 'H03', 'H04', 'H07', 'H08', 'H13', 'H15']

    for file_name in files:
        file_raw = os.path.join(raw_file_path, file_name+'_raw.csv')
        file_clean = os.path.join(raw_file_path, file_name+'_clean.csv')
        df = pd.read_csv(file_raw, header=None).T.rename(columns = {0: 'speed'})
        df['speed_clean'] = pd.read_csv(file_clean, header=None)[0]

        if file_name[0] == 'C':
            df.to_csv(os.path.join(save_path, 'old', file_name[1:]+'.csv'), index=False)
        else:
            df.to_csv(os.path.join(save_path, 'young', file_name[1:]+'.csv'), index=False)

def calculate_jerk(fs: float = 120):
    repo = git.Repo('.', search_parent_directories=True)
    file_path = os.path.join(repo.working_tree_dir, 'input_data', 'adl')
    for file_name in os.listdir(os.path.join(file_path, 'old')):
        if file_name == 'halves':
            continue
        file = os.path.join(file_path, 'old', file_name)
        df = pd.read_csv(file)
        df['accel'] = df['speed'].diff()*fs
        df['jerk'] = df['accel'].diff()*fs
        df['accel_clean'] = df['speed_clean'].diff()*fs
        df['jerk_clean'] = df['accel_clean'].diff()*fs

        df.to_csv(file, index=False)
        
    for file_name in os.listdir(os.path.join(file_path, 'young')):
        if file_name == 'halves':
            continue
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
        if file_name == 'halves':
            continue
        file = os.path.join(file_path, 'old', file_name)
        df = pd.read_csv(file)
        split = int(len(df)/2)
        df_1, df_2 = df[:split], df[split:].reset_index(drop=True)

        df_1.to_csv(os.path.join(file_path, 'old', 'halves', file_name[:2]+'_1.csv'))
        df_2.to_csv(os.path.join(file_path, 'old', 'halves', file_name[:2]+'_2.csv'))

    for file_name in os.listdir(os.path.join(file_path, 'young')):
        if file_name == 'halves':
            continue
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
        if file_name == 'halves':
            continue
        file = os.path.join(file_path, 'old', file_name)
        df = pd.read_csv(file)
        peaks = df.iloc[find_peaks(df['speed_clean'], prominence=0.05)[0]]['speed_clean']
        rest = len(df[df['speed_clean']<0.05])

        cur_dict = {'id': file_name[:2]+'_o', 'duration': len(df)/fs, 'mean_peak': np.mean(peaks), 
                    'path_length': sum(df['speed_clean']/fs), 'rel_activity': rest/len(df)}

        parameter_df = parameter_df.append(cur_dict, ignore_index=True)


    for file_name in os.listdir(os.path.join(file_path, 'young')):
        if file_name == 'halves':
            continue
        file = os.path.join(file_path, 'young', file_name)
        df = pd.read_csv(file)

        peaks = df.iloc[find_peaks(df['speed_clean'], prominence=0.05)[0]]['speed_clean']
        rest = len(df[df['speed_clean']<0.05])

        cur_dict = {'id': file_name[:2]+'_y', 'duration': len(df)/fs, 'mean_peak': np.mean(peaks), 
                    'path_length': sum(df['speed_clean']/fs), 'rel_activity': rest/len(df)}

        parameter_df = parameter_df.append(cur_dict, ignore_index=True)
    
    parameter_df.to_csv(os.path.join(repo.working_tree_dir, 'outputs', 'adl_parameters.csv'), index=False)

if __name__ == "__main__":
    #preprocess()
    #calculate_jerk()
    #split_files()
    parameter_values()
