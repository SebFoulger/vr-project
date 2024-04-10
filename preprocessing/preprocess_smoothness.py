import pandas as pd
import numpy as np
import os
import git
from scipy import signal
from scipy.signal import find_peaks

def preprocess(fs: float = 90, files = []):
    repo = git.Repo('.', search_parent_directories=True)
    file_path = os.path.join(repo.working_tree_dir, 'input_data', 'controller', 'speed')

    if len(files)==0:
        files = os.listdir(file_path)
    b, a = signal.butter(N=8, Wn = 10, fs=fs)
    for file_name in files:
        file = os.path.join(file_path, file_name)
        df = pd.read_csv(file)
        
        df['controller_speed_clean_2'] = signal.filtfilt(b, a, df['controller_speed']) 

        df.to_csv(file, index=False)
        print(file, 'done')

def calculate_jerk(fs: float = 90, files = []):
    repo = git.Repo('.', search_parent_directories=True)
    file_path = os.path.join(repo.working_tree_dir, 'input_data', 'controller', 'speed')
    if len(files)==0:
        files=os.listdir(file_path)
    for file_name in files:
        file = os.path.join(file_path, file_name)
        df = pd.read_csv(file)
        df['controller_jerk'] = df['controller_speed'].diff().diff()*(fs**2)
        df['controller_jerk_clean_2'] = df['controller_speed_clean_2'].diff().diff()*(fs**2)

        df.to_csv(file, index=False)

def parameter_values(fs: float = 90, files = []):
    # Path length, mean peak velocity, trial duration, relative activity
    parameter_df = pd.DataFrame(columns = ['id', 'duration', 'mean_peak', 'path_length', 'rel_activity'])
    repo = git.Repo('.', search_parent_directories=True)
    file_path = os.path.join(repo.working_tree_dir, 'input_data', 'controller', 'speed')
    if len(files)==0:
        files=os.listdir(file_path)
    for file_name in files:
        file = os.path.join(file_path, file_name)
        df = pd.read_csv(file)
        peaks = df.iloc[find_peaks(df['controller_speed'], prominence=0.05)[0]]['controller_speed']
        rest = len(df[df['controller_speed']<0.05])

        cur_dict = {'id': file_name.strip('.csv'), 'duration': len(df)/fs, 'mean_peak': np.mean(peaks), 
                    'path_length': sum(df['controller_speed']/fs), 'rel_activity': rest/len(df)}

        parameter_df = parameter_df.append(cur_dict, ignore_index=True)
        print(file_name)

    parameter_df.to_csv(os.path.join(repo.working_tree_dir, 'outputs', 'vr_parameters.csv'), index=False)

if __name__ == "__main__":
    #preprocess(files = ['10_2_1.csv'])
    #calculate_jerk()
    parameter_values()