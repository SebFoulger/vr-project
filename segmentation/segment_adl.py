from linear_segmentation import LinearSegmentation
import os
import git
import pandas as pd
import json
from summarize_segments import summarize

if __name__ == "__main__":

    repo = git.Repo('.', search_parent_directories=True)
    file_path = os.path.join(repo.working_tree_dir, 'input_data', 'adl')

    breakpoint_file_path = os.path.join(repo.working_tree_dir, 'outputs', 'adl_breakpoints')

    for file_name in os.listdir(os.path.join(file_path, 'young', 'halves')):
        if file_name == 'halves':
            continue
        df = pd.read_csv(os.path.join(file_path, 'young', 'halves',  file_name))
        time = pd.Series([i/120 for i in range(len(df))])
        time.name = 'timeExp'
        return_dict = LinearSegmentation().segment(time, df['speed_clean'], sig_level=0.0001, return_models=True)
        df['timeExp'] = time
        breakpoints = [0]+return_dict['breakpoints']+[len(df)-1]
        summarize(df, breakpoints, 'speed_clean', return_dict['model_results'],
                  f'summarize_{file_name.strip(".csv")}_y.csv')
        breakpoint_file_name = os.path.join(breakpoint_file_path, file_name.strip('.csv')+'_y.json')

        with open(breakpoint_file_name, 'w') as breakpoint_file:
            json.dump(return_dict['breakpoints'], breakpoint_file)
    
    for file_name in os.listdir(os.path.join(file_path, 'old', 'halves')):
        if file_name == 'halves':
            continue
        df = pd.read_csv(os.path.join(file_path, 'old', 'halves', file_name))
        time = pd.Series([i/120 for i in range(len(df))])
        time.name = 'timeExp'
        return_dict = LinearSegmentation().segment(time, df['speed_clean'], sig_level=0.0001, return_models=True)
        df['timeExp'] = time
        breakpoints = [0]+return_dict['breakpoints']+[len(df)-1]
        summarize(df, breakpoints , 'speed_clean', return_dict['model_results'],
                  f'summarize_{file_name.strip(".csv")}_o.csv')
        breakpoint_file_name = os.path.join(breakpoint_file_path, file_name.strip('.csv')+'_o.json')

        with open(breakpoint_file_name, 'w') as breakpoint_file:
            json.dump(return_dict['breakpoints'], breakpoint_file)