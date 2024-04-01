import git
import pandas as pd
import numpy as np
import os
from linear_segmentation import LinearSegmentation
from summarize_segments import summarize
import sys
import time
import json

"""
Segments all files of a given object/variable type (e.g. controller speed) and saves breakpoints and summarization of
segmentation.
"""

def segment_total(time: pd.Series,
                    y: pd.Series,
                    cut_time: float = 1,
                    sig_level: float = 0.01,
                    window_size: int = 10,
                    return_models: bool = False):
    """Segment all data
    Args:
        time (pd.Series): Time data.
        y (pd.Series): y data to segment.
        cut_time (float, optional): The minimum time between data points at which we break into two larger sections. 
        Defaults to 1.
        sig_level (float, optional): Significance level. Defaults to 0.01.
        window_size (int, optional): Window size to use in segmentation. Defaults to 10.
        return_models (bool, optional): Indicates whether to return the models for each segment. Defaults to False.
    Returns:
        dict: {'breakpoints' (list of int): List of all breakpoints,
               'model_results' (list of RegressionResults): List of models for each segment. Requires return_models to 
               be returned.}
    """    
    assert(len(time) == len(y))

    all_breakpoints = []
    prev_break = 0
    if return_models:
        model_results = []
    # Loop over all larger sections of data
    for _break in list(time.loc[time.diff() > cut_time].index) + [len(time)]:
        # Find the current time and y values
        cur_time = time[prev_break:_break - 1].reset_index(drop=True)
        cur_y = y[prev_break:_break - 1].reset_index(drop=True)
        # Segment current data
        segmentation = LinearSegmentation()
        segmentation_dict = segmentation.segment(x = cur_time, y = cur_y, window_size = window_size, 
                                                        sig_level = sig_level, return_models=return_models)
        predictions, breakpoints = segmentation_dict['predictions'], segmentation_dict['breakpoints']                                         
        # Add breakpoints to total list of breakpoints
        all_breakpoints += [prev_break] + list(map(lambda x: x + prev_break, breakpoints))
        all_breakpoints += [prev_break + len(predictions)]
        if return_models:
            model_results += segmentation_dict['model_results']

        prev_break = _break 
    if return_models:
        return {'breakpoints': all_breakpoints, 'model_results': model_results}
    else:
        return {'breakpoints': all_breakpoints}

if __name__ == "__main__":
    # input_args = [object, variable]. e.g. [head, speed]
    input_args = sys.argv[1:]

    if len(input_args) == 0:
        input_args = ['controller', 'speed']

    repo = git.Repo('.', search_parent_directories = True)

    files = os.listdir(os.path.join(repo.working_tree_dir, 'input_data', input_args[0], input_args[1]))

    total_start = time.time()
    for file_num, file_name in enumerate(files):
        print('Progress:', file_num, '/', len(files))
        file_start = time.time()
        sub, week, session = file_name.strip('.csv').split('_')
        file = os.path.join(repo.working_tree_dir, 'input_data', input_args[0], input_args[1], file_name)
        df = pd.read_csv(file).reset_index(drop = True)

        col_name = f'{input_args[0]}_{input_args[1]}_clean'

        if input_args[1] == 'accel':
            var_name = (input_args[0] + ' acceleration').capitalize()
        elif input_args[1] == 'disp':
            var_name = (input_args[0] + ' displacement').capitalize()
        elif input_args[1] == 'dist':
            var_name = (input_args[0] + ' distance').capitalize()
        else:
            var_name = (input_args[0] + ' ' + input_args[1]).capitalize()
        
        df = df[['timeExp', col_name]].copy().dropna().reset_index(drop=True)
        
        breakpoint_file_name = os.path.join(repo.working_tree_dir, 'outputs', 'breakpoints')
        
        if input_args[0] == 'controller':
            start = time.time()
            return_dict = segment_total(time = df['timeExp'], y = df[col_name], sig_level = 10**(-4), 
                                        return_models=True)
            print(file_name, 'segmentation done in', time.time()-start, 'seconds.')
            start = time.time()
            summarize(df, return_dict['breakpoints'], col_name, model_results=return_dict['model_results'], 
                      save_name = 'summarize_'+'_'.join([sub, week, session]) + '_c.csv')
            print(file_name, 'summarizing done in', time.time()-start, 'seconds.')
            breakpoint_file_name = os.path.join(breakpoint_file_name, '_'.join([sub, week, session])+'_c.json')
        
        else:
            return_dict = segment_total(time = df['timeExp'], y = df[col_name], window_size = 20, sig_level = 10**(-5),
                                        return_models=True)
            summarize(df, return_dict['breakpoints'], col_name, model_results=return_dict['model_results'], 
                save_name = 'summarize_'+'_'.join([sub, week, session]) + '_h.csv', window_size=20)
            breakpoint_file_name = os.path.join(breakpoint_file_name, '_'.join([sub, week, session])+'_h.json')

        with open(breakpoint_file_name, 'w') as breakpoint_file:
            json.dump(return_dict['breakpoints'], breakpoint_file)

        print(file_name, 'done in', time.time()-file_start, 'seconds.')
        
    print('Finished in', time.time()-total_start)
