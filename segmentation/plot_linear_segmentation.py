import git
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from linear_segmentation import LinearSegmentation
from summarize_segments import summarize
import sys
import time
import json

def plot_prediction(time: pd.Series,
                    y: pd.Series,
                    cut_time: float = 1,
                    plot_breaks: bool = False,
                    sig_level: float = 0.01,
                    break_line_color: str = 'black',
                    prediction_line_color: str = 'orange',
                    window_size: int = 10,
                    return_models: bool = False):
    """Plot model and breakpoints (optionally)
    Args:
        time (pd.Series): Time data.
        y (pd.Series): y data to segment.
        cut_time (float, optional): The minimum time between data points at which we break into two larger sections. 
        Defaults to 1.
        plot_breaks (bool, optional): Indicate whether to plot breakpoints. Defaults to False.
        sig_level (float, optional): Significance level. Defaults to 0.01.
        break_line_color (str, optional): Colour to use for breakpoint plots. Defaults to 'black'.
        prediction_line_color (str, optional): Colour to use for model plots. Defaults to 'orange'.
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
        # Plot breakpoints
        if plot_breaks:
            for small_break in breakpoints:
                plt.plot([time[small_break+prev_break], time[small_break+prev_break]], [min(y), max(y)], 
                         color = break_line_color)
        # Plot predictions
        plt.plot(time[prev_break:prev_break + len(predictions)], predictions, color = prediction_line_color)
        prev_break = _break 
    if return_models:
        return {'breakpoints': all_breakpoints, 'model_results': model_results}
    else:
        return {'breakpoints': all_breakpoints}


if __name__ == "__main__":
    # input_args = [sub, session, object, variable]. i.e. [1, 2, head, speed]
    input_args = sys.argv[1:]

    if len(input_args) == 0:
        input_args = ['1', '1', '1', 'controller', 'speed']
    repo = git.Repo('.', search_parent_directories = True)

    file_name = '_'.join(input_args[:3]) + '.csv'

    file = os.path.join(repo.working_tree_dir, 'input_data', input_args[3], input_args[4], file_name)
    df = pd.read_csv(file).reset_index(drop = True)

    col_name = f'{input_args[3]}_{input_args[4]}_clean'
    if input_args[4] == 'accel':
        var_name = (input_args[3] + ' acceleration').capitalize()
    elif input_args[4] == 'disp':
        var_name = (input_args[3] + ' displacement').capitalize()
    elif input_args[4] == 'dist':
        var_name = (input_args[3] + ' distance').capitalize()
    else:
        var_name = (input_args[3] + ' ' + input_args[4]).capitalize()
    

    plt.plot(df['timeExp'], df[col_name])
    df = df[['timeExp', col_name]].copy().dropna().reset_index(drop=True)
    print(np.mean(df[col_name]**2))
    start = time.time()
    
    breakpoint_file_name = os.path.join(repo.working_tree_dir, 'outputs', 'breakpoints')
    """
    if input_args[3] == 'controller':
        return_dict = plot_prediction(time = df['timeExp'], y = df[col_name], sig_level = 10**(-4), return_models=True)
        summarize(df, return_dict['breakpoints'], col_name, model_results=return_dict['model_results'], 
              save_name = 'summarize_'+'_'.join(input_args[:3]) + '_c.csv')
        breakpoint_file_name = os.path.join(breakpoint_file_name, '_'.join(input_args[:3])+'_c.json')
    
    else:
        return_dict = plot_prediction(time = df['timeExp'], y = df[col_name], window_size = 20, sig_level = 10**(-5),
                                      return_models=True)
        summarize(df, return_dict['breakpoints'], col_name, model_results=return_dict['model_results'], 
              save_name = 'summarize_'+'_'.join(input_args[:3]) + '_h.csv', window_size=20)
        breakpoint_file_name = os.path.join(breakpoint_file_name, '_'.join(input_args[:3])+'_h.json')

    with open(breakpoint_file_name, 'w') as breakpoint_file:
        json.dump(return_dict['breakpoints'], breakpoint_file)

    print(time.time()-start)
    """
    plt.xlabel('Time elapsed (seconds)')

    plt.ylabel(var_name+' (unit$/s^2$)')
    plt.title(f'Subject {input_args[0]}, week {input_args[1]}, session {input_args[2]}, {var_name}')
    plt.show()