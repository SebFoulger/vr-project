import git
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from linear_approach import LinearSegmentation
from summarize_segments import summarize
import sys
import time

def plot_prediction(time: pd.Series,
                    y: pd.Series,
                    cut_time: float = 1,
                    plot_breaks: bool = False,
                    sig_level: float = 0.01,
                    break_line_color: str = 'black',
                    prediction_line_color: str = 'orange',
                    window_size: int = 10):
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

    Returns:
        list: List of all breakpoints.
    """    
    assert(len(time) == len(y))

    all_breakpoints = []
    prev_break = 0
    # Loop over all larger sections of data
    for _break in list(time.loc[time.diff() > cut_time].index) + [len(time)]:
        # Find the current time and y values
        cur_time = time[prev_break:_break - 1].reset_index(drop=True)
        cur_y = y[prev_break:_break - 1].reset_index(drop=True)
        # Segment current data
        segmentation = LinearSegmentation()
        segmentation_dict = segmentation.segment(x = cur_time, y = cur_y, window_size = window_size, 
                                                        sig_level = sig_level)
        predictions, breakpoints = segmentation_dict['predictions'], segmentation_dict['breakpoints']                                         
        # Add breakpoints to total list of breakpoints
        all_breakpoints += [prev_break] + list(map(lambda x: x + prev_break, breakpoints))
        all_breakpoints += [prev_break + len(predictions)]
        # Plot breakpoints
        if plot_breaks:
            for small_break in breakpoints:
                plt.plot([time[small_break+prev_break], time[small_break+prev_break]], [min(y), max(y)], 
                         color = break_line_color)
        # Plot predictions
        plt.plot(time[prev_break:prev_break + len(predictions)], predictions, color = prediction_line_color)
        prev_break = _break 
    return all_breakpoints


if __name__ == "__main__":
    # input_args = [sub, session, object, variable]. i.e. [1, 2, head, speed]
    input_args = sys.argv[1:]

    if len(input_args) == 0:
        input_args = ['1', '1', 'controller', 'speed']
    repo = git.Repo('.', search_parent_directories = True)

    file_name = '_'.join(input_args[:2]) + '.csv'

    file = os.path.join(repo.working_tree_dir, 'input_data', input_args[2], input_args[3], file_name)
    df = pd.read_csv(file)[:5000].reset_index(drop = True)

    col_name = input_args[2] + '_' + input_args[3]
    var_name = (input_args[2] + ' ' + input_args[3]).capitalize()

    df = df[['timeExp', col_name]].copy().dropna().reset_index(drop=True)

    plt.plot(df['timeExp'], df[col_name])
    start = time.time()

    plot_prediction(time = df['timeExp'], y = df[col_name])

    print(time.time()-start)
    plt.xlabel('Time (seconds)')
    plt.ylabel(var_name)
    plt.title('Subject ' + input_args[0] + ', session ' + input_args[1] + ', ' + var_name)
    plt.show()

