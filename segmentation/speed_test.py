import git
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from linear_approach import LinearSegmentation
from summarize_segments import summarize
import sys

def plot_prediction(time: pd.Series,
                    y: pd.Series,
                    cut_time: float = 1,
                    plot_breaks: bool = False,
                    beta_bool: bool = True,
                    sig_level: float = 0.01,
                    break_line_color: str = 'black',
                    prediction_line_color: str = 'orange',
                    prediction_line_label: str = 'prediction',
                    init_segment_size: int = 10,
                    window_size: int = 10,
                    step: int = 1,
                    force_left_intersection: bool = False,
                    force_right_intersection: bool = False):
    all_breakpoints = []
    prev_break = 0
    for _break in list(time.loc[time.diff() > cut_time].index) + [len(time)]:
        cur_time = time[prev_break:_break - 1]
        cur_y = y[prev_break:_break - 1]
        
        segmentation = LinearSegmentation(x = cur_time, y = cur_y)
        predictions, breakpoints = segmentation.segment(init_segment_size = init_segment_size, 
                                                        window_size = window_size,
                                                        step = step,
                                                        sig_level = sig_level,
                                                        beta_bool = beta_bool,
                                                        force_left_intersection = force_left_intersection,
                                                        force_right_intersection = force_right_intersection)
        all_breakpoints += [prev_break] + list(map(lambda x: x + prev_break, breakpoints))
        all_breakpoints += [prev_break + len(predictions)]

        if plot_breaks:
            for small_break in breakpoints:
                plt.plot([time[small_break], time[small_break]], [min(y), max(y)], color = break_line_color)
        plt.plot(time[prev_break:prev_break + len(predictions)], predictions, color = prediction_line_color, 
                 label = prediction_line_label)
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

    plot_prediction(time = df['timeExp'], y = df[col_name], force_left_intersection = True,
                    force_right_intersection = True, prediction_line_color = 'red')

    plt.xlabel('Time (seconds)')
    plt.ylabel(var_name)
    plt.title('Subject ' + input_args[0] + ', session ' + input_args[1] + ', ' + var_name)
    plt.show()

