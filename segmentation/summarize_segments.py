import pandas as pd
import numpy as np 
import statsmodels.api as sm
import git
import os

def summarize(df:pd.DataFrame,
              breakpoints: list,
              col: str,
              model_results: list,
              save_name: str,
              no_splits: int = 5,
              window_size: int = 10):
    """Summarizes segmented data

    Args:
        df (pd.DataFrame): dataframe with columns ['timeExp',col].
        breakpoints (list): list of breakpoints indicating sections.
        col (str): name of column (y values).
        model_results (list of RegressionResults). List of regression results for each segment.
        min_break_dist (int, optional): minimum number of points for a segment to be considered. Defaults to 10.
        no_splits (int, optional): number of splits for considering data. Defaults to 5.
        window_size (int, optional): _description_. Defaults to 10.
        save_name (str, optional): name to save output file as. Defaults to 'summarize'.
    """    
    segments = [df.iloc[breakpoints[i-1]:breakpoints[i]] for i in range(1,len(breakpoints)) if breakpoints[i]-breakpoints[i-1] >= window_size-1]

    mean_segments = list(map(lambda x: np.mean(x[col]),segments))

    df_segments = pd.DataFrame({'mean': mean_segments})

    df_segments['median'] = list(map(lambda x:np.median(x[col]), segments))

    df_segments['start_time'] = list(map(lambda x:x.iloc[0]['timeExp'],segments))

    df_segments['end_time'] = list(map(lambda x:x.iloc[-1]['timeExp'],segments))

    df_segments['min'] = list(map(lambda x: min(x[col]),segments))

    df_segments['max'] = list(map(lambda x: max(x[col]),segments))

    df_segments['std'] = list(map(lambda x: np.std(x[col]),segments))

    df_segments['duration'] = df_segments['end_time']-df_segments['start_time']

    df_segments['range'] = df_segments['max'] - df_segments['min']

    df_segments['segment_size'] = list(map(lambda x: x.nobs, model_results))

    df_segments['beta'] = list(map(lambda x: x.params['timeExp'], model_results))

    df_segments['model_std'] = list(map(lambda x: x.scale**.5, model_results))

    repo = git.Repo('.', search_parent_directories = True)
    save_path = os.path.join(repo.working_tree_dir, 'outputs', 'summarize', save_name)

    df_segments.to_csv(save_path, index = False)