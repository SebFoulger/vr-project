import pandas as pd
import numpy as np 

def summarize(df:pd.DataFrame, breakpoints: list, col: str, min_break_dist: int):
    segments = [df.iloc[breakpoints[i-1]:breakpoints[i]] for i in range(1,len(breakpoints))]
    
    mean_segments = list(map(lambda x: np.mean(x[col]),segments))

    df_segments = pd.DataFrame({'mean': mean_segments})