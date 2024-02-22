import pandas as pd
import numpy as np 
import statsmodels.api as sm

def summarize(df:pd.DataFrame,
              breakpoints: list,
              col: str,
              min_break_dist: int = 10,
              no_splits: int = 5,
              beta_bool: bool = False,
              force_left_intersection: bool = False,
              force_right_intersection: bool = False,
              window_size: int = 10,
              save_name: str = 'summarize.csv'):
    """Summarizes segmented data

    Args:
        df (pd.DataFrame): dataframe with columns ['timeExp',col].
        breakpoints (list): list of breakpoints indicating sections.
        col (str): name of column (y values).
        min_break_dist (int, optional): minimum number of points for a segment to be considered. Defaults to 10.
        no_splits (int, optional): number of splits for considering data. Defaults to 5.
        beta_bool (bool, optional): _description_. Defaults to False.
        force_left_intersection (bool, optional): _description_. Defaults to False.
        force_right_intersection (bool, optional): _description_. Defaults to False.
        window_size (int, optional): _description_. Defaults to 10.
        save_name (str, optional): name to save output file as. Defaults to 'summarize'.
    """    
    segments = [df.iloc[breakpoints[i-1]:breakpoints[i]] for i in range(1,len(breakpoints)) if len(df.iloc[breakpoints[i-1]:breakpoints[i]])>=min_break_dist]
    
    mean_segments = list(map(lambda x: np.mean(x[col]),segments))

    df_segments = pd.DataFrame({'mean': mean_segments})

    no_segments = len(df_segments)

    df_segments['start_time'] = list(map(lambda x:x.iloc[0]['timeExp'],segments))

    df_segments['end_time'] = list(map(lambda x:x.iloc[-1]['timeExp'],segments))

    df_segments['min'] = list(map(lambda x: min(x[col]),segments))

    df_segments['max'] = list(map(lambda x: max(x[col]),segments))

    df_segments['std'] = list(map(lambda x: np.std(x[col]),segments))

    df_segments['duration'] = df_segments['end_time']-df_segments['start_time']

    df_segments['range'] = df_segments['max'] - df_segments['min']

    if force_left_intersection:
        # This has not been implemented correctly, need to shift data to force intersections
        df_segments['model'] = list(map(lambda df:sm.OLS(df[col],df['timeExp']).fit(use_t=True), segments))
    else:
        df_segments['model'] = list(map(lambda df:sm.OLS(df[col],sm.add_constant(df['timeExp'])).fit(use_t=True), segments))

    df_segments['beta'] = np.vectorize(lambda x: x.params['timeExp'])(df_segments['model'])

    if not beta_bool:
        df_segments['const'] = np.vectorize(lambda x: x.params['const'])(df_segments['model'])

    df_segments['mse'] = np.vectorize(lambda x: x.mse_total)(df_segments['model'])

    df_segments['fvalue'] = np.vectorize(lambda x: x.fvalue)(df_segments['model'])

    df_segments['rsquared_adj'] = np.vectorize(lambda x: x.rsquared_adj)(df_segments['model'])

    df_segments['t_beta'] = np.vectorize(lambda x: x.tvalues['timeExp'])(df_segments['model'])

    if not beta_bool:
        df_segments['t_const'] = np.vectorize(lambda x: x.tvalues['const'])(df_segments['model'])

    df_segments['t_pvalue_beta'] = np.vectorize(lambda x: x.pvalues['timeExp'])(df_segments['model'])

    if not beta_bool:
        df_segments['t_pvalue_const'] = np.vectorize(lambda x: x.pvalues['const'])(df_segments['model'])
        df_segments['t_pvalue']=np.vectorize(lambda x: x.t_test(x.params).pvalue)(df_segments['model'])
    
    if force_right_intersection:
        # This has not been implemented correctly, need to shift data to force intersections
        df_segments['init_model'] = list(map(lambda df:sm.OLS(df[:window_size][col],df[:window_size]['timeExp']).fit(use_t=True), segments))
    else:
        df_segments['init_model'] = list(map(lambda df:sm.OLS(df[:window_size][col],sm.add_constant(df[:window_size]['timeExp'])).fit(use_t=True), segments))
    
    df_segments['init_beta_prev'] = [0]+list(np.vectorize(lambda x:x.params['timeExp'])(df_segments['init_model']))[1:]

    if not beta_bool:
        df_segments['init_const_prev'] = [0]+list(np.vectorize(lambda x: x.params['const'])(df_segments['init_model']))[1:]

    if beta_bool:
        df_segments['t_pvalue_right']=df_segments.apply(lambda row: row['init_model'].t_test(f'timeExp = {row["init_beta_prev"]}').pvalue, axis=1)
    else:
        df_segments['t_pvalue_right']=df_segments.apply(lambda row: row['init_model'].t_test(pd.Series({'timeExp': row['init_beta_prev'], 'const': row['init_const_prev']})).pvalue, axis=1)

    # Can no longer use segments from here as we change order

    df_segments = df_segments.sort_values(by='range', ignore_index=True)

    split_list = []
    for i in range(1,no_splits+1):
        split_list += (no_segments//no_splits)*[i]
        if i<= no_segments%no_splits:
            split_list.append(i)

    df_segments['split'] = split_list

    df_segments.to_csv(f'outputs/{save_name}', index = False)