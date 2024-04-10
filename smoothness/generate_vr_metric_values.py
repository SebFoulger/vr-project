import pandas as pd
import numpy as np
import os
import git
from smoothness_metrics import *
import json

def metric_values(metrics, df_exists = True):

    repo = git.Repo('.', search_parent_directories=True)

    input_path = os.path.join(repo.working_tree_dir, 'input_data', 'controller', 'speed')

    breakpoint_file_path = os.path.join(repo.working_tree_dir, 'outputs', 'breakpoints')

    if df_exists:
        metric_df = pd.read_csv(os.path.join(repo.working_tree_dir, 'outputs', 'vr_metrics.csv'))
    else:
        metric_df = pd.DataFrame(columns = list(map(lambda x: x[2], metrics))+
                                             list(map(lambda x: x[2]+'_clean', metrics))+['id'])

    for metric, kwargs, metric_name, col_name in metrics:
        kwargs['fs'] = 90
        
        metric_val_list = []
        metric_val_list_clean = []
        id_list = []
        for file_name in os.listdir(os.path.join(input_path)):
            print(file_name)
            id_list.append(file_name.strip('.csv'))
            if 'breakpoints' in kwargs.keys():
                with open(os.path.join(breakpoint_file_path, file_name.strip('.csv')+'_c.json')) as breakpoint_file:
                    kwargs['breakpoints'] = json.load(breakpoint_file)
            
            
            df = pd.read_csv(os.path.join(input_path, file_name))

            kwargs['movement'] = df[col_name]
            if 'betas' in kwargs.keys():
                kwargs['betas'] = pd.read_csv(os.path.join(repo.working_tree_dir, 'outputs', 'summarize', 
                                                           'summarize_'+file_name.strip('.csv')+'_c.csv'))['beta']
            kwargs_clean = kwargs.copy()
            kwargs_clean['movement'] = df[col_name+'_clean_2']
            
            if 'speed' in kwargs.keys():
                kwargs['speed'] = df['controller_speed']
                kwargs_clean['speed'] = df['controller_speed_clean_2']

            metric_val_list.append(metric(**kwargs))
            metric_val_list_clean.append(metric(**kwargs_clean))

        if not df_exists:
            metric_df['id'] = id_list
        metric_df[metric_name] = metric_val_list
        metric_df[metric_name+'_clean'] = metric_val_list_clean

        print(metric_name, 'done')
    
        metric_df.to_csv(os.path.join(repo.working_tree_dir, 'outputs', 'vr_metrics.csv'), index=False)

if __name__ == "__main__":
    
    metrics = [(nosp, {'breakpoints': [], 'betas': []}, 'NoSP', 'controller_speed'), 
               (SegmentMetric(sparc).value, {'breakpoints': []}, 'Segment_SPARC', 'controller_speed')]
    
    """
    metrics = [(sparc, {} , 'SPARC', 'controller_speed'), 
               (ldj, {'data_type': 'speed'}, 'LDJ', 'controller_speed'), 
               (nop, {}, 'NoP', 'controller_speed'), 
               (nos, {'breakpoints': []}, 'NoS', 'controller_speed'),
                (SegmentMetric(ldj).value, {'data_type': 'jerk', 'breakpoints': [], 'speed': []} , 'Segment_LDJ', 'controller_jerk')]
    """

    metric_values(metrics)