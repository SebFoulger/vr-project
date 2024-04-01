import pandas as pd
import numpy as np
import os
import git
from smoothness_metrics import *
import json

def metric_values(metrics):

    metric_df_young = pd.DataFrame(columns = list(map(lambda x: x[2], metrics))+
                                             list(map(lambda x: x[2]+'_clean', metrics))+['id'])
    metric_df_old = metric_df_young.copy()

    repo = git.Repo('.', search_parent_directories=True)

    input_path = os.path.join(repo.working_tree_dir, 'input_data', 'adl')

    breakpoint_file_path = os.path.join(repo.working_tree_dir, 'outputs', 'adl_breakpoints')

    for metric, kwargs, metric_name, col_name in metrics:
        kwargs['fs'] = 120
        
        metric_val_list = []
        metric_val_list_clean = []
        id_list = []
        for file_name in os.listdir(os.path.join(input_path, 'young', 'halves')):
            if file_name != 'halves':
                id_list.append(file_name[:4])
                if 'breakpoints' in kwargs.keys():
                    with open(os.path.join(breakpoint_file_path, file_name[:4]+'_y.json')) as breakpoint_file:
                        kwargs['breakpoints'] = json.load(breakpoint_file)

                df = pd.read_csv(os.path.join(input_path, 'young', 'halves', file_name))

                kwargs['movement'] = df[col_name]
                kwargs_clean = kwargs.copy()
                kwargs_clean['movement'] = df[col_name+'_clean']
                metric_val_list.append(metric(**kwargs))
                metric_val_list_clean.append(metric(**kwargs_clean))

        metric_df_young['id'] = id_list
        metric_df_young[metric_name] = metric_val_list
        metric_df_young[metric_name+'_clean'] = metric_val_list_clean

        metric_val_list = []
        metric_val_list_clean = []
        id_list = []
        for file_name in os.listdir(os.path.join(input_path, 'old', 'halves')):
            if file_name != 'halves':
                id_list.append(file_name[:4])
                if 'breakpoints' in kwargs.keys():
                    with open(os.path.join(breakpoint_file_path, file_name[:4]+'_o.json')) as breakpoint_file:
                        kwargs['breakpoints'] = json.load(breakpoint_file)

                df = pd.read_csv(os.path.join(input_path, 'old', 'halves', file_name))

                kwargs['movement'] = df[col_name]
                kwargs_clean = kwargs.copy()
                kwargs_clean['movement'] = df[col_name+'_clean']

                metric_val_list.append(metric(**kwargs))
                metric_val_list_clean.append(metric(**kwargs_clean))

        metric_df_old['id'] = id_list
        metric_df_old[metric_name] = metric_val_list
        metric_df_old[metric_name+'_clean'] = metric_val_list_clean
    
    metric_df_young.to_csv(os.path.join(repo.working_tree_dir, 'outputs', 'young_metrics_halves.csv'), index=False)
    metric_df_old.to_csv(os.path.join(repo.working_tree_dir, 'outputs', 'old_metrics_halves.csv'), index=False)

if __name__ == "__main__":

    metrics = [(sparc, {} , 'SPARC', 'speed'), (ldj, {'data_type': 'speed'}, 'LDJ', 'speed'), 
               (SegmentMetric(sparc).value, {'breakpoints': []}, 'Segment_SPARC', 'speed'), 
               (SegmentMetric(ldj).value, {'data_type': 'jerk', 'breakpoints': []} , 'Segment_LDJ', 'jerk')]

    metric_values(metrics)