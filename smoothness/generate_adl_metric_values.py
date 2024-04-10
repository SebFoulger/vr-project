import pandas as pd
import numpy as np
import os
import git
from smoothness_metrics import *
import json



def metric_ind(kwargs, breakpoint_file_path, file_name, col_name, working_tree_dir, df_path, y_or_o):
    
    if 'breakpoints' in kwargs.keys():
        with open(os.path.join(breakpoint_file_path, file_name.strip('.csv')+f'_{y_or_o}.json')) as breakpoint_file:
            kwargs['breakpoints'] = json.load(breakpoint_file)

    df = pd.read_csv(os.path.join(df_path, file_name))

    kwargs['movement'] = df[col_name+'_clean']
    print(file_name)
    if 'betas' in kwargs.keys():
        kwargs['betas'] = pd.read_csv(os.path.join(working_tree_dir, 'outputs', 'adl_summarize', 
                                                    f'summarize_{file_name.strip(".csv")}_{y_or_o}.csv'))['beta']
    
    if 'speed' in kwargs.keys():
        kwargs['speed'] = df['speed_clean']

    return kwargs


def metric_values(metrics):

    metric_df_young = pd.DataFrame(columns = list(map(lambda x: x[2], metrics))+['id'])
    metric_df_old = metric_df_young.copy()

    repo = git.Repo('.', search_parent_directories=True)

    input_path = os.path.join(repo.working_tree_dir, 'input_data', 'adl')

    breakpoint_file_path = os.path.join(repo.working_tree_dir, 'outputs', 'adl_breakpoints')
    
    for metric, kwargs, metric_name, col_name in metrics:
        kwargs['fs'] = 120
        
        metric_val_list = []
        id_list = []
        young_path = os.path.join(input_path, 'young')
        for file_name in os.listdir(young_path):
            if file_name == 'halves':
                continue
            id_list.append(file_name.strip('.csv'))

            kwargs = metric_ind(kwargs, breakpoint_file_path, file_name, col_name, 
                                                 repo.working_tree_dir, young_path, 'y')

            metric_val_list.append(metric(**kwargs))

        metric_df_young['id'] = id_list
        metric_df_young[metric_name] = metric_val_list

        metric_val_list = []
        id_list = []
        old_path = os.path.join(input_path, 'old')
        for file_name in os.listdir(old_path):
            if file_name == 'halves':
                continue
            id_list.append(file_name.strip('.csv'))

            kwargs= metric_ind(kwargs, breakpoint_file_path, file_name, col_name, 
                                                 repo.working_tree_dir, old_path, 'o')

            metric_val_list.append(metric(**kwargs))


        metric_df_old['id'] = id_list
        metric_df_old[metric_name] = metric_val_list
    
    metric_df_young.to_csv(os.path.join(repo.working_tree_dir, 'outputs', 'young_metrics.csv'), index=False)
    metric_df_old.to_csv(os.path.join(repo.working_tree_dir, 'outputs', 'old_metrics.csv'), index=False)


def metric_values_halves(metrics):

    metric_df_young = pd.DataFrame(columns = list(map(lambda x: x[2], metrics))+['id'])
    metric_df_old = metric_df_young.copy()

    repo = git.Repo('.', search_parent_directories=True)

    input_path = os.path.join(repo.working_tree_dir, 'input_data', 'adl')

    breakpoint_file_path = os.path.join(repo.working_tree_dir, 'outputs', 'adl_breakpoints')
    
    for metric, kwargs, metric_name, col_name in metrics:
        kwargs['fs'] = 120
        
        metric_val_list = []
        id_list = []
        young_halves_path = os.path.join(input_path, 'young', 'halves')
        for file_name in os.listdir(young_halves_path):
            if file_name == 'halves':
                continue
            id_list.append(file_name.strip('.csv'))

            kwargs = metric_ind(kwargs, breakpoint_file_path, file_name, col_name, 
                                                 repo.working_tree_dir, young_halves_path, 'y')

            metric_val_list.append(metric(**kwargs))

        metric_df_young['id'] = id_list
        metric_df_young[metric_name] = metric_val_list

        metric_val_list = []
        id_list = []
        old_halves_path = os.path.join(input_path, 'old', 'halves')
        for file_name in os.listdir(old_halves_path):
            if file_name == 'halves':
                continue
            id_list.append(file_name.strip('.csv'))

            kwargs = metric_ind(kwargs, breakpoint_file_path, file_name, col_name, 
                                                 repo.working_tree_dir, old_halves_path, 'o')


            metric_val_list.append(metric(**kwargs))

        metric_df_old['id'] = id_list
        metric_df_old[metric_name] = metric_val_list
    
    metric_df_young.to_csv(os.path.join(repo.working_tree_dir, 'outputs', 'young_metrics_halves.csv'), index=False)
    metric_df_old.to_csv(os.path.join(repo.working_tree_dir, 'outputs', 'old_metrics_halves.csv'), index=False)

if __name__ == "__main__":

    metrics = [(nosp, {'betas': [], 'breakpoints': []}, 'NoSP', 'speed'),
               (sparc, {} , 'SPARC', 'speed'), 
               (ldj_adl, {'data_type': 'speed'}, 'LDJ', 'speed'), 
               (nop, {}, 'NoP', 'speed'), 
               (nos, {'breakpoints': []}, 'NoS', 'speed'),
               (SegmentMetric(ldj).value, {'data_type': 'jerk', 'breakpoints': [], 'speed': []} , 'Segment_LDJ', 'jerk'),
               (SegmentMetric(sparc).value, {'breakpoints': []}, 'Segment_SPARC', 'speed')]
    

    metric_values(metrics)

    metric_values_halves(metrics)