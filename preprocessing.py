import os
import git
import sys
import pandas as pd
import numpy as np

# Processing data
def preprocess(files: list = None):
    """Preprocesses files inputted.

    Args:
        files (list, optional): List of file. Set to None if preprocessing all files in raw_data is desired. Defaults to
        None.
    """    
    repo = git.Repo('.', search_parent_directories=True)
    raw_file_path = os.path.join(repo.working_tree_dir, 'raw_data')

    if len(files) == 0:
        files = os.listdir(raw_file_path)

    for file in files:
        df = pd.read_csv(os.path.join(raw_file_path, file))

        rename_dict = {}
        for col in df.columns:
            rename_dict[col] = col.replace(' ', '')

        df=df.rename(columns=rename_dict)

        sub = str(df['sub'][0])
        session = str(df['session'][0])

        df = df.drop(columns=['frame', 'sub', 'subID', 'timepoint', 'session'])
        df.to_csv(os.path.join(repo.working_tree_dir, 'input_data', sub+'_'+session+'_raw.csv'), index = False)

        df = df[['timeExp','head_x','head_y','head_z','controller_x','controller_y','controller_z']].copy()

        df_diff = df.diff().dropna()
        df_diff = df_diff.rename(columns = {'timeExp': 'timeFrame'})

        df_diff['head_dist'] = np.sqrt(df_diff['head_x']**2+df_diff['head_y']**2+df_diff['head_z']**2)
        controller_dist = np.sqrt(df_diff['controller_x']**2+df_diff['controller_y']**2+df_diff['controller_z']**2)
        df_diff['controller_dist'] = controller_dist

        df_dist = df_diff[['timeFrame','head_dist','controller_dist']]   
        df_dist['timeExp'] = df['timeExp'][1:]
        df_dist[['timeExp', 'head_dist']].to_csv(
            os.path.join(repo.working_tree_dir, 'input_data', sub+'_'+session+'_head_dist.csv'), index = False)
        df_dist[['timeExp', 'controller_dist']].to_csv(
            os.path.join(repo.working_tree_dir, 'input_data', sub+'_'+session+'_controller_dist.csv'), index = False)

        df_speed = df_dist[['timeExp']].copy()
        df_speed['head_speed'] = df_dist['head_dist']/df_dist['timeFrame']
        df_speed['controller_speed'] = df_dist['controller_dist']/df_dist['timeFrame']

        df_speed[['timeExp', 'head_speed']].to_csv(
            os.path.join(repo.working_tree_dir, 'input_data', sub+'_'+session+'_head_speed.csv'), index = False)
        df_speed[['timeExp', 'controller_speed']].to_csv(
            os.path.join(repo.working_tree_dir, 'input_data', sub+'_'+session+'_controller_speed.csv'), index = False)


if __name__ == "__main__":
    preprocess(sys.argv[1:])