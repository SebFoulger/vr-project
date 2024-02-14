import os
import git
import sys

# Processing data
def preprocess(files: list = None):

    if len(files) == 0:
        repo = git.Repo('.', search_parent_directories=True)

        raw_file_path = os.path.join(repo.working_tree_dir, 'raw_data')
        files = os.listdir(raw_file_path)

    print(files)

if __name__ == "__main__":
    preprocess(sys.argv[1:])