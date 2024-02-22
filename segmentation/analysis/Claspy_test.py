from claspy.segmentation import BinaryClaSPSegmentation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import git
import os

clasp = BinaryClaSPSegmentation(validation='score_threshold', window_size=10, excl_radius=4)

repo = git.Repo('.', search_parent_directories = True)

file_name = 'head/dist/1_1.csv'

file = os.path.join(repo.working_tree_dir, 'input_data', file_name)
df = pd.read_csv(file)[:10000].reset_index(drop = True)

col_name = 'head_dist_clean'

plt.plot(df['timeExp'], df[col_name])
df = df[['timeExp', col_name]].copy().dropna().reset_index(drop=True)

print(clasp.fit_predict(np.array(df[col_name])))

ax1, ax2 = clasp.plot()
ax1.plot()
ax2.plot()
plt.show()
