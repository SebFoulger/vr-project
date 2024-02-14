from claspy.segmentation import BinaryClaSPSegmentation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

clasp = BinaryClaSPSegmentation(validation='score_threshold', window_size=10, excl_radius=4)

df = pd.read_csv('speed.csv')
print(clasp.fit_predict(np.array(df['controller_speed'])))

ax1, ax2 = clasp.plot()
ax1.plot()
ax2.plot()
plt.show()
