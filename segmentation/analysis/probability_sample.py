from linear_approach import LinearSegmentation
import numpy as np
import pandas as pd
from numpy.random import normal
import matplotlib.pyplot as plt
import time
import itertools

m1 = 0.413
m2 = -0.5

c1 = 0
c2 = 0.2739

var1 = 0.00005
var2 = 0.000076

left_size = 30
xs = np.arange(0,0.6,0.01)
yss = []
"""
sig_levels = [0.05] 
beta_bools = [True]
window_sizes = [5]
init_segment_sizes = [5]
left_intersections = [False]
right_intersections = [False]
"""
sig_levels = [0.05, 0.01, 0.001, 0.0001]
beta_bools = [True, False]
window_sizes = [5, 10, 15, 20]
init_segment_sizes = [5, 10, 15, 20]
left_intersections = [False, True]
right_intersections = [False, True]

no_loops = 100
start = time.time()
output_df = pd.DataFrame(columns = ['sig_level','beta_bool','window_size','init_segment_size','left_intersection','right_intersection', 'mean' ,'min', 'min_count', 'max', 'max_count', 'var', 'var_outlier_removed', 'breakpoints'])
for i in range(no_loops):
    ys = np.array(list(map(lambda x: x+normal(scale=np.sqrt(var1)),m1*xs[:left_size]+c1)))
    ys = np.append(ys,np.array(list(map(lambda x:x+normal(scale=np.sqrt(var2)),m2*xs[left_size:]+c2))))
    ys = pd.Series( ys)
    yss.append(ys)
xs = pd.Series( xs)
xs.name = 'time_exp'
k=1
start1=time.time()
no_its = len(list(itertools.product(sig_levels, beta_bools, window_sizes, init_segment_sizes, left_intersections, right_intersections)))
for sig_level, beta_bool, window_size, init_segment_size, left_intersection, right_intersection in itertools.product(sig_levels, beta_bools, window_sizes, init_segment_sizes, left_intersections, right_intersections):
    print(k,"/",no_its, start1-time.time())
    start1  =time.time()
    
    k+=1
    if not beta_bool and (left_intersection or right_intersection):
        pass
    else:
        breakpoints = []
        for ys in yss:
            
            if left_intersection:
                breakpoint = LinearSegmentation(x=xs,y=ys)._ind_segment(x=xs,y=ys,init_segment_size=init_segment_size,
                                                                    window_size=window_size, sig_level=sig_level,
                                                                    beta_bool=beta_bool, left_intersection=(xs[0],m1*xs[0]+c1),
                                                                    right_intersection=right_intersection)[0]
            else:
                breakpoint = LinearSegmentation(x=xs,y=ys)._ind_segment(x=xs,y=ys,init_segment_size=init_segment_size,
                                                                    window_size=window_size, sig_level=sig_level,
                                                                    beta_bool=beta_bool, right_intersection=right_intersection)[0]
            breakpoints.append(breakpoint)
        min_outlier = init_segment_size
        max_outlier = len(xs)-1-window_size
        min_break = min(breakpoints)
        max_break = max(breakpoints)
        breakpoints_outlier_removed = list(filter(lambda x: x != min_outlier and x < max_outlier,breakpoints))
        row = {'sig_level': sig_level, 'beta_bool': beta_bool, 'window_size': window_size, 
            'init_segment_size': init_segment_size, 'left_intersection': left_intersection, 
            'right_intersection': right_intersection, 'mean': np.mean(breakpoints), 
            'mean_outlier_removed': np.mean(breakpoints_outlier_removed),
            'min': min_break, 'min_count': breakpoints.count(min_break) , 'max': max_break, 
            'max_count': len(list(filter(lambda x:x>=max_outlier,breakpoints))), 'var': np.var(breakpoints), 
            'var_outlier_removed': np.var(breakpoints_outlier_removed), 'breakpoints': breakpoints}
        output_df = output_df.append(row,ignore_index=True)
print(time.time()-start)
output_df.to_csv('output6.csv', index=False)


for ys in yss:
    plt.plot(xs,ys)
ys_flat = np.array(m1*xs[:left_size]+c1)
ys_flat = np.append(ys_flat,m2*xs[left_size:]+c2)
plt.plot(xs,ys_flat)
plt.show()
