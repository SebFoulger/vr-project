import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt



def prob(xs,ys,x_new):

    xs_new = np.append(xs,x_new)
    
    n = len(xs)

    cov_x_y_mat = np.cov(xs,ys)

    var_x, cov_x_y = cov_x_y_mat[0,0], cov_x_y_mat[0,1]

    var_x_new = np.var(xs_new, ddof=1)

    beta = cov_x_y/var_x
    const = np.mean(ys)-beta*np.mean(xs)

    sigma = np.var(ys-beta*xs-const,ddof=1)
    t = (n*cov_x_y)/(sigma*np.sqrt(var_x))
    
    F = x_new/(n+1)-sum(xs_new)/((n+1)**2)
    G = np.dot(xs,ys)/(n+1)-sum(xs_new)*sum(ys)/((n+1)**2)+(beta*x_new+const)*F
    H = sigma*F

    A = t*(sigma**2)*np.sqrt(var_x_new)
    B = ((n+1)**2)*H
    C = t*n*sigma*np.sqrt(var_x_new)-((n+1)**2)*G

    z_1 = (-B-np.sqrt(B**2-4*A*C))/(2*A)
    z_2 = (-B+np.sqrt(B**2-4*A*C))/(2*A)

    z_1, z_2 = min(z_1,z_2), max(z_1,z_2)

    return norm.cdf(z_1)+1-norm.cdf(z_2), beta, const

df = pd.read_csv('speed.csv')

xs = df['time_exp'][:190]
ys = df['controller_speed'][:190]
x_new = df['time_exp'][190]

print(xs)
print(ys)
print(x_new)

p, beta, const = prob(xs=xs,ys=ys,x_new=x_new)
print(p)
"""
predictions = list(map(lambda x: x*beta+const,xs))


breakpoints = [0, 128, 176, 473, 620, 635, 649, 665, 700, 711, 740, 750, 791, 816, 830, 872, 901, 931, 947, 981, 1033, 1051, 1061, 1071, 1086, 1101, 1115, 1151, 1164, 1196, 1213, 1272, 1325, 1338, 1348, 1376, 1396, 1409, 1449, 1463, 1474, 1486, 1496, 1534, 1565, 1596, 1606, 1660, 1670, 1711, 1721, 1737, 1774, 1784, 1817, 1838, 1848, 1858, 1868, 1884, 1912, 1923, 1949, 1969, 1980, 1998]

for small_break in breakpoints:
    plt.plot([xs[small_break],xs[small_break]],[min(ys),max(ys)], color='black')

plt.plot(xs,ys)
plt.plot(xs,predictions)
plt.show()
"""