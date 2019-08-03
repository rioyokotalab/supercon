import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
f = open("initial.dat",'rb')
N = np.fromfile(f,dtype='i',count=1)
x = np.fromfile(f,dtype='d',count=N)
y = np.fromfile(f,dtype='d',count=N)
z = np.fromfile(f,dtype='d',count=N)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x,y,z,s=1)
plt.show()
