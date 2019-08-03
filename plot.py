import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
f = open("initial.dat")
X = np.loadtxt(f)
X = X.transpose()
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[0],X[1],X[2],s=1)
plt.show()
