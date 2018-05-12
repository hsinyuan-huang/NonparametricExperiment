import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

M1 = np.array([1, 1, -1, -1])
M2 = np.array([2, -2, 2, -2])

XMin = M1.min()
XMax = M1.max()
YMin = M2.min()
YMax = M2.max()

X, Y = np.mgrid[XMin:XMax:100j, YMin:YMax:100j]
Positions = np.vstack([X.ravel(), Y.ravel()])
Values = np.vstack([M1, M2])
Kernel = stats.gaussian_kde(Values)
Z = np.reshape(Kernel.evaluate(Positions).T, X.shape)

# Create a surface plot and projected filled contour plot under it.
Fig = plt.figure()
Ax = Fig.gca(projection='3d')
Ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)

CSet = Ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

# Adjust the limits, ticks and view angle
# Ax.set_zlim(-0.15, 0.2)
# Ax.set_zticks(np.linspace(0, .2, 5))
# Ax.view_init(27, -21)

plt.show()


