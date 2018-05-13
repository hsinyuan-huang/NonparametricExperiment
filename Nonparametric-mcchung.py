import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from random import choices

# two dimension coordinate
M1 = np.array([1, 1, -1, -1])
M2 = np.array([2, -2, 2, -2])

# x, y axis max & min
XMin = M1.min()
XMax = M1.max()
YMin = M2.min()
YMax = M2.max()

# generate a probability distribution
X, Y = np.mgrid[XMin:XMax:1000j, YMin:YMax:1000j]  # 1000*1000 size
print('array extension down')

Positions = np.vstack([X.ravel(), Y.ravel()])
Values = np.vstack([M1, M2])
Kernel = stats.gaussian_kde(Values)
Z = np.reshape(Kernel.evaluate(Positions).T, X.shape)
print('probability distribution down')

Population = list(zip(X.flatten().tolist(), Y.flatten().tolist()))

# randomly choose 50,0000 * 2 points
# Z = Distribution[np.random.randint(Distribution.shape[0], size=500000), :]
# Index = np.random.randint(Distribution.shape[0], size=(500000, 2))
# print('randomly choosing down')
Probability = Z.flatten().tolist()

C = 0
RandomChoice = []

while C < 500000:
    RandomChoice.extend(choices(Population, Probability))
    C = C + 1
    print(C)
    continue

print(RandomChoice)

# draw a surface plot and a contour plot under
# Fig = plt.figure()
# Ax = Fig.gca(projection='3d')
# Ax.plot_surface(X, Y, Distribution, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)

# CSet = Ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

# limits, ticks and view angle
# Ax.set_zlim(-0.15, 0.2)
# Ax.set_zticks(np.linspace(0, .2, 5))
# Ax.view_init(27, -21)

plt.figure(figsize=(120, 90))
x, y = zip(*RandomChoice)
plt.plot(x, y, 'ro')
plt.savefig('500000.png')
plt.show()


