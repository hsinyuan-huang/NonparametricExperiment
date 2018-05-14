###[1]###
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x_axis = np.linspace(-10, 10, num=500000)
mixture_gaussian = (norm.pdf(x_axis, -5, 1) + norm.pdf(x_axis, 0, 1) + norm.pdf(x_axis, 5, 1)) / 3
plt.plot(x_axis, mixture_gaussian)

xy_matrix = np.transpose(np.array([x_axis, mixture_gaussian]))


###[2]###
import numpy as np
import matplotlib.pyplot as pl
import scipy.stats as st

data1 = np.random.multivariate_normal((-5,1), [[1,0], [0,1]], 50000//4)
data2 = np.random.multivariate_normal((0,1), [[1,0], [0,1]], 50000//4)
data3 = np.random.multivariate_normal((5,1), [[1,0], [0,1]], 50000//4)
data = np.concatenate((data1,data2,data3),axis=0)
x = data[:, 0]
y = data[:, 1]
xmin, xmax = -10, 10
ymin, ymax = -10, 10

# Peform the kernel density estimate
xx, yy = np.mgrid[xmin:xmax:50000j, ymin:ymax:50000j]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = st.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)

fig = pl.figure()
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
# Contourf plot
cfset = ax.contourf(xx, yy, f, cmap='Blues')
## Or kernel density estimate plot instead of the contourf plot
#ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
# Contour plot
cset = ax.contour(xx, yy, f, colors='k')
# Label plot
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('Y1')
ax.set_ylabel('Y0')

pl.show()
