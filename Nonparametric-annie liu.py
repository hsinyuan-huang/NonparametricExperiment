
# coding: utf-8

# In[3]:


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


# In[17]:


from scipy import stats
import numpy as np
def measure(n):
    m1 = np.random.normal(size=n)
    m2 = np.random.normal(scale=0.5, size=n)
    return m1+m2, m1-m2
m1, m2 = measure(2000)
xmin = m1.min()
xmax = m1.max()
ymin = m2.min()
ymax = m2.max()
X, Y = np.mgrid[xmin:xmax:500000j, ymin:ymax:500000j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([m1, m2])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)
Z


# In[18]:


import numpy as np
x, y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
d = np.sqrt(x*x+y*y)
d


# In[4]:


a = np.array([10, 1]) 
b = np.array([1, 10])


# In[5]:


a
b


# In[3]:


import numpy as np
x = np.random.normal(0, 1, 500000)
y = 1/(np.sqrt(2 * np.pi)) *np.exp( - (x)**2 / (2) )
s = np.array([x,y])
s


# In[6]:


import numpy as np
x = np.random.normal(0, 1, 500000)
y = 1/(np.sqrt(2 * np.pi)) *np.exp( - (x)**2 / (2) )
s = np.column_stack((x,y))
s


# In[16]:


from sklearn.mixture import GaussianMixture
import numpy as np
samples = np.random.normal(0, 1, 500000)
samples = np.append(samples, np.random.normal(1, 2, 500000))
samples = np.append(samples, np.random.normal(2, 3, 500000))
y = 1/(np.sqrt(2 * np.pi)) *np.exp( - (samples)**2 / (2) )
s = np.array([x,y])
s
gmm = GaussianMixture(n_components=3).fit(s)
sample = gmm.sample(n_samples=500000)
#gmm_x = samples
#gmm_y = gmm.predict(gmm_x)
#gmm_x
#gmm_y


# In[55]:


from numpy import *
from pypr.clustering import gmm
mc = [0.4, 0.4, 0.2]
centroids = [ array([0,0]), array([3,3]), array([0,4]) ]
ccov = [ array([[1,0.4],[0.4,1]]), diag((1,2)), diag((0.4,0.1)) ]
x = gmm.sample_gaussian_mixture(centroids, ccov, mc, samples=500000)
x


# In[58]:


import mixture


# In[34]:


import pip
pip.main(['install','PyPR'])
import PyPR


# In[52]:




