import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mean1 = [3,0]
mean2 = [-3,0]
mean3 = [0,3]
mean4 = [0,-3]
cov = [[1,0],[0,1]]
nvar = 500000//4

aa = np.random.multivariate_normal(mean1,cov,nvar)
bb = np.random.multivariate_normal(mean2,cov,nvar)
cc = np.random.multivariate_normal(mean3,cov,nvar)
dd = np.random.multivariate_normal(mean4,cov,nvar)

fnldist = np.concatenate((aa,bb,cc,dd),axis=0)
print(fnldist.shape)


plt.hexbin(fnldist[:,0],fnldist[:,1])
plt.show()
