#A test for the diffusion maps algorithm on the swiss roll dataset.
#The embedding should be a rectangle.

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import diffusion_map as dm
import numpy as np


length_phi = 15;
length_Z = 10;
sigma = 0.1;
m = 10000

phi = length_phi*np.random.rand(m);
xi = np.random.rand(m);
Z = length_Z*np.random.rand(m);
X = 1./6*(phi + sigma*xi)*np.sin(phi);
Y = 1./6*(phi + sigma*xi)*np.cos(phi);

swiss_roll = np.array([X, Y, Z]).transpose()

dmap = dm.DiffusionMap(n_evecs = 2, epsilon = 0.1, alpha = 1.0, k=200).fit_transform(swiss_roll)

# plotting
plt.figure(figsize=(16,6))
ax = plt.subplot(121)
ax.scatter(dmap[:,0],dmap[:,1], c=dmap[:,0], cmap=plt.cm.Spectral);
plt.title('embedding of swiss roll');
plt.xlabel('\psi_1')
plt.ylabel('\psi_2')
ax.axis('tight');

ax2 = plt.subplot(122,projection='3d')
ax2.scatter(X,Y,Z, c=dmap[:,0], cmap=plt.cm.Spectral);
plt.title('swiss roll dataset');
ax2.view_init(75, 10)

plt.show()
