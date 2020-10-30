from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
figure = plt.figure()
ax = Axes3D(figure)
X = np.arange(-10,10,1)
Y = np.arange(-10,10,1)
print(X.shape)
print(Y)
X,Y = np.meshgrid(X,Y)
R = X**2 + Y**2
Z = R
print(Z)
print(Z.shape)
ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')
plt.show()
