import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

Nx,Ny = 64,64
Nt = 1000
grid = np.genfromtxt('grid.csv', delimiter=',')
gridX = np.reshape(np.copy(grid[:,0]),(Nx,Ny))
gridY = np.reshape(np.copy(grid[:,1]),(Nx,Ny))
for i in range(Nt)[::10]:
    u = np.genfromtxt('u_{}.csv'.format(i), delimiter=',')
    v = np.genfromtxt('v_{}.csv'.format(i), delimiter=',')
    omega = np.genfromtxt('omega_{}.csv'.format(i), delimiter=',')
    u,v,omega = u[:-1],v[:-1],omega[:-1]
    uMat = np.zeros((Nx,Ny))
    vMat = np.zeros((Nx,Ny))
    omegaMat = np.zeros((Nx,Ny))
    count = 0
    for j in range(Ny):
        for i in range(Nx):
            uMat[i,j] = u[count]
            vMat[i,j] = v[count]
            omegaMat[i,j] = omega[count]
            count+=1
    plt.pcolormesh(gridX,gridY,omegaMat,cmap='bwr',vmin=-1,vmax=1)
    # plt.show()
    plt.show(block=False)
    plt.colorbar()
    plt.colorbar().remove()
    plt.pause(0.01)
