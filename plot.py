import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

Nx,Ny = 256,256
Nt = 1000
grid = np.genfromtxt('grid.csv', delimiter=',')
gridX = np.transpose(np.reshape(np.copy(grid[:,0]),(Nx,Ny)))
gridY = np.transpose(np.reshape(np.copy(grid[:,1]),(Nx,Ny)))
for i in range(Nt):
    print("iter:{}".format(i))
    # u = np.genfromtxt('u_{}.csv'.format(i), delimiter=',')
    # v = np.genfromtxt('v_{}.csv'.format(i), delimiter=',')
    omega = np.genfromtxt('omega_{}.csv'.format(i), delimiter=',')
    # p = np.genfromtxt('p_{}.csv'.format(i), delimiter=',')
    # u,v,omega = u[:-1],v[:-1],omega[:-1]
    omega = omega[:-1]
    # u = u[:-1]
    # uMat = np.zeros((Nx,Ny))
    # vMat = np.zeros((Nx,Ny))
    omegaMat = np.zeros((Nx,Ny))
    # pMat = np.zeros((Nx,Ny))
    count = 0
    for j in range(Ny):
        for ii in range(Nx):
            # uMat[ii,j] = u[count]
            # vMat[ii,j] = v[count]
            omegaMat[ii,j] = omega[count]
            # pMat[ii,j] = p[count]
            count+=1
    
    plt.pcolormesh(gridX,gridY,omegaMat,cmap='RdBu',vmin=-1.5,vmax=1.5)
    # plt.quiver(gridX,gridY,uMat,vMat)
    # plt.pcolormesh(gridX,gridY,np.sqrt(uMat*uMat + vMat*vMat),cmap='viridis',vmin=0,vmax=1)
    plt.xlim([0,2*np.pi])
    plt.ylim([0,2*np.pi])
    # plt.show(block=False)
    plt.axis('square')
    plt.savefig('omega_{}.png'.format(i))
    plt.close()
    # plt.colorbar()
    # plt.colorbar().remove()
    # plt.pause(0.01)
