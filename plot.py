import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
u_hat = np.genfromtxt('u_hat.csv', delimiter=',')
u_sol = np.genfromtxt('u_sol.csv', delimiter=',')
t = np.genfromtxt('t.csv', delimiter=',')

# plt.plot(t,u_hat[0,:],'r')
# plt.plot(t,u_sol[0,:],'k')
N = np.genfromtxt('N.csv', delimiter=',')
e = np.genfromtxt('e.csv',delimiter=',')
plt.plot(N,e)
plt.xscale('log')
plt.yscale('log')
plt.show()
embed()