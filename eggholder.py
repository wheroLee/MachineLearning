import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def eggholder(x):
    return (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 47))))
             -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 47)))))

bounds = [(-512, 512), (-512, 512)]


x = np.arange(-512, 513)
y = np.arange(-512, 513)
xgrid, ygrid = np.meshgrid(x, y)
xy = np.stack([xgrid, ygrid])

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(121, projection='3d')
ax.view_init(45, -45)
ax.plot_surface(xgrid, ygrid, eggholder(xy), cmap='terrain')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('eggholder(x, y)')
#plt.show()

from scipy import optimize

results = dict()

results['shgo'] = optimize.shgo(eggholder, bounds)
results['DA'] = optimize.dual_annealing(eggholder, bounds)
results['DE'] = optimize.differential_evolution(eggholder, bounds)
results['BH'] = optimize.basinhopping(eggholder, bounds)
results['shgo_sobol'] = optimize.shgo(eggholder, bounds, n=200, iters=5,
                                      sampling_method='sobol')

ax2 = fig.add_subplot(122)
im = ax2.imshow(eggholder(xy), interpolation='bilinear', origin='lower',
               cmap='gray')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

def plot_point(res,  color=None, label=None,s=40):
#def plot_point(res, marker='o', color=None, label=None,s=None):
#    ax2.plot(512+res.x[0], 512+res.x[1], marker=marker, color=color, ms=10, label=label)
    ax2.scatter(512+res.x[0], 512+res.x[1],  color=color, s=s, label=label,facecolors='none',edgecolors=color,linewidths=3)

def plot_solid_point(res, marker='o', color=None, label=None,s=None):
    ax2.plot(512+res.x[0], 512+res.x[1], marker=marker, color=color, ms=10, label=label)

plot_point(results['BH'], color='y',label='BH',s=100)  # basinhopping           - yellow
plot_point(results['DE'], color='c',label='DE',s=200)  # differential_evolution - cyan
plot_point(results['DA'], color='m',label='DA',s=300)  # dual_annealing.        - white

# SHGO produces multiple minima, plot them all (with a smaller marker size)
plot_solid_point(results['shgo'], color='r',label='SHGO')
plot_solid_point(results['shgo_sobol'], color='b', label='SHGO_SOBOL')
ax2.scatter(512+512,512+404.2319, color='r',s=800, label='Min. location',facecolors='none', edgecolors='r')
#plot_point(results['shgo'], color='y', marker='+')
#plot_point(results['shgo_sobol'], color='b', marker='x')
for i in range(results['shgo_sobol'].xl.shape[0]):
    ax2.plot(512 + results['shgo_sobol'].xl[i, 0],
            512 + results['shgo_sobol'].xl[i, 1],
            'ro', ms=2)

ax2.set_xlim([-4, 514*2])
ax2.set_ylim([-4, 514*2])
plt.legend(bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), loc='upper left')
plt.show()
