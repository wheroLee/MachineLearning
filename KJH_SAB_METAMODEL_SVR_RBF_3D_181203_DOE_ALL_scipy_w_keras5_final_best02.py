import pandas as pd
from numpy import shape, ravel, linspace, meshgrid, min, max, arange, array
from matplotlib import cm
from matplotlib import pyplot
import matplotlib.pyplot as plt
#from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits import natgrid


dataset = pd.read_csv('DOE_ALL.csv')


contour_min = 2     
contour_max = 5  

S = dataset.iloc[:,0:13].values
t = dataset.iloc[:,13].values

x = S[:, 0]
y = S[:, 1]
ratio = S[:, 4]
z = t

xi = linspace(min(x), max(x),100)
yi = linspace(min(y), max(y),100)
X, Y = meshgrid(xi, yi)
from scipy.interpolate import Rbf
#rbf = Rbf(x,y,z,  smooth=0.1)
#rbf = Rbf(x,y,z, function='multiquadric' , smooth=0.1, epsilon=0)
rbf = Rbf(x,y,z, function='multiquadric' , smooth=0.1)
Z = rbf(X,Y)

original_xmaxvalue = 0
original_ymaxvalue = 0
original_zmaxvalue = 0

originx, originy, originz = x, y, z

for i,j,k in zip(originx,originy,originz):
    if k > original_zmaxvalue:
        original_xmaxvalue = i
        original_ymaxvalue = j
        original_zmaxvalue = k

print ("Max value of Sampling Point = {:.2f}, {:.2f}, {:.2f}".format(original_xmaxvalue,original_ymaxvalue,original_zmaxvalue))


fig = pyplot.figure(1)
ax = Axes3D(fig)
CMAP=cm.jet
CMAP_VAL = linspace(contour_min,contour_max,10)
CMAP_BOUND = cm.colors.BoundaryNorm(CMAP_VAL,CMAP.N)
#surf = ax.plot_surface(X,Y ,Z,facecolors=cm.jet(Z/max(z)), rstride=1, cstride=1,linewidth=0,antialiased=True,alpha=0.8 )
surf = ax.plot_surface(X,Y ,Z,cmap=CMAP, rstride=1, cstride=1,linewidth=0.3,antialiased=True,alpha=0.8,edgecolor="black",norm=CMAP_BOUND)
surf.set_clim(contour_min,contour_max)
#ax.scatter3D(x, y, z,c=z, cmap=CMAP, s=100,edgecolor="black",norm=CMAP_BOUND)
ax.set_zlim3d(min(z), max(z))
colorscale = cm.ScalarMappable(cmap=CMAP)

colorscale.set_array(z)
#fig.colorbar(colorscale)
fig.colorbar(colorscale,cmap=CMAP, ticks=CMAP_VAL,spacing='uniform',norm=CMAP_BOUND,boundaries=CMAP_VAL).set_clim(contour_min,contour_max)



ax2 = fig.gca(projection='3d')
pyplot.xlabel('Diff')
pyplot.ylabel('Vent')
ax.set_zlabel('Rating')
#ax.axis('equal')
ax.axis('tight')
pyplot.title("SVR - RBF Interpolation 3D")


pyplot.figure(2)
n = pyplot.Normalize(-2,2)
pyplot.pcolor(X,Y,Z,cmap=CMAP, alpha=0.5,edgecolor="black",norm=CMAP_BOUND)
pyplot.scatter(x,y,50,z, cmap=CMAP,edgecolor="black",norm=CMAP_BOUND)
#pyplot.colorbar()
pyplot.clim(contour_min,contour_max)
pyplot.colorbar().set_clim(contour_min,contour_max)

pyplot.xlabel('Diff')
pyplot.ylabel('Vent')
pyplot.title("SVR - Z projection (Normalized)")

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler


train_X = []

train_X = array([list(a) for a in zip(x,y)])
train_Y = array([z]).reshape(len(z),1)

#print("train data shape = ",shape(train_X))
#print("result data shape = ",shape(train_Y))
#print("train data list = \n",train_X)
#print("result data list = \n",train_Y)

model = Sequential()
model.add(Dense(64, input_dim=2, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

#optimizer = optimizers.Adam(0.001)
#optimizer = optimizers.Adadelta(0.001)
optimizer = optimizers.RMSprop(0.001)
"""
Adadelta: Optimizer that implements the Adadelta algorithm.
Adagrad: Optimizer that implements the Adagrad algorithm.
Adam: Optimizer that implements the Adam algorithm.
Adamax: Optimizer that implements the Adamax algorithm.
Ftrl: Optimizer that implements the FTRL algorithm.
Nadam: Optimizer that implements the NAdam algorithm.
Optimizer: Updated base class for optimizers.
RMSprop: Optimizer that implements the RMSprop algorithm.
SGD: Stochastic gradient descent and momentum optimizer.
"""
model.compile(loss='mae', optimizer=optimizer, metric=['mae','mse'])

model.fit(train_X, train_Y, epochs=9000, verbose=0)

tf_xi = linspace(min(x), max(x),100)
tf_yi = linspace(min(y), max(y),100)
tf_X, tf_Y = meshgrid(tf_xi, tf_yi)
pred_X = ravel(tf_X).tolist()
pred_Y = ravel(tf_Y).tolist()

all_XY = zip(pred_X,pred_Y)

mesh_all = [list(a) for a in all_XY]

out_Z = model.predict(mesh_all)
pred_Z = out_Z.reshape(tf_X.shape)

fig_tf = pyplot.figure(3)
ax_tf = Axes3D(fig_tf)
surf_tf = ax_tf.plot_surface(tf_X,tf_Y ,pred_Z,cmap=CMAP, rstride=1, cstride=1,linewidth=0.3,antialiased=True,alpha=0.8,edgecolor="black",norm=CMAP_BOUND)
surf_tf.set_clim(contour_min,contour_max)
ax_tf.set_zlim3d(min(z), max(z))
#fig_tf.colorbar(colorscale)
fig_tf.colorbar(colorscale,cmap=CMAP, ticks=CMAP_VAL,spacing='uniform',norm=CMAP_BOUND,boundaries=CMAP_VAL).set_clim(contour_min,contour_max)
ax.axis('tight')
pyplot.xlabel('Diff')
pyplot.ylabel('Vent')
ax.set_zlabel('Rating')
pyplot.title("Tensorflow KERAS Prediction 3D")
pyplot.show()
