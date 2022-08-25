from sklearn.model_selection import GridSearchCV
import lhsmdu
import matplotlib.pyplot as plt
import numpy
import time
Stime = time.time()
l = lhsmdu.sample(45,2400) # Latin Hypercube Sampling of two variables, and 10 samples each
Etime = time.time()
second = (Etime - Stime)/1545925769.9618232
print(second)

k = lhsmdu.createRandomStandardUniformMatrix(2,10)

fig = plt.figure()
ax = fig.gca()
ax.set_xticks(numpy.arange(0,1,0.1))
ax.set_yticks(numpy.arange(0,1,0.1))
plt.scatter(k[0], k[1], color= 'b', label='LHS-MDU' )
#plt.scatter(l[0], l[1], color=\ r\ , label=\ MC\ )\n ,
plt.grid()
plt.show() 
