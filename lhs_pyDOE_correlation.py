from pyDOE import lhs
import matplotlib.pyplot as plt
import numpy
import time

Stime = time.time()
print(Stime)

var_num = 2 
point_num = 10   
lhs_base=lhs(var_num,point_num)
lhs_ctr=lhs(var_num,point_num,criterion='center')
lhs_minmax=lhs(var_num,point_num,criterion='m')
lhs_CM=lhs(var_num,point_num,criterion='cm')
lhs_corr=lhs(var_num,point_num,criterion='corr')

for i in lhs_corr:
    imsi = []

    for j in i:

        if j > 0 and j < 0.2:
            imsi.append(1)
        if j >= 0.2 and j < 0.4:
            imsi.append(2)
        if j >= 0.4 and j < 0.6:
            imsi.append(3)
        if j >= 0.6 and j < 0.8:
            imsi.append(4)
        if j >= 0.8 and j < 1:
            imsi.append(5)

    print (imsi)
Etime = time.time()
print(Etime - Stime)
