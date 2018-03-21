#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:32:10 2018

@author: Alex
"""

import Solvers.classPVProblem as pb
import Solvers.classCHPProblemnew as BBC
import pandas as pd
import sqlite3
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import curve_fit
import datetime
from sklearn.metrics import mean_absolute_error, r2_score

start = datetime.datetime.now()
database_path = "Sainsburys.sqlite"
conn = sqlite3.connect(database_path)
cur = conn.cursor()

id_store = 2003
PV_tech_id = 1
PV_tech_price = []
CHP_tech_size = []
CHP_tech_price = []
PV_capex = []
PV_array = []
CHP_array = []
Capex_array = []
OPEX_array = []
Carbon_array = []

max_panels = pb.PVproblem(id_store).Max_panel_number(PV_tech_id)
panel_range = np.linspace(0,max_panels,10)

for n_panels in panel_range:
    cur.execute('''SELECT * FROM PV_Technologies WHERE id=?''', (PV_tech_id,))
    dummy = cur.fetchall()
    PV_tech_price = dummy[0][2]
    PV_capex = PV_tech_price*n_panels
    PV_pb = pb.PVproblem(id_store)
    PV_solution = PV_pb.SimulatePVonAllRoof(PV_tech_id,n_panels)
    PV_opex = PV_solution[1]
    PV_Carbon = PV_solution[4]
    PV_prod = PV_solution[6]
    
    old_d_ele = PV_pb.store.d_ele
    
    for tech_id in range(1,20):
        cur.execute('''SELECT * FROM Technologies WHERE id=?''', (tech_id,))
        dummy = cur.fetchall()
        CHP_tech_size =(list(map(int, re.findall('\d+', dummy[0][1]))))
        CHP_tech_price = (dummy[0][2])
        
         
        CHP_pb = BBC.CHPproblem(id_store)
        CHP_pb.store.d_ele= abs(old_d_ele - PV_prod)
        CHP_solution = CHP_pb.SimpleOpti5NPV(tech_range=[tech_id,tech_id],mod = [11.9/8.787,2.35/2.618,1,1], ECA_value = 0.26, table_string = 'Utility_Prices_Aitor _NoGasCCL')
        CHP_opex = CHP_solution[4][0]
        CHP_Carbon=CHP_solution[5][2]

        PV_array.append(n_panels)
        CHP_array.extend(CHP_tech_size)
        Capex_array.append(CHP_tech_price+PV_capex)
        OPEX_array.append(CHP_opex+PV_opex)
        Carbon_array.append(PV_Carbon+CHP_Carbon)



ind_variable = [PV_array,CHP_array]
dep_variable1 = Capex_array
dep_variable2 = OPEX_array
dep_variable3 = Carbon_array
ind_variable = np.array(ind_variable, dtype=np.float64)
dep_variable1 = np.array(dep_variable1, dtype=np.float64)
dep_variable2 = np.array(dep_variable2, dtype=np.float64)
dep_variable3 = np.array(dep_variable3, dtype=np.float64)

def func1(x, a, b, c): 
    return a*x[0] + b*x[1] + c
#a*np.exp((-b)*x[0])+c*np.exp((-d)*x[1])
popt1, pcov1 = curve_fit(func1, ind_variable, dep_variable1)

fig = plt.figure(1)
ax = Axes3D(fig)
ax.scatter(ind_variable[0], ind_variable[1], dep_variable1, zdir='z', s=20, c='r', depthshade=True)
x1 = np.linspace(min(ind_variable[0]), max(ind_variable[0]),len(ind_variable[0]))
x2 = np.linspace(min(ind_variable[1]), max(ind_variable[1]),len(ind_variable[1]))
x = [x1,x2]
X1, X2 = np.meshgrid(x1, x2)
zs = np.array([func1([x1, x2],*popt1) for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])
Z = zs.reshape(X1.shape)
ax.plot_wireframe(X1, X2, Z, rstride=10, cstride=10)
ax.set_xlabel('PV', labelpad=8)
ax.set_ylabel('CHP', labelpad=8)
ax.set_zlabel('Capex',labelpad=8)
ax.tick_params(axis='both', which='major', pad=3)
plt.show()
    
def func2(x, a, b, c,d,e,f): 
    return a*x[0]+b*x[0]**2+e*x[0]**3+c*x[1]+d*x[1]**2+f*x[1]**3
# a*np.exp((-b)*x[0])+c*np.exp((-d)*x[1])
#a*x[0] + b*x[1] + c
init_guess = [1,1,1,1,1,1]
popt2, pcov2 = curve_fit(func2, ind_variable, dep_variable2, init_guess)

fig = plt.figure(2)
ax = Axes3D(fig)
ax.scatter(ind_variable[0], ind_variable[1], dep_variable2, zdir='z', s=20, c='r', depthshade=True)
x1 = np.linspace(min(ind_variable[0]), max(ind_variable[0]),len(ind_variable[0]))
x2 = np.linspace(min(ind_variable[1]), max(ind_variable[1]),len(ind_variable[1]))
x = [x1,x2]
X1, X2 = np.meshgrid(x1, x2)
zs = np.array([func2([x1, x2],*popt2) for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])
Z = zs.reshape(X1.shape)
ax.plot_wireframe(X1, X2, Z, rstride=10, cstride=10)
ax.set_xlabel('PV', labelpad=8)
ax.set_ylabel('CHP', labelpad=8)
ax.set_zlabel('OPEX',labelpad=8)
ax.tick_params(axis='both', which='major', pad=3)
plt.show()

def func3(x, a, b, c): 
    return a*x[0] + b*x[1] + c
# a*np.exp((-b)*x[0])+c*np.exp((-d)*x[1])
#a*x[0]+b*x[0]**2+e*x[0]**3+c*x[1]+d*x[1]**2+f*x[1]**3

k=np.argmax(dep_variable3)
popt3, pcov3 = curve_fit(func3, ind_variable[:,:k], dep_variable3[:k])
popt4, pcov4 = curve_fit(func3, ind_variable[:,k:], dep_variable3[k:])

fig = plt.figure(3)
ax = Axes3D(fig)
ax.scatter(ind_variable[0], ind_variable[1], dep_variable3, zdir='z', s=20, c='r', depthshade=True)

x1 = np.linspace(min(ind_variable[0,:k]), max(ind_variable[0,:k]),len(ind_variable[0,:k]))
x2 = np.linspace(min(ind_variable[1,:k]), max(ind_variable[1,:k]),len(ind_variable[1,:k]))
x = [x1,x2]
X1, X2 = np.meshgrid(x1, x2)
zs = np.array([func3([x1, x2],*popt3) for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])
Z = zs.reshape(X1.shape)
ax.plot_wireframe(X1, X2, Z, rstride=10, cstride=10)

x01 = np.linspace(min(ind_variable[0,k:]), max(ind_variable[0,k:]),len(ind_variable[0,k:]))
x02 = np.linspace(min(ind_variable[1,k:]), max(ind_variable[1,k:]),len(ind_variable[1,k:]))
x0 = [x01,x02]
X01, X02 = np.meshgrid(x01, x02)
zs0 = np.array([func3([x01, x02],*popt4) for x01,x02 in zip(np.ravel(X01), np.ravel(X02))])
Z0 = zs0.reshape(X01.shape)
ax.plot_wireframe(X01, X02, Z0, rstride=10, cstride=10)

ax.set_xlabel('PV', labelpad=8)
ax.set_ylabel('CHP', labelpad=8)
ax.set_zlabel('Carbon',labelpad=8)
ax.tick_params(axis='both', which='major', pad=3)
plt.show()

#Calculate and print prediction error indicators

# =============================================================================
# CAPEX
# =============================================================================
Target_test = dep_variable1
Target_pred = func1(ind_variable, *popt1)
Relative_error=[]
Bias = []
for i in range(0, len(Target_pred)):
    Relative_error.append((abs(abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)
    Bias.append(((abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)
print('------CAPEX')
print("Mean absolute error: %.2f" % mean_absolute_error(Target_test, Target_pred))
print("Mean relative error: %.2f %%" % np.average(Relative_error))
print("Bias: %.2f %%" %np.average(Bias))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Target_test, Target_pred))


# =============================================================================
# OPEX
# =============================================================================
Target_test = dep_variable2
Target_pred = func2(ind_variable, *popt2)
Relative_error=[]
Bias = []
for i in range(0, len(Target_pred)):
    Relative_error.append((abs(abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)
    Bias.append(((abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)
print('------OPEX')
print("Mean absolute error: %.2f" % mean_absolute_error(Target_test, Target_pred))
print("Mean relative error: %.2f %%" % np.average(Relative_error))
print("Bias: %.2f %%" %np.average(Bias))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Target_test, Target_pred))

# =============================================================================
# CARBON
# =============================================================================

Target_test = dep_variable3
Target_pred = np.hstack((func3(ind_variable[:,:k], *popt3),func3(ind_variable[:,k:],*popt4)))
Relative_error=[]
Bias = []
for i in range(0, len(Target_pred)):
    Relative_error.append((abs(abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)
    Bias.append(((abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)
print('------CARBON')
print("Mean absolute error: %.2f" % mean_absolute_error(Target_test, Target_pred))
print("Mean relative error: %.2f %%" % np.average(Relative_error))
print("Bias: %.2f %%" %np.average(Bias))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Target_test, Target_pred))

end= datetime.datetime.now()
print('time:%s'%(end-start))




import decompose_fun_2 as decfun

spl = 2
X_input = np.transpose(ind_variable)
Y_input = dep_variable2


dim = 2
spl = 2
n_iter = 15  
n = 95
X_input = np.random.rand(n,dim)
X_input = np.transpose(ind_variable)
Y_input = dep_variable2


[p_best, intercept_best, lb_best,ub_best,res_best_history] = decfun.decompose(X_input,Y_input,spl)
plt.plot(res_best_history)
print(p_best)



