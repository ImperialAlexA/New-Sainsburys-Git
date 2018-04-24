# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 22:35:54 2018

@author: Anatole
"""

import classPV_CHP as PC
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
import xlsxwriter

store_id = 26
sol = PC.PV_CHP(store_id)

coef = sol.function_approx()

ind_variable = sol.ind_variable #PV CHP size array
ind_variable4 = sol.ind_variable4 #  PV size array
ind_variable5 = sol.ind_variable5 # CHP size array

dep_variable1 = sol.dep_variable1 #3d capex
dep_variable2 = sol.dep_variable2 # OPEX
dep_variable3 = sol.dep_variable3 # CARBON
dep_variable4 = sol.dep_variable4 # PV capex
dep_variable5 = sol.dep_variable5 #CHP capex


#fig = plt.figure(1)
#ax = Axes3D(fig)
#ax.scatter(ind_variable[0], ind_variable[1], dep_variable1, zdir='z', s=20, c='r', depthshade=True, label= 'data')
#x1 = np.linspace(min(ind_variable[0]), max(ind_variable[0]),len(ind_variable[0]))
#x2 = np.linspace(min(ind_variable[1]), max(ind_variable[1]),len(ind_variable[1]))
#x = [x1,x2]
#X1, X2 = np.meshgrid(x1, x2)
#zs = np.array([sol.func_linear_3d([x1, x2],*coef[0]) for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])
#Z = zs.reshape(X1.shape)
#ax.plot_wireframe(X1, X2, Z, rstride=10, cstride=10, label='fit')
#ax.set_xlabel('PV', labelpad=8)
#ax.set_ylabel('CHP', labelpad=8)
#ax.set_zlabel('Capex',labelpad=8)
#ax.tick_params(axis='both', which='major', pad=3)
#ax.legend()
#plt.show()
#
#fig = plt.figure(2)
#ax = Axes3D(fig)
#ax.scatter(ind_variable[0], ind_variable[1], dep_variable2, zdir='z', s=20, c='r', depthshade=True, label = 'data')
#x1 = np.linspace(min(ind_variable[0]), max(ind_variable[0]),len(ind_variable[0]))
#x2 = np.linspace(min(ind_variable[1]), max(ind_variable[1]),len(ind_variable[1]))
#x = [x1,x2]
#X1, X2 = np.meshgrid(x1, x2)
#zs = np.array([sol.func_poly([x1, x2],*coef[1]) for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])
#Z = zs.reshape(X1.shape)
#ax.plot_wireframe(X1, X2, Z, rstride=10, cstride=10, label = 'fit')
#ax.set_xlabel('PV', labelpad=8)
#ax.set_ylabel('CHP', labelpad=8)
#ax.set_zlabel('OPEX',labelpad=8)
#ax.tick_params(axis='both', which='major', pad=3)
#ax.legend()
#plt.show()
#
#fig = plt.figure(3)
#ax = Axes3D(fig)
#ax.scatter(ind_variable[0], ind_variable[1], dep_variable3, zdir='z', s=20, c='r', depthshade=True, label = 'data')
#x1 = np.linspace(min(ind_variable[0]), max(ind_variable[0]),len(ind_variable[0]))
#x2 = np.linspace(min(ind_variable[1]), max(ind_variable[1]),len(ind_variable[1]))
#x = [x1,x2]
#X1, X2 = np.meshgrid(x1, x2)
#zs = np.array([sol.func_poly([x1, x2],*coef[2]) for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])
#Z = zs.reshape(X1.shape)
#ax.plot_wireframe(X1, X2, Z, rstride=10, cstride=10, label = 'fit')
#ax.set_xlabel('PV', labelpad=8)
#ax.set_ylabel('CHP', labelpad=8)
#ax.set_zlabel('CARBON',labelpad=8)
#ax.tick_params(axis='both', which='major', pad=3)
#ax.legend()
#plt.show()

plt.figure(4)
X = np.linspace(min(ind_variable4), max(ind_variable4),len(ind_variable4))
Y_PV = sol.func_linear_2d(ind_variable4, *coef[3])
plt.plot(ind_variable4, Y_PV, 'b-', label='fit' )
plt.plot(ind_variable4, dep_variable4, 'ro', label='data')
plt.xlabel('Number of PV panels')
plt.ylabel('Capex £')
plt.legend()
plt.show()

plt.figure(5)
X = np.linspace(min(ind_variable5), max(ind_variable5),len(ind_variable5))
Y_CHP = sol.func_linear_2d(ind_variable5, *coef[4])
plt.plot(ind_variable5, Y_CHP, 'b-', label='fit' )
plt.plot(ind_variable5, dep_variable5, 'ro', label='data')
plt.xlabel('CHP size kW')
plt.ylabel('Capex £')
plt.legend()
plt.show()



workbook=xlsxwriter.Workbook('CAPEX_regression_plots.xlsx')
worksheet=workbook.add_worksheet()

headers = ['CHP_size','CHP_capex','CHP_capex_fit','PV_size','PV_capex','PV_capex_fit']
row = 0
col = 0
for h in headers:
    worksheet.write(row,col,h)
    col +=1

col=0
row=1
for i in ind_variable5:
    worksheet.write(row,col,i)
    row +=1

col=1
row=1
for j in dep_variable5:
    worksheet.write(row,col,j)
    row +=1

col=2
row=1
for k in Y_CHP:
    worksheet.write(row,col,k)
    row +=1
    
col=3
row=1
for k in ind_variable4:
    worksheet.write(row,col,k)
    row +=1
col=4
row=1
for l in dep_variable4:
    worksheet.write(row,col,l)
    row +=1
    
col=5
row=1
for l in Y_PV:
    worksheet.write(row,col,l)
    row +=1
workbook.close()