# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 15:19:47 2018

@author: Anatole
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
import decompose_fun_2 as decfun

df = pd.read_excel('test.xlsx')
matrix =np.transpose(df.as_matrix())
ind_variable=np.array(matrix[:-1], dtype=np.float64)
dep_variable3 = np.array(matrix[-1], dtype=np.float64)


spl = 2
X_input = np.transpose(ind_variable)
Y_input = dep_variable3

[p_best, intercept_best, lb_best,ub_best,res_best_history] = decfun.decompose(X_input,Y_input,spl)
plt.plot(res_best_history)
print(p_best)

for i in range(p_best.shape[1]):
            lb_IO = X_input > lb_best[:,i]
            ub_IO = X_input < ub_best[:,i]     
            mask = np.logical_and(np.all(lb_IO, axis = 1), np.all(ub_IO, axis = 1))
            X0 = X_input[mask]
            print(X0.shape)
            Y_fit = intercept_best[:,i] +  np.dot(X0,p_best[:,i][:,None])
            if i == 0:
               X_tot = X0
               Y_tot = Y_fit
            else:    
                X_tot= np.append(X_tot,X0, axis =0)
                Y_tot=np.append(Y_tot,Y_fit)
#                fig = plt.figure()
#                ax = Axes3D(fig)
#                ax.scatter(X_input[:,0], X_input[:,1], Y_input,s = 1)
#                ax.scatter(X_tot[:,0], X_tot[:,1], Y_tot, c = 'r', s = 1)    


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_input[:,0], X_input[:,1], Y_input,s = 1)
ax.scatter(X_tot[:,0], X_tot[:,1], Y_tot, c = 'r', s = 5)
print(intercept_best)