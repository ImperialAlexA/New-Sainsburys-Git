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


database_path = "Sainsburys.sqlite"
conn = sqlite3.connect(database_path)
cur = conn.cursor()

id_store = 2001
PV_tech_id = 1
tech_name = []
tech_price = []
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
panel_range = np.linspace(0,max_panels,2)


for tech_id in range(1,20):
    cur.execute('''SELECT * FROM Technologies WHERE id=?''', (tech_id,))
    dummy = cur.fetchall()
    CHP_tech_size =(list(map(int, re.findall('\d+', dummy[0][1]))))
    CHP_tech_price = (dummy[0][2])
    CHP_opex=BBC.CHPproblem(id_store).SimpleOpti5NPV(tech_range=[tech_id,tech_id])[4][0]
    CHP_Carbon=BBC.CHPproblem(id_store).SimpleOpti5NPV(tech_range=[tech_id,tech_id])[5][2]
    
    for n_panels in panel_range:
        cur.execute('''SELECT * FROM PV_Technologies WHERE id=?''', (PV_tech_id,))
        dummy = cur.fetchall()
        PV_tech_price = dummy[0][2]
        PV_capex = dummy[0][2]*n_panels
        PV_opex = pb.PVproblem(id_store).SimulatePVonAllRoof(PV_tech_id,n_panels)[1]
        PV_Carbon = pb.PVproblem(id_store).SimulatePVonAllRoof(PV_tech_id,n_panels)[4]
        
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
    
def func2(x, a, b, c, d): 
    return a*x[0] + b*x[1] + c
# a*np.exp((-b)*x[0])+c*np.exp((-d)*x[1])
    
init_guess = [10000,1,10000,1]
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
popt3, pcov3 = curve_fit(func3, ind_variable, dep_variable3)

fig = plt.figure(3)
ax = Axes3D(fig)
ax.scatter(ind_variable[0], ind_variable[1], dep_variable3, zdir='z', s=20, c='r', depthshade=True)
x1 = np.linspace(min(ind_variable[0]), max(ind_variable[0]),len(ind_variable[0]))
x2 = np.linspace(min(ind_variable[1]), max(ind_variable[1]),len(ind_variable[1]))
x = [x1,x2]
X1, X2 = np.meshgrid(x1, x2)
zs = np.array([func3([x1, x2],*popt3) for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])
Z = zs.reshape(X1.shape)
ax.plot_wireframe(X1, X2, Z, rstride=10, cstride=10)
ax.set_xlabel('PV', labelpad=8)
ax.set_ylabel('CHP', labelpad=8)
ax.set_zlabel('Carbon',labelpad=8)
ax.tick_params(axis='both', which='major', pad=3)
plt.show()


# =============================================================================
# 
# PV_capex_array = []
# store_index = 2001
# #Max_panel = pb.PVproblem(store_index).OptiPVpanels()[4]
# 
# CHP_capex = BBC.CHPproblem(store_index).SimpleOpti5NPV(mod = [10.6/8.787,2.35/2.618,1,1], ECA_value = 0.26, table_string = 'Utility_Prices_Aitor _NoGasCCL')[4][5]
# for n_panel in range(1,10):
#     PV_capex_array.append(pb.PVproblem(store_index).SimulatePVonAllRoof(1,n_panel)[2])
#     
# =============================================================================
