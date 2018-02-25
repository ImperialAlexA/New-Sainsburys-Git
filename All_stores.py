# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 16:44:28 2018

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


database_path = "Sainsburys.sqlite"
conn = sqlite3.connect(database_path)
cur = conn.cursor()

cur.execute('''SELECT Stores_id FROM Demand_Check Where Ele= {vn1} and Gas= {vn2}'''.format(vn1=1,vn2=1))
Index = cur.fetchall()
Store_id_range = np.array([elt[0] for elt in Index],dtype=np.float64)

CAPEX_fit = []
OPEX_fit = []
CARBON_fit = []
CAPEX_p = []
OPEX_p = []
CARBON_p1 = []
CARBON_p2 = []


for store_id in Store_id_range:
    print(store_id)
    solution = PC.PV_CHP(store_id).function_approx()
    CAPEX_p.append(solution[0])
    OPEX_p.append(solution[1])
    CARBON_p1.append(solution[2])
    CARBON_p2.append(solution[3])
    CAPEX_fit.append(solution[4])
    OPEX_fit.append(solution[5])
    CARBON_fit.append(solution[6])
    
plt.figure(1)   
plt.scatter(Store_id_range[:14],np.array(CARBON_fit)[:,3])

plt.figure(2)
plt.scatter(Store_id_range[:5], np.array(OPEX_fit)[:,3])
