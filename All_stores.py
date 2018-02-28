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

#Scenario Parameters
num_years = 2050-2018
num_periods = 3
Periods_length = num_years/num_periods
Elec_price_var= 1.06 # 6% increase p.a.
Gas_price_var= 1.03 # 3% increase p.a.
PV_price_var= 0.90 # 10% decrease p.a.
#CHP_price_var=?

p_elec_mod_array =[]
p_gas_mod_array = []
for y in range(0, num_years):
    p_elec_mod_array.append(Elec_price_var**y)
    p_gas_mod_array.append(Gas_price_var**y)

p_elec_mod = []
p_gas_mod = []
for i in range(0, len(p_elec_mod_array), int(Periods_length)):
    p_elec_mod.append(np.average(p_elec_mod_array[i:i + int(Periods_length)]))
    p_gas_mod.append(np.average(p_gas_mod_array[i:i + int(Periods_length)]))


for store_id in Store_id_range[:1]:
    for n in range(0,num_periods):
        solution = PC.PV_CHP(store_id,p_elec_mod=p_elec_mod[n]).function_approx()
        
#        CAPEX_p.append(solution[0])
        OPEX_p.append(solution[1])
#        CARBON_p1.append(solution[2])
#        CARBON_p2.append(solution[3])
#        CAPEX_fit.append(solution[4])
#        OPEX_fit.append(solution[5])
#        CARBON_fit.append(solution[6])
    
