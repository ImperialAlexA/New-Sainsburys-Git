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
#import Connection.classClusterConnection as cc
from gams import GamsWorkspace

database_path = "Sainsburys.sqlite"
conn = sqlite3.connect(database_path)
cur = conn.cursor()

cur.execute('''SELECT Stores_id FROM Demand_Check Where Ele= {vn1} and Gas= {vn2}'''.format(vn1=1,vn2=1))
Index = cur.fetchall()
Store_id_range = np.array([elt[0] for elt in Index],dtype=np.float64)




time_window = 2
stores = 2
year_start = 2020
year_stop = 2050
CO2_target = [2,4]

tech_range = ['PV', 'CHP','dummy','ppa']
modular = [1,0,1,1]
ppa_co2_coef = np.zeros(4) #CO2 savings=ppa_co2_coef*ppa_size
ppa_opex_coef = np.zeros(4) #opex savings=ppa_opex_coef*ppa_size
split = 2 #SPlit the data for opex and carbon to generate coef of piecewise linear function

ele_price_increase = 0.06  # % electricity price increase each year
gas_price_increase = 0.03 # 3% increase p.a.
capex_reduction_CHP = -0.03 # % capex reduction each year
capex_reduction_PV = -0.10 # 10% decrease p.a.
############################################
### generate coefficients ###    
############################################
year = np.linspace(year_start,year_stop, time_window+1)[0:-1]
## array of ele_price_modifier 
p_elec_mod = np.power(1+ ele_price_increase, np.linspace(year_start,year_stop, time_window+1)[0:-1] - year_start)
p_gas_mod = np.power(1+ gas_price_increase, np.linspace(year_start,year_stop, time_window+1)[0:-1] - year_start)
CHP_mod = np.power(1+capex_reduction_CHP, np.linspace(year_start,year_stop, time_window+1)[0:-1] - year_start)
PV_mod = np.power(1+capex_reduction_PV, np.linspace(year_start,year_stop, time_window+1)[0:-1] - year_start)


Carbon_matrix =[]
OPEX_matrix = []
for store_id in Store_id_range[:stores]:
    Carbonh = []
    OPEXh = []
    Capex_p0 = []
    Capex_p1 = []
    for n in range(0,time_window):

        solution = PC.PV_CHP(store_id,p_elec_mod=p_elec_mod[n], p_gas_mod=p_gas_mod[n], PV_price_mod= PV_mod[n], CHP_price_mod=CHP_mod[n]).function_approx(spl=split)
        OPEX_p = solution[1]
        CARBON_p = solution[2]
        CAPEX_PV_p =  solution[3]
        CAPEX_CHP_p = solution[4]
        
        Carbonh.append(np.vstack([CARBON_p,ppa_co2_coef]))
        OPEXh.append(np.vstack([OPEX_p, ppa_opex_coef]))
        Capex_p0.append([CAPEX_PV_p[1],CAPEX_CHP_p[1],0,0]) # two last entries are for dummy and ppa
        Capex_p1.append([CAPEX_PV_p[0],CAPEX_CHP_p[0],0,0])
        
    Carbon_matrix.append(Carbonh)
    OPEX_matrix.append(OPEXh)
    
############################################
### generate GAMS gdx file ###    
############################################
GAMS_model = "Strategic.gms"
ws = GamsWorkspace()
db =ws.add_database()

time_set = np.char.mod('%d', year)
store_set = np.char.mod('%d', Store_id_range[:stores])
tech_set = np.array(tech_range)
split_set = np.char.mod('%d', np.arange(split**2))

tech = db.add_set("tech",1,"")
t = db.add_set("t",1,"")
s = db.add_set("s",1,"")
d = db.add_set("d",1,"")


for n in tech_set:
    tech.add_record(n)
for m in time_set:
    t.add_record(m)
for z in store_set:
    s.add_record(z)
for k in split_set:
    d.add_record(k)
    
p0 = db.add_parameter_dc("K_CO2", [tech,t,s,d], "")
p1 = db.add_parameter_dc("K_opex", [tech,t,s,d], "")
p2 = db.add_parameter_dc("K0_capex", [tech,t], "")
p3 = db.add_parameter_dc("K1_capex", [tech,t], "")
p4 = db.add_parameter_dc("CO2_savingTarget", [t], "")
#p5 = db.add_parameter_dc("Max_x", [tech,t,s], "")
p6 = db.add_parameter_dc("IO_modular", [tech], "")
       
for i in range(len(tech_set)):
    tech_i = tech_set[i]
    p6.add_record(tech_i).value = modular[i]
    
    for j in range(len(time_set)):
        time_j = time_set[j] 
        p2.add_record([tech_i, time_j]).value = Capex_p0[j][i]
        p3.add_record([tech_i, time_j]).value = Capex_p1[j][i]

        for z in range(len(store_set)):
            store_z = store_set[z]
            
            for k in range(len(split_set)):
                split_k = split_set[k]
                
                p0.add_record([tech_i, time_j, store_z, split_k]).value = Carbon_matrix[z][j][i][k]      
                p1.add_record([tech_i, time_j, store_z, split_k]).value = OPEX_matrix[z][j][i][k]

for j in range(len(time_set)):
    time_j = time_set[j]   
    p4.add_record(time_j).value = CO2_target[j]

db.export("C:\\Users\\Anatole\\Documents\\GitHub\\New-Sainsburys-Git\\input.gdx")






    
