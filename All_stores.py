conda install -c anaconda paramiko# -*- coding: utf-8 -*-
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

#Scenario Parameters
#num_years = 2050-2018
#num_periods = 3
#Periods_length = num_years/num_periods
#Elec_price_var= 1.06 # 6% increase p.a.
#Gas_price_var= 1.03 # 3% increase p.a.
#PV_price_var= 0.90 # 10% decrease p.a.
#CHP_price_var=?

#p_elec_mod_array =[]
#p_gas_mod_array = []
#for y in range(0, num_years):
#    p_elec_mod_array.append(Elec_price_var**y)
#    p_gas_mod_array.append(Gas_price_var**y)
#
#p_elec_mod = []
#p_gas_mod = []
#for i in range(0, len(p_elec_mod_array), int(Periods_length)):
#    p_elec_mod.append(np.average(p_elec_mod_array[i:i + int(Periods_length)]))
#    p_gas_mod.append(np.average(p_gas_mod_array[i:i + int(Periods_length)]))


time_window = 30
stores = 2
year_start = 2020
year_stop = 2050
<<<<<<< HEAD
CO2_target = np.zeros(time_window)
=======
>>>>>>> 298dc1e4907d4896c9f840827a94cd7e46f52bc0

tech_range = ['PV', 'CHP','dummy','ppa']
modular = [1,0,1]
ppa_co2_coef = 0 #CO2 savings=ppa_co2_coef*ppa_size
ppa_opex_coef = 0 #opex savings=ppa_opex_coef*ppa_size


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


matrix = []
CAPEX = []

store_id = 2003
solution = PC.PV_CHP(store_id).function_approx()

Carbon_matrix =[]
OPEX_matrix = []
for store_id in Store_id_range[:stores]:
    Carbonh = []
    OPEXh = []
    for n in range(0,time_window):

        solution = PC.PV_CHP(store_id,p_elec_mod=p_elec_mod[n], p_gas_mod=p_gas_mod[n], PV_price_mod= PV_mod[n], CHP_price_mod=CHP_mod[n]).function_approx()
        OPEX_p = solution[1]
        CARBON_p1 = solution[2]
        CARBON_p2 = solution[3]
        CAPEX_p =  solution[0]
        
        Carbonh.append([CARBON_p1,ppa_co2_coef])
        OPEXh.append([OPEX_p, ppa_opex_coef])
        
    Carbon_matrix.append(Carbonh)
    OPEX_matrix.append(OPEXh)
    x_limit_bot_opex_matrix.append(x_limit_bot_opex_h)
    x_limit_top_opex_matrix.append(x_limit_top_opex_h)
    x_limit_bot_co2_matrix.append(x_limit_bot_co2_h)
    x_limit_top_co2_matrix.append(x_limit_top_co2_h)
    
    
#        matrix.append([OPEX_p, CARBON_p1, CARBON_p2, CAPEX_p])

############################################
### generate GAMS gdx file ###    
############################################
GAMS_model = "Strategic.gms"
ws = GamsWorkspace()
db =ws.add_database()

time_set = np.char.mod('%d', year)
store_set = np.char.mod('%d', Store_id_range[:stores])
tech_set = np.array(tech_range)

tech = db.add_set("tech",1,"")
t = db.add_set("t",1,"")
s = db.add_set("s",1,"")

#for n in time_set:
#    print(n)
#    t.add_record(n)


for n in tech_set:
    tech.add_record(n)
for m in time_set:
    t.add_record(m)
for z in store_set:
    s.add_record(z)
    
p0 = db.add_parameter_dc("K_co2", [d,tech,t,s], "")
p1 = db.add_parameter_dc("K_opex", [d,tech,t,s], "")
p2 = db.add_parameter_dc("K0_capex", [tech,t], "")
p3 = db.add_parameter_dc("K1_capex", [tech,t], "")
p4 = db.add_parameter_dc("CO2_savingTarget", [t], "")
p5 = db.add_parameter_dc("IO_modular", [tech], "")
p6 = db.add_parameter_dc("x_limit_bot_opex", [d,tech,t,s], "")
p7 = db.add_parameter_dc("x_limit_top_opex", [d,tech,t,s], "")
p8 = db.add_parameter_dc("x_limit_bot_co2", [d,tech,t,s], "")
p9 = db.add_parameter_dc("x_limit_top_co2", [d,tech,t,s], "")

       
for i in range(len(tech_set)):
    tech_i = tech_set[i]
    p6.add_record(tech_i).value = modular[i]
    
    for j in range(len(time_set)):
        time_j = time_set[j] 
#        p2.add_record(time_i).value = 
#        p3.add_record(time_j).value =

        for z in range(len(store_set)):
            store_z = store_set[z]
            
            for k in range(len(split_set)):
                split_k = split_set[k]
                
                p0.add_record([split_k, tech_i, time_j, store_z]).value = Carbon_matrix[z][j][i][k]      
                p1.add_record([split_k, tech_i, time_j, store_z]).value = OPEX_matrix[z][j][i][k]
                p6.add_record([split_k, tech_i, time_j, store_z]).value = x_limit_bot_opex_matrix[z][j][i][k]
                p7.add_record([split_k, tech_i, time_j, store_z]).value = x_limit_top_opex_matrix[z][j][i][k]
                p8.add_record([split_k, tech_i, time_j, store_z]).value = x_limit_bot_co2_matrix[z][j][i][k]
                p9.add_record([split_k, tech_i, time_j, store_z]).value = x_limit_top_co2_matrix[z][j][i][k]

for j in range(len(time_set)):
    time_j = time_set[j]   
    p4.add_record(time_j).value = CO2_target[j]

db.export("C:\\Users\\Anatole\\Documents\\GitHub\\New-Sainsburys-Git\\in.gdx ")

end = datetime.datetime.now()
print(end-start)






    
