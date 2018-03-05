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


time_window = 2
stores = 2
year_start = 2020
year_stop = 2050
tech_range = ['PV', 'CHP','dummy']
modular = [1,0,1]

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
        
        Carbonh.append(CARBON_p1)
        OPEXh.append(OPEX_p)
        
    Carbon_matrix.append(Carbonh)
    OPEX_matrix.append(OPEXh)
    
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
    
p0 = db.add_parameter_dc("K_CO2", [tech,t,s], "")
p1 = db.add_parameter_dc("K_opex", [tech,t,s], "")
p2 = db.add_parameter_dc("K0_capex", [tech,t], "")
p3 = db.add_parameter_dc("K1_capex", [tech,t], "")
p4 = db.add_parameter_dc("CO2_savingTarget", [t], "")
p5 = db.add_parameter_dc("Max_x", [tech,t,s], "")
p6 = db.add_parameter_dc("IO_modular", [tech], "")
       
for i in range(len(tech_set)):
    tech_i = tech_set[i]
    p6.add_record(tech_i).value = modular[i]
    
    for j in range(len(time_set)):
        time_j = time_set[j] 
#        p2.add_record(time_i).value = 
#        p3.add_record(time_j).value =

        for z in range(len(store_set)):
            store_z = store_set[z]
                      
            p0.add_record([tech_i, time_j, store_z]).value = Carbon_matrix[z][j][i]         
            p1.add_record([tech_i, time_j, store_z]).value = OPEX_matrix[z][j][i] 
            p5.add_record([tech_i, time_j, store_z]).value = 



    
    for j in range()
        p0.add_record(tech_i, time_i, store_i).value = matrix[i][0]
    p1.add_record(time_i).value = matrix[i][1]
    p2.add_record(time_i).value = matrix[i][2][0]
    p3.add_record(time_i).value = matrix[i][2][1]
    p4.add_record(time_i).value = matrix[i][2][2]
#    p3.add_record(time_i).value = matrix[i]
    p5.add_record(time_i).value = matrix[i][3][0]
    p6.add_record(time_i).value = matrix[i][3][1]
    p7.add_record(time_i).value = matrix[i][3][2]
    p8.add_record(time_i).value = matrix[i][4][0]
    p9.add_record(time_i).value = matrix[i][4][1]
    p10.add_record(time_i).value = matrix[i][4][2]
    p11.add_record(time_i).value = matrix[i][5][0]    
    p12.add_record(time_i).value = matrix[i][5][1]   
    p13.add_record(time_i).value = matrix[i][5][2]   
db.export("input.gdx")






    
