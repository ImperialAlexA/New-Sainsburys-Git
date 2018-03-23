# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 22:33:41 2018

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


database_path = "Sainsburys.sqlite"
conn = sqlite3.connect(database_path)
cur = conn.cursor()
conn.commit()
PV_tech_price = []
CHP_tech_size = []
CHP_tech_price = []
PV_capex = []
PV_array = []
CHP_array = []
Capex_array = []
OPEX_array = []
Carbon_array = []
id_store =26
PV_tech_id =1

max_panels = pb.PVproblem(id_store).Max_panel_number(PV_tech_id)
panel_range = np.linspace(0,max_panels,10)

PV_capex_array = []
PV_size_array = []
for n_panels in panel_range:
    cur.execute('''SELECT * FROM PV_Technologies WHERE id=?''', (PV_tech_id,))
    dummy = cur.fetchall()
    PV_tech_price = dummy[0][2]
    PV_capex = PV_tech_price*n_panels
    PV_pb = pb.PVproblem(id_store)
    
    #arrays for 2D capex plots
    PV_capex_array.append(PV_capex)
    PV_size_array.append(n_panels)
    
    #calculate solution, extract opex savings, carbon savings and electricity production
    PV_solution = PV_pb.SimulatePVonAllRoof(PV_tech_id,n_panels)
    PV_opex = PV_solution[1]
    PV_Carbon = PV_solution[4]
    PV_prod = PV_solution[6]
    

    CHP_capex_array =[]
    CHP_size_array =[]
    for tech_id in range(1,21):
        cur.execute('''SELECT * FROM Technologies WHERE id=?''', (tech_id,))
        dummy = cur.fetchall()
        CHP_tech_size =(list(map(int, re.findall('\d+', dummy[0][1]))))
        CHP_tech_price = (dummy[0][2])
        CHP_pb = BBC.CHPproblem(id_store)
        
        CHP_solution = CHP_pb.SimpleOpti5NPV(tech_range=[tech_id,tech_id],mod = [11.9/8.787,2.35/2.618,1,1], ECA_value = 0.26, table_string = 'Utility_Prices_Aitor _NoGasCCL')
        CHP_opex = CHP_solution[4][0]
        CHP_Carbon=CHP_solution[5][2]

        PV_array.append(n_panels)
        CHP_array.extend(CHP_tech_size)
        Capex_array.append(CHP_tech_price+PV_capex)
        OPEX_array.append(CHP_opex+PV_opex)
        Carbon_array.append(PV_Carbon+CHP_Carbon)
  
        #arrays for 2D capex plots
        CHP_capex_array.append(CHP_tech_price)
        CHP_size_array.extend(CHP_tech_size)



ind_variable = np.array([PV_array,CHP_array], dtype=np.float64)
dep_variable1 = np.array(Capex_array, dtype=np.float64)
dep_variable2 = np.array(OPEX_array, dtype=np.float64)
dep_variable3 = np.array(Carbon_array, dtype=np.float64)
dep_variable4 = np.array(PV_capex_array, dtype=np.float64)
dep_variable5 = np.array(CHP_capex_array, dtype=np.float64)   


import decompose_fun_2 as decfun

spl = 2

X_input = np.transpose(ind_variable)
Y_input = dep_variable2



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



#import xlsxwriter
#workbook = xlsxwriter.Workbook('test.xlsx')
#worksheet = workbook.add_worksheet()
#row = 0
#col =0
#for PV, CHP in np.transpose(ind_variable):
#    worksheet.write(row,col, PV)
#    worksheet.write(row,col+1, CHP)
#    row +=1 
#row=0
#for carbon in np.transpose(dep_variable3):
#    worksheet.write(row, col+3, carbon)
#    row+=1
#workbook.close()