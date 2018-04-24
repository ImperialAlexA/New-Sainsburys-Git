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
import decompose_fun_2 as decfun
import Common.classStore as st
import xlsxwriter
import subprocess
import classPV_CHP as PC
import os

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

store = st.store(id_store)
price_table = 'Utility_Prices_SSL'
default_initial_time = datetime.datetime(2016,1,1)
default_final_time = datetime.datetime(2017,1,1)
time_start= int((default_initial_time-datetime.datetime(1970,1,1)).total_seconds()/60/30)
time_stop= int((default_final_time-datetime.datetime(1970,1,1)).total_seconds()/60/30)
store.getSimplePrice(time_start, time_stop, price_table)
store.getSimpleDemand(time_start, time_stop)        
store.getWeatherData(time_start, time_stop)
init_p_ele = store.p_ele
init_p_gas = store.p_gas
init_d_ele = store.d_ele

#p_elec_mod = 2.759
#p_gas_mod = 1.675
p_elec_mod = 1
p_gas_mod = 1
        
max_panels = pb.PVproblem(id_store).Max_panel_number(PV_tech_id)
panel_range = np.linspace(0,max_panels,15)

PV_capex_array = []
PV_size_array = []
for n_panels in panel_range:
    cur.execute('''SELECT * FROM PV_Technologies WHERE id=?''', (1,))
    dummy = cur.fetchall()
    PV_tech_price = dummy[0][2]
    PV_capex = PV_tech_price*n_panels
    PV_pb = pb.PVproblem(id_store)

    #arrays for 2D capex plots
    PV_capex_array.append(PV_capex)
    PV_size_array.append(n_panels)

    #elec and gas price modifiers

    PV_pb.elec_price = p_elec_mod*init_p_ele
    PV_pb.gas_price = p_gas_mod*init_p_gas

    #calculate solution, extract opex savings, carbon savings and electricity production
    PV_solution = PV_pb.SimulatePVonAllRoof(1,n_panels)
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

        CHP_pb.store.d_ele= abs(init_d_ele - PV_prod)

        CHP_solution = CHP_pb.SimpleOpti5NPV(tech_range=[tech_id],mod=[p_elec_mod,p_gas_mod,1,1], ECA_value = 0.26)
        
        CHP_opex = CHP_solution[4][0]
        CHP_Carbon=CHP_solution[5][2]

        PV_array.append(n_panels)
        CHP_array.extend(CHP_tech_size)
        Capex_array.append(CHP_tech_price+PV_capex)
        OPEX_array.append(CHP_opex+PV_opex)
        Carbon_array.append((PV_Carbon+CHP_Carbon))

        #arrays for 2D capex plots
        CHP_capex_array.append(CHP_tech_price)
        CHP_size_array.extend(CHP_tech_size)



ind_variable = np.array([PV_array,CHP_array], dtype=np.float64)
dep_variable1 = np.array(Capex_array, dtype=np.float64)
dep_variable2 = np.array(OPEX_array, dtype=np.float64)
dep_variable3 = np.array(Carbon_array, dtype=np.float64)
dep_variable4 = np.array(PV_capex_array, dtype=np.float64)
dep_variable5 = np.array(CHP_capex_array, dtype=np.float64)   

spl = 2

X_input = np.transpose(ind_variable)
Y_input = dep_variable3


[p_best, intercept_best, lb_best,ub_best,res_best_history,rel_err_best,res_best_by_domain,R2_best] = decfun.decompose(X_input,Y_input,spl)
#plt.plot(res_best_history)
print(p_best)
print(intercept_best)

fig = plt.figure(1)
ax = Axes3D(fig)
for i in np.arange(p_best.shape[1]):
            lb_IO = X_input >= lb_best[:,i]
            ub_IO = X_input <= ub_best[:,i]     
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
            
            x1 = np.linspace(min(X0[:,0]), max(X0[:,0]),len(X0[:,0]))
            x2 = np.linspace(min(X0[:,1]), max(X0[:,1]),len(X0[:,1]))
            x = [x1,x2]
            X1, X2 = np.meshgrid(x1, x2)
            zs = np.array([intercept_best[:,i] +  np.dot([x1,x2],p_best[:,i][:,None]) for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])
            Z = zs.reshape(X1.shape)
            ax.plot_wireframe(X1, X2, Z, rstride=10, cstride=10)


ax.scatter(X_input[:,0], X_input[:,1], Y_input,s = 10 ,c='r')

ax.set_xlabel('Number of PV panels', labelpad=8)
ax.set_ylabel('CHP size \n$kW$', labelpad=8)
ax.set_zlabel('Carbon savings \n$tonnes$',labelpad=8)
#ax.set_zlabel('OPEX savings \n$£K$',labelpad=8)
ax.tick_params(axis='both', which='major', pad=3)
ax.view_init(30, -60)
plt.show()
os.chdir('C:\\Users\\Anatole\\Documents\\GitHub\\New-Sainsburys-Git\\GIF')
# rotate the axes and update
for angle in np.arange(0, 360,2):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(0.001)
#    plt.savefig('picture'+str(angle).replace('.', '_').zfill(3),bbox_inches='tight',dpi=200)




