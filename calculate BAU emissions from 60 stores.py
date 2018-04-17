# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:06:13 2018

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
import Connection.classClusterConnection as cc
from gams import GamsWorkspace
import xlsxwriter


scenario_names = ['two degrees','slow progression','steady state','consumer power']
scenarios = [[0.07,0.035,-0.0025,-0.015,-0.06],
             [0.06,0.03,-0.0025,-0.01,-0.04],
             [0.045,0.025,-0.0025,-0.0075,-0.03],
             [0.035,0.035,-0.005,-0.0125,-0.03]] #[Two degrees:[ele,gas,CHP,PV,cf], Slow progression:[], Steady State:[],Consumer power: []]



for scen in range(4):
    print(scenario_names[scen])

    database_path = "Sainsburys.sqlite"
    conn = sqlite3.connect(database_path)
    cur = conn.cursor()
    
    cur.execute('''SELECT Stores_id FROM Demand_Check Where Ele= {vn1} and Gas= {vn2}'''.format(vn1=1,vn2=1))
    Index = cur.fetchall()
    Store_id_range = np.array([elt[0] for elt in Index],dtype=np.float64)
    conn.close()
    
    Store_id_range = np.delete(Store_id_range,44) # =store 2017 not included because of errors
    
    time_window = 30 
    stores = 62
    year_start = 2020
    year_stop = 2050
    NG_True_False = False #True: Natural Gas is used as fuel for CHP, False:Biomethane is used as fuel for CHP
    time_window_length=(year_stop-year_start)/time_window
    
    CO2_target = np.zeros(time_window)
    tech_range = ['PV', 'CHP','dummy','ppa']
    modular = [1,0,1,1]
    split = 2 #SPlit the data for opex and carbon to generate coef of piecewise linear function
    
    ele_price_increase = scenarios[scen][0]  # % electricity price increase each year
    gas_price_increase = scenarios[scen][1] # % increase p.a. (modifies both natural gas and biomethane price)
    capex_reduction_CHP = scenarios[scen][2] # % capex reduction each year
    capex_reduction_PV = scenarios[scen][3] # % decrease p.a.
    cf_decrease = scenarios[scen][4] #% decrease of carbon factor each year
    ############################################
    ### generate coefficients ###    
    ############################################
    year = np.linspace(year_start,year_stop, time_window+1)[0:-1]
    ## array of ele_price_modifier 
    p_elec_mod = np.power(1+ ele_price_increase, np.linspace(year_start,year_stop, time_window+1)[0:-1] - year_start)
    p_gas_mod = np.power(1+ gas_price_increase, np.linspace(year_start,year_stop, time_window+1)[0:-1] - year_start)
    CHP_mod = np.power(1+capex_reduction_CHP, np.linspace(year_start,year_stop, time_window+1)[0:-1] - year_start)
    PV_mod = np.power(1+capex_reduction_PV, np.linspace(year_start,year_stop, time_window+1)[0:-1] - year_start)
    cf_mod = np.power(1+cf_decrease, np.linspace(year_start,year_stop, time_window+1)[0:-1] - year_start)

    BAU_op_cost = []    
    BAU_carbon = []
    time = []
    store_range = []
    for n in range(0,time_window):
        print('Time window:%d' %year[n])
        
        for store_id in Store_id_range[:stores]:
            print('store:%d' %store_id)
            
            time.append(year[n])
            store_range.append(store_id)
            solution = BBC.CHPproblem(store_id).SimpleOpti5NPV(tech_range=[1],mod=[p_elec_mod[n],p_gas_mod[n],1,1],ECA_value = 0.26)
            BAU_op_cost.append(solution[6])
            BAU_carbon.append(solution[5][0]*cf_mod[n])



    workbook=xlsxwriter.Workbook(scenario_names[scen]+' BAU.xlsx')
    worksheet=workbook.add_worksheet()
    
    headers = ['time','store','BAU Cost','BAU carbon']
    row = 0
    col = 0
    for h in headers:
        worksheet.write(row,col,h)
        col +=1
    
    row = 1
    col = 0
    for t in time:
        worksheet.write(row,col,t)
        row +=1
    row = 1
    for store in store_range:
        worksheet.write(row,col+1,store)
        row +=1
    row = 1
    for BAU_cost in BAU_op_cost:
        worksheet.write(row,col+2,BAU_cost)
        row +=1
        
    row = 1
    for carbon in BAU_carbon:
        worksheet.write(row,col+3,carbon)
        row +=1
    workbook.close()
    


