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
import datetime
from sklearn.metrics import mean_absolute_error, r2_score
import decompose_fun_2 as decfun
import Common.classStore as st


class PV_CHP:

    def __init__(self,id_store,p_elec_mod= None,p_gas_mod = None, PV_price_mod = None, CHP_price_mod = None):
        if p_elec_mod is not None:
            self.p_elec_mod = p_elec_mod
        else:
            self.p_elec_mod = 1

        if p_gas_mod is not None:
            self.p_gas_mod = p_gas_mod
        else:
            self.p_gas_mod = 1

        if PV_price_mod is not None:
            self.PV_price_mod = PV_price_mod
        else:
            self.PV_price_mod = 1
        if CHP_price_mod is not None:
            self.CHP_price_mod = CHP_price_mod
        else:
            self.CHP_price_mod = 1

#        if CHP_capex_mod is not None:
#            self.CHP_price
        self.PV_tech_id = 1
        self.id_store = id_store

        self.store = st.store(id_store)
        self.price_table = 'Utility_Prices_Aitor'
        default_initial_time = datetime.datetime(2016,1,1)
        default_final_time = datetime.datetime(2017,1,1)
        self.time_start= int((default_initial_time-datetime.datetime(1970,1,1)).total_seconds()/60/30)
        self.time_stop= int((default_final_time-datetime.datetime(1970,1,1)).total_seconds()/60/30)
        self.store.getSimplePrice(self.time_start, self.time_stop, self.price_table)
        self.store.getSimpleDemand(self.time_start, self.time_stop)        
        self.store.getWeatherData(self.time_start, self.time_stop)
        self.init_p_ele = self.store.p_ele
        self.init_p_gas = self.store.p_gas
        self.init_d_ele = self.store.d_ele
    
    def func_linear_2d(self,x, a, b):
        return a*x+b

    def func_linear_3d(self,x, a, b, c):
        return a*x[0] + b*x[1] + c

    def func_poly(self, x, a, b, c, d, e, f):
        return a*x[0]+b*x[0]**2+e*x[0]**3+c*x[1]+d*x[1]**2+f*x[1]**3

    def func_exp(self, x, a, b, c, d):
        return a*np.exp((-b)*x[0])+c*np.exp((-d)*x[1])

    def function_approx(self,spl = None):
        if spl is not None:# number of domains the data will be split in for the piecewise linear regression
            spl = spl
        else:
            spl = 2

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

        max_panels = pb.PVproblem(self.id_store).Max_panel_number(self.PV_tech_id)
        panel_range = np.linspace(0,max_panels,5)

        PV_capex_array = []
        PV_size_array = []
        for n_panels in panel_range:
            cur.execute('''SELECT * FROM PV_Technologies WHERE id=?''', (self.PV_tech_id,))
            dummy = cur.fetchall()
            PV_tech_price = dummy[0][2]*self.PV_price_mod
            PV_capex = PV_tech_price*n_panels
            PV_pb = pb.PVproblem(self.id_store)

            #arrays for 2D capex plots
            PV_capex_array.append(PV_capex)
            PV_size_array.append(n_panels)

            #elec and gas price modifiers

            PV_pb.elec_price = self.p_elec_mod*self.init_p_ele
            PV_pb.gas_price = self.p_gas_mod*self.init_p_gas

            #calculate solution, extract opex savings, carbon savings and electricity production
            PV_solution = PV_pb.SimulatePVonAllRoof(self.PV_tech_id,n_panels)
            PV_opex = PV_solution[1]
            PV_Carbon = PV_solution[4]
            PV_prod = PV_solution[6]

            CHP_capex_array =[]
            CHP_size_array =[]
            for tech_id in range(1,21):
                cur.execute('''SELECT * FROM Technologies WHERE id=?''', (tech_id,))
                dummy = cur.fetchall()
                CHP_tech_size =(list(map(int, re.findall('\d+', dummy[0][1]))))
                CHP_tech_price = (dummy[0][2])*self.CHP_price_mod
                CHP_pb = BBC.CHPproblem(self.id_store)


                CHP_pb.store.p_ele = self.p_elec_mod*self.init_p_ele
                CHP_pb.store.p_gas = self.p_gas_mod*self.init_p_gas

                CHP_pb.store.d_ele= abs(self.init_d_ele - PV_prod)
                
                CHP_solution = CHP_pb.SimpleOpti5NPV(tech_range=[tech_id],mod=[self.p_elec_mod,self.p_gas_mod,1,1], ECA_value = 0.26)
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


        self.ind_variable = np.array([PV_array,CHP_array], dtype=np.float64)
        self.dep_variable1 = np.array(Capex_array, dtype=np.float64)
        self.dep_variable2 = np.array(OPEX_array, dtype=np.float64)
        self.dep_variable3 = np.array(Carbon_array, dtype=np.float64)
        self.dep_variable4 = np.array(PV_capex_array, dtype=np.float64)
        self.dep_variable5 = np.array(CHP_capex_array, dtype=np.float64)
        print(self.dep_variable2[-1])
        #============CALCULATE CURVE COEFFICIENTS==============================
        #capex PV+CHP
        popt1, pcov1 = curve_fit(self.func_linear_3d, self.ind_variable, self.dep_variable1)
        #OPEX
        [p_best, intercept_best, lb_best,ub_best,res_best_history] = decfun.decompose(np.transpose(self.ind_variable),self.dep_variable2,spl)
        popt2 = np.vstack([p_best,intercept_best])
        opex_lb = lb_best
        opex_ub =ub_best
        #Carbon
        [p_best, intercept_best, lb_best,ub_best,res_best_history] = decfun.decompose(np.transpose(self.ind_variable),self.dep_variable3,spl)
        popt3 = np.vstack([p_best,intercept_best])
        carbon_lb = lb_best
        carbon_ub =ub_best
        #Capex PV
        self.ind_variable4 = np.array(PV_size_array, dtype=np.float64)
        popt4, pcov4 = curve_fit(self.func_linear_2d, self.ind_variable4, self.dep_variable4)
        #Capex CHP
        self.ind_variable5 = np.array(CHP_size_array, dtype=np.float64)
        popt5, pcov5 = curve_fit(self.func_linear_2d, self.ind_variable5, self.dep_variable5)

        return(popt1,popt2,popt3,popt4,popt5,opex_lb,opex_ub,carbon_lb,carbon_ub)

    def error(self): #Calculate and print prediction error indicators

        #NEED TO CHANGE FOR OPEX AND CARBON 
        coef = self.function_approx()
        # =============================================================================
        # CAPEX
        # =============================================================================

        Target_test = self.dep_variable1
        Target_pred = self.func_linear_3d(self.ind_variable, *coef[0])
        Relative_error=[]
        Bias = []
        for i in range(0, len(Target_pred)):
            Relative_error.append((abs(abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)
            Bias.append(((abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)

        CAPEX_fit = [mean_absolute_error(Target_test, Target_pred),np.average(Relative_error),np.average(Bias),r2_score(Target_test, Target_pred)]
        # =============================================================================
        # OPEX
        # =============================================================================
        Target_test = self.dep_variable2
        Target_pred = self.func_poly(self.ind_variable, *coef[1])
        Relative_error=[]
        Bias = []
        for i in range(0, len(Target_pred)):
            Relative_error.append((abs(abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)
            Bias.append(((abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)

        OPEX_fit = [mean_absolute_error(Target_test, Target_pred),np.average(Relative_error),np.average(Bias),r2_score(Target_test, Target_pred)]

        # =============================================================================
        # CARBON
        # =============================================================================

        Target_test = self.dep_variable3
        Target_pred = self.func_poly(self.ind_variable, *coef[2])
        Relative_error=[]
        Bias = []
        for i in range(0, len(Target_pred)):
            Relative_error.append((abs(abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)
            Bias.append(((abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)

        CARBON_fit= [mean_absolute_error(Target_test, Target_pred),np.average(Relative_error),np.average(Bias),r2_score(Target_test, Target_pred)]

        # =============================================================================
        # PV CAPEX
        # =============================================================================

        Target_test = self.dep_variable4
        Target_pred = self.func_linear_2d(self.ind_variable4, *coef[3])
        Relative_error=[]
        Bias = []
        for i in range(0, len(Target_pred)):
            Relative_error.append((abs(abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)
            Bias.append(((abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)

        PV_capex_fit= [mean_absolute_error(Target_test, Target_pred),np.average(Relative_error),np.average(Bias),r2_score(Target_test, Target_pred)]

        # =============================================================================
        # CHP CAPEX
        # =============================================================================

        Target_test = self.dep_variable5
        Target_pred = self.func_linear_2d(self.ind_variable5, *coef[4])
        Relative_error=[]
        Bias = []
        for i in range(0, len(Target_pred)):
            Relative_error.append((abs(abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)
            Bias.append(((abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)

        CHP_capex_fit= [mean_absolute_error(Target_test, Target_pred),np.average(Relative_error),np.average(Bias),r2_score(Target_test, Target_pred)]

        return(CAPEX_fit,OPEX_fit,CARBON_fit,PV_capex_fit, CHP_capex_fit)
