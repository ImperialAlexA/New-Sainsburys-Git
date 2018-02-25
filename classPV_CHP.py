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


class PV_CHP:
    
    def __init__(self,id_store):
        self.PV_tech_id = 1
        self.id_store = id_store
        
    def function_approx(self):
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
        
        
        for tech_id in range(1,20):
            cur.execute('''SELECT * FROM Technologies WHERE id=?''', (tech_id,))
            dummy = cur.fetchall()
            CHP_tech_size =(list(map(int, re.findall('\d+', dummy[0][1]))))
            CHP_tech_price = (dummy[0][2])
            CHP_opex=BBC.CHPproblem(self.id_store).SimpleOpti5NPV(tech_range=[tech_id,tech_id])[4][0]
            CHP_Carbon=BBC.CHPproblem(self.id_store).SimpleOpti5NPV(tech_range=[tech_id,tech_id])[5][2]
            
            for n_panels in panel_range:
                cur.execute('''SELECT * FROM PV_Technologies WHERE id=?''', (self.PV_tech_id,))
                dummy = cur.fetchall()
                PV_tech_price = dummy[0][2]
                PV_capex = PV_tech_price*n_panels
                PV_opex = pb.PVproblem(self.id_store).SimulatePVonAllRoof(self.PV_tech_id,n_panels)[1]
                PV_Carbon = pb.PVproblem(self.id_store).SimulatePVonAllRoof(self.PV_tech_id,n_panels)[4]
                
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
            
        def func2(x, a, b, c,d,e,f): 
            return a*x[0]+b*x[0]**2+e*x[0]**3+c*x[1]+d*x[1]**2+f*x[1]**3
        # a*np.exp((-b)*x[0])+c*np.exp((-d)*x[1])
        #a*x[0] + b*x[1] + c
        
        popt2, pcov2 = curve_fit(func2, ind_variable, dep_variable2)
        
        def func3(x, a, b, c): 
            return a*x[0] + b*x[1] + c
        # a*np.exp((-b)*x[0])+c*np.exp((-d)*x[1])
        #a*x[0]+b*x[0]**2+e*x[0]**3+c*x[1]+d*x[1]**2+f*x[1]**3
        
        k=np.argmax(dep_variable3)
        popt3, pcov3 = curve_fit(func3, ind_variable[:,:k], dep_variable3[:k])
        popt4, pcov4 = curve_fit(func3, ind_variable[:,k:], dep_variable3[k:])
        
        #Calculate and print prediction error indicators
        
        # =============================================================================
        # CAPEX
        # =============================================================================
        Target_test = dep_variable1
        Target_pred = func1(ind_variable, *popt1)
        Relative_error=[]
        Bias = []
        for i in range(0, len(Target_pred)):
            Relative_error.append((abs(abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)
            Bias.append(((abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)
        
        CAPEX_fit = [mean_absolute_error(Target_test, Target_pred),np.average(Relative_error),np.average(Bias),r2_score(Target_test, Target_pred)]
        # =============================================================================
        # OPEX
        # =============================================================================
        Target_test = dep_variable2
        Target_pred = func2(ind_variable, *popt2)
        Relative_error=[]
        Bias = []
        for i in range(0, len(Target_pred)):
            Relative_error.append((abs(abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)
            Bias.append(((abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)
        
        OPEX_fit = [mean_absolute_error(Target_test, Target_pred),np.average(Relative_error),np.average(Bias),r2_score(Target_test, Target_pred)]
        
        # =============================================================================
        # CARBON
        # =============================================================================
        
        Target_test = dep_variable3
        Target_pred = np.hstack((func3(ind_variable[:,:k], *popt3),func3(ind_variable[:,k:],*popt4)))
        Relative_error=[]
        Bias = []
        for i in range(0, len(Target_pred)):
            Relative_error.append((abs(abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)
            Bias.append(((abs(Target_pred[i])-abs(Target_test[i]))/max(abs(Target_pred[i]),abs(Target_test[i])))*100)
        
        CARBON_fit= [mean_absolute_error(Target_test, Target_pred),np.average(Relative_error),np.average(Bias),r2_score(Target_test, Target_pred)]
        return(popt1,popt2,popt3,popt4,CAPEX_fit,OPEX_fit,CARBON_fit)
