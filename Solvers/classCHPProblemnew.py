# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:32:24 2017

@author: nl211
"""

import sqlite3
import numpy as np
import datetime
import calendar
import os
import sys
scriptpath = "..\\Common" # This is written in the Windows way of specifying paths, hopefully it works on Linux?
sys.path.append(os.path.abspath(scriptpath))
import Common.classStore as st # Module is in seperate folder, hence the elaboration
import Common.classTech as tc
#from pyomo.environ import * # Linear programming module
#import pyomo as pyo
#import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory # Solver
import time # To time code, simply use start = time.clock() then print start - time.clock()


#database_path = "C:\\Users\\GOBS\\Dropbox\\Uni\Other\\UROP - Salvador\\Niccolo_project\\Code\\Sainsburys.sqlite" # Path to database file
database_path = "Sainsburys.sqlite" # Path to database file


class CHPproblem:
    
    def __init__(self, store_id):
        self.store = st.store(store_id)
        self.price_table = 'Utility_Prices_Aitor'
        default_initial_time = datetime.datetime(2015,1,1)
        default_final_time = datetime.datetime(2016,1,1)
        self.time_start= int((default_initial_time-datetime.datetime(1970,1,1)).total_seconds()/60/30)
        self.time_stop= int((default_final_time-datetime.datetime(1970,1,1)).total_seconds()/60/30)
        self.store.getSimplePrice(self.time_start, self.time_stop, self.price_table)
        self.store.getSimpleDemand(self.time_start, self.time_stop)
        self.boiler_eff = 0.87
        self.hidden_costs = 353000
        #self.financial_lifetime = 15
        self.discount_rate = 0.09
        self.CHPQI_threshold = 105



    # Find the best tech by iterating over each technology then determining
    # which is best. 3 different optimisation methods are provided, of which
    # the first 2 aren't even optimisations (only last uses Pyomo to optimise
    # CHP operation) - TODO: add Pyomo!
    
    def SimpleOpti5NPV(self, method = None, tech_range = None, time_start = None, time_stop = None, table_string = None, ECA_value = 0, uncertainty = None, mod = None):
        if time_start is not None or time_stop is not None or table_string is not None: 
                old_time_start = self.time_start; old_time_stop = self.time_stop;  old_price_table = self.price_table
                self.putUtility(time_start =time_start, time_stop = time_stop, table_string=table_string)            
        if tech_range is not None:
            array_tech = tech_range
        else:
            array_tech = range(1,20)    
                    
        hidden_costs = self.hidden_costs
        discount_rate =  self.discount_rate                 
        
        optimal_objective = -1000000
        opti_tech = -1
        opti_tech_name = 'None'
       
        for id_tech_index in array_tech:
            tech_id = id_tech_index
            self.putTech(tech_id)
            tech_name = self.tech.tech_name
            tech_price = self.tech.tech_price*(1-ECA_value)          
            tech_lifetime = self.tech.lifetime            
            
            if method is not None:
                methodToRun = method
            else:
                methodToRun = 1
            if methodToRun == 1:        
                [BAU_op_cost_HH_pound, op_cost_HH_pound, part_load, CHPQI] = self.SimpleOptiControl(uncertainty = uncertainty, mod = mod)  
            elif methodToRun == 2:
                [BAU_op_cost_HH_pound, op_cost_HH_pound, part_load, CHPQI] = self.LoadFollowControl(uncertainty = uncertainty, mod = mod)  
            elif methodToRun == 3:
                [BAU_op_cost_HH_pound, op_cost_HH_pound, opti_CHPQI , CHPQI] = self.SebastianControl(tech_id)
            elif methodToRun == 4:
                [BAU_op_cost_HH_pound, op_cost_HH_pound, part_load, CHPQI] = self.LoadFollowControlOnOff(uncertainty = uncertainty, mod = mod)
            else:
                print("Method chosen is wrong")
                raise ValueError

            # Calculate finantial
            numb_years = (self.time_stop-self.time_start)/2/24/365
            year_op_cost = sum(op_cost_HH_pound)/numb_years
            year_BAU_cost = sum(BAU_op_cost_HH_pound)/numb_years
            Total_capex = tech_price + hidden_costs            
            [year_savings, payback, NPV5savings, ROI, Cum_disc_cash_flow] = self.calculate_financials(discount_rate, tech_lifetime, year_BAU_cost, year_op_cost, Total_capex)
            
            objective = NPV5savings
            # Check if this is the optimum technology
            if objective > optimal_objective:
                opti_tech = id_tech_index
                opti_tech_name = tech_name
                opti_CHPQI = CHPQI
                opti_part_load = part_load
                optimal_objective = objective
                opti_year_savings = year_savings
                opti_payback = payback
                opti_NPV5savings = NPV5savings
                opti_ROI = ROI
                opti_capex = Total_capex
                opti_Cum_disc_cash_flow = Cum_disc_cash_flow
                [opti_BAU_carbon, opti_CHP_carbon, opti_carbon_savings,biomethane_usage] = self.calculate_carbon(opti_part_load)

        
        #restore previous values        
        if time_start is not None or time_stop is not None or table_string is not None: 
                self.putUtility(time_start =old_time_start, time_stop = old_time_stop, table_string=old_price_table)
        financials = [opti_year_savings, opti_payback, opti_NPV5savings, opti_ROI, opti_Cum_disc_cash_flow, opti_capex]
        carbon = [opti_BAU_carbon, opti_CHP_carbon, opti_carbon_savings,biomethane_usage]
        
        return(opti_tech, opti_tech_name, opti_CHPQI, opti_part_load, financials, carbon)
    
    
    
    # find the the best technology considering the CHPQI benefits and assuming advanced control strategies
    def CHPQIOpti5NPV(self, tech_range = None, time_start = None, time_stop = None, table_string = None, uncertainty = None):
        if time_start is not None or time_stop is not None or table_string is not None: 
                old_time_start = self.time_start; old_time_stop = self.time_stop;  old_price_table = self.price_table
                self.putUtility(time_start =time_start, time_stop = time_stop, table_string=table_string)          
    
        NPV_years = 5 
        hidden_costs = self.hidden_costs
        financial_lifetime = self.financial_lifetime
        discount_rate =  self.discount_rate                 
        conn = sqlite3.connect(database_path)
        cur = conn.cursor()
        
        optimal_savings = -1000000
        opti_payback = -1
        opti_tech = -1
        opti_tech_name = 'None'
        CHPQI_threshold = 105
        ECA_value = 0.26
        
        if tech_range is not None:
            array_tech = tech_range
        else:
            array_tech = range(13,20)
        for id_tech_index in array_tech: 
            
            tech_id = id_tech_index
            
            cur.execute('''SELECT * FROM Technologies WHERE id=?''', (tech_id,))
            dummy = cur.fetchall()
            tech_name = dummy[0][1]
            tech_price = dummy[0][2]            
            tech_price_CHPQI = tech_price*(1-ECA_value)           
            
            results = self.SimpleOptiControl(tech_id)              
            year_BAU_cost = sum(results[0])
            year_op_cost = sum(results[1])
            CHPQI = results[3]
            
            results_CHPQI = self.SimpleOptiControl(tech_id = tech_id, CHPQI_IO = 1, table_string =  'Utility_Prices_Aitor_NoGasCCL' ) 
            year_op_cost_CHPQI = sum(results_CHPQI[1]) 
            CHPQI_CHPQI = results_CHPQI[3]
            if table_string is not None:
                self.price_table = table_string   
            else:
                self.price_table = 'Utility_Prices_Aitor' 
            self.store.getSimplePrice(self.time_start, self.time_stop, self.price_table)

            # Calculate annualised CAPEX
            numb_years = (self.time_stop-self.time_start)/2/24/365
            year_op_cost = year_op_cost/numb_years
            year_op_cost_CHPQI = year_op_cost_CHPQI/numb_years
            year_BAU_cost = year_BAU_cost/numb_years
            year_savings = year_BAU_cost - year_op_cost 
            year_savings_CHPQI = year_BAU_cost - year_op_cost_CHPQI 
            Total_capex = tech_price+hidden_costs 
            Total_capex_CHPQI = tech_price_CHPQI+hidden_costs 
            payback = Total_capex/year_savings
            payback_CHPQI = Total_capex_CHPQI/year_savings_CHPQI
            ann_capex = -np.pmt(discount_rate, financial_lifetime, Total_capex)
            ann_capex_CHPQI = -np.pmt(discount_rate, financial_lifetime, Total_capex_CHPQI)
            year_cost = year_op_cost  + ann_capex
            year_cost_CHPQI = year_op_cost_CHPQI  + ann_capex_CHPQI
            tot_OPTI_cost  = -np.npv(discount_rate, np.array([year_cost]*NPV_years))
            tot_OPTI_cost_CHPQI  = -np.npv(discount_rate, np.array([year_cost_CHPQI]*NPV_years))
            tot_BAU_cost = -np.npv(discount_rate, np.array([year_BAU_cost]*NPV_years)) 
            savings = tot_OPTI_cost - tot_BAU_cost
            savings_CHPQI = tot_OPTI_cost_CHPQI - tot_BAU_cost
            if savings_CHPQI > savings and CHPQI_CHPQI >= CHPQI_threshold:
                savings = savings_CHPQI
                payback = payback_CHPQI
                CHPQI = CHPQI_CHPQI 
                    
            # Check if this is the optimum technology
            if savings > optimal_savings:
                opti_tech = id_tech_index
                opti_tech_name = tech_name
                optimal_savings = savings
                opti_payback = payback 
                opti_CHPQI = CHPQI

        if time_start is not None or time_stop is not None or table_string is not None: 
                self.putUtility(time_start =old_time_start, time_stop = old_time_stop, table_string=old_price_table)          
        
        return(opti_tech, opti_tech_name, optimal_savings, opti_payback, opti_CHPQI)    

    # Find the optimal part load of tech. time start and time stop need to be passed as datetime objects
    # Return operational cost (BAU and Optimised) and part load for each interval
    def SimpleOptiControl(self, tech_id = None, time_start = None, time_stop = None, table_string=None, mod=None, uncertainty=None, CHPQI_IO = None):
        if time_start is not None or time_stop is not None or table_string is not None: 
                old_time_start = self.time_start; old_time_stop = self.time_stop;  old_price_table = self.price_table
                self.putUtility(time_start =time_start, time_stop = time_stop, table_string=table_string)           
        if tech_id is not None:
            self.putTech(tech_id)
        if hasattr(self, 'tech') == False:
            raise Exception("tech not initialized")         

        ##########  MAIN CODE #######    
        ## get all data        
        CHPQI_threshold = self.CHPQI_threshold          
        
        [tech_data, utility_data] = self.calculate_data(mod = mod, uncertainty = uncertainty)
        [Boiler_eff, a_fuel, b_fuel, a_el, b_el, a_th,  b_th, psi_min, parasitic_load, mant_costs]  = tech_data  
        [el_price, el_price_exp, gas_price, th_demand, el_demand, gas_price_CHP] = utility_data
        
        ## calculate optimum part load    
        psi_el = (el_demand - b_el)/a_el
        psi_th = (th_demand - b_th)/a_th  
        
        PL = np.zeros(shape = (len(th_demand),5))
        PL[:,1] = psi_min
        PL[:,4] = 1
        col2 = np.minimum(psi_el, psi_th)
        col3 = np.maximum(psi_el, psi_th)
        col2[col2<psi_min] = psi_min
        col2[col2>1] = 1
        col3[col3>1] = 1
        col3[col3<psi_min] = psi_min
        PL[:,2] = col2
        PL[:,3] = col3
        
        mask000 = PL > 0.01
        mask011 = (a_el*PL+b_el)*mask000>el_demand.reshape(len(el_demand),1)
        mask012 = (a_th*PL+b_th)*mask000 > th_demand.reshape(len(th_demand),1)   
        a=(a_fuel*PL+b_fuel)*mask000*gas_price_CHP.reshape(len(gas_price_CHP),1) 
        op_cost_HH = a+(el_demand.reshape(len(el_demand),1) -(a_el*PL+b_el)*mask000)*(1-mask011)*el_price.reshape(len(el_price),1) + (th_demand.reshape(len(th_demand),1) - (a_th*PL+b_th)*mask000)*(1-mask012)/Boiler_eff*gas_price.reshape(len(gas_price),1) - ((a_el*PL+b_el)*mask000-el_demand.reshape(len(el_demand),1) )*(mask011)*el_price_exp.reshape(len(el_price_exp),1)
        part_load = PL[np.arange(PL.shape[0]),np.argmin(op_cost_HH,axis = 1)]
        
        ## calcualte outputs
        [op_cost_HH_pound, BAU_op_cost_HH_pound]  = self.calculate_op_cost(part_load, mod = mod, uncertainty = uncertainty)
       
        CHPQI  = self.calculate_CHPQI(part_load, mod = mod, uncertainty = uncertainty)        
        
        mask000 = part_load > 0.01
        mask011 = (a_el*part_load+b_el)*mask000>el_demand
        mask012 = (a_th*part_load+b_th)*mask000 > th_demand   
        #find CHPQI
        el_utilisation = (a_el*part_load+b_el)*mask000
        el_tot_utilisation = np.sum(el_utilisation)
        fuel_utilisation = (a_fuel*part_load+b_fuel)*mask000 
        fuel_tot_utilisation = np.sum(fuel_utilisation)
        th_utilisation = np.minimum((a_th*part_load+b_th)*mask000, th_demand)
        th_tot_utilisation = np.sum(th_utilisation)
        el_efficiency_tot = el_tot_utilisation/fuel_tot_utilisation 
        th_efficiency_tot =th_tot_utilisation/fuel_tot_utilisation
        CHPQI = el_efficiency_tot*238+th_efficiency_tot*120
        op_cost_HH= op_cost_HH_pound*100
        BAU_op_cost_HH= BAU_op_cost_HH_pound*100 
       
        #enforcing CHPQI in case         
        if CHPQI_IO is not None:
            if CHPQI_IO == 1:
                if CHPQI >= CHPQI_threshold:
                    pass #do nothing, CHPQI is good already
                else:
                    niter = 0
                    count = 0
                    while CHPQI < 105 and niter < 300:   
                            D_psi = np.zeros(len(part_load))
                            D_psi_2 = np.zeros(len(part_load))
                            IO_change = np.zeros(len(part_load))
                            der_CHPQI = np.zeros(len(part_load))
                            con1 = part_load == psi_min
                            con2 = part_load > psi_th
                            con3 = part_load > psi_min
                            D_psi[con1 & con2] = psi_min
                            IO_change[con1 & con2] = 1
                            D_psi_2[con1 & con2] = 1
                            D_psi[con3 & con2] = part_load[con3 & con2] -  np.maximum(psi_th[con3 & con2], psi_min)
                            IO_change[con3 & con2] = 1
                            
                            new_part_load = part_load - D_psi
                            new_mask000 = new_part_load > 0.01
                            new_el_utilisation = (a_el*new_part_load+b_el)*new_mask000
                            new_fuel_utilisation = (a_fuel*new_part_load+b_fuel)*new_mask000 
                            new_th_utilisation = np.minimum((a_th*new_part_load+b_th)*new_mask000, th_demand)
                            D_el_utilisation = el_utilisation - new_el_utilisation
                            D_fuel_utilisation = fuel_utilisation - new_fuel_utilisation
                            D_th_utilisation = th_utilisation - new_th_utilisation
                            
                            new_mask011 = (a_el*new_part_load+b_el)*new_mask000>el_demand
                            new_mask012 = (a_th*new_part_load+b_th)*new_mask000 > th_demand   
                            new_op_cost_HH = (a_fuel*new_part_load+b_fuel)*new_mask000*gas_price_CHP +(el_demand-(a_el*new_part_load+b_el)*new_mask000)*(1-new_mask011)*el_price + (th_demand - (a_th*new_part_load+b_th)*new_mask000)*(1-new_mask012)/Boiler_eff*gas_price - ((a_el*new_part_load+b_el)*new_mask000-el_demand)*(new_mask011)*el_price_exp
                           
                            D_CHPQI_el = np.divide(el_tot_utilisation,fuel_tot_utilisation) - np.divide((el_tot_utilisation - D_el_utilisation),(fuel_tot_utilisation-D_fuel_utilisation)) 
                            D_CHPQI_th = np.divide(th_tot_utilisation,fuel_tot_utilisation) - np.divide((th_tot_utilisation - D_th_utilisation),(fuel_tot_utilisation-D_fuel_utilisation)) 
                            D_CHPQI = D_CHPQI_el*238 + D_CHPQI_th*120
                            D_op_cost = op_cost_HH -  new_op_cost_HH
                            D_op_cost[IO_change < 1]  = -1000
                            der_CHPQI= np.divide(D_CHPQI,D_op_cost)
                            der_CHPQI[IO_change < 1] = 0 ##strange situation were to increase the CHPQI at one point it needs to decrease first and the increase
                            index_CHPQI =np.argsort(der_CHPQI)
                            index_CHPQI = np.flip(index_CHPQI, 0)                     
                            index = index_CHPQI[0:50]
                            part_load[index]=new_part_load[index]
                            #op_cost_HH[index] = new_op_cost_HH[index]

                            mask000 = part_load > 0.01
                            mask011 = (a_el*part_load+b_el)*mask000>el_demand
                            mask012 = (a_th*part_load+b_th)*mask000 > th_demand   
                            el_utilisation = (a_el*part_load+b_el)*mask000
                            el_tot_utilisation = np.sum(el_utilisation)
                            fuel_utilisation = (a_fuel*part_load+b_fuel)*mask000 
                            fuel_tot_utilisation = np.sum(fuel_utilisation)
                            th_utilisation = np.minimum((a_th*part_load+b_th)*mask000, th_demand)
                            th_tot_utilisation = np.sum(th_utilisation)
                            el_efficiency_tot = el_tot_utilisation/fuel_tot_utilisation 
                            th_efficiency_tot =th_tot_utilisation/fuel_tot_utilisation
                            new_CHPQI = el_efficiency_tot*238+th_efficiency_tot*120
                            if (new_CHPQI-CHPQI)/CHPQI < 0.0001:
                                niter = 100000
                                new_CHPQI = - 1000
                            CHPQI = new_CHPQI
                            op_cost_HH = (a_fuel*part_load+b_fuel)*mask000*gas_price_CHP +(el_demand-(a_el*part_load+b_el)*mask000)*(1-mask011)*el_price + (th_demand - (a_th*part_load+b_th)*mask000)*(1-mask012)/Boiler_eff*gas_price - ((a_el*part_load+b_el)*mask000-el_demand)*(mask011)*el_price_exp
                            BAU_op_cost_HH = el_demand*el_price + th_demand/Boiler_eff*gas_price 
                            op_cost_HH_pound = op_cost_HH /100
                            BAU_op_cost_HH_pound = BAU_op_cost_HH/100 
#                            if count == 10:
#                                count = 0
#                                print('iteration:',niter)
#                                print('CHPQI:',CHPQI)
#                                print('operational_cost:', sum(op_cost_HH)/100)
                            niter = niter + 1
                            count = count + 1
        if time_start is not None or time_stop is not None or table_string is not None: 
                self.putUtility(time_start =old_time_start, time_stop = old_time_stop, table_string=old_price_table) 
            
        return(BAU_op_cost_HH_pound, op_cost_HH_pound, part_load, CHPQI) 

        
        #find the part load of tech and cost considering a CHP which follows the load
    def LoadFollowControl(self, tech_id= None, time_start = None, time_stop = None, table_string=None, mod = [1,1,1,1], uncertainty = [0,0,0]):
        if time_start is not None or time_stop is not None or table_string is not None: 
                old_time_start = self.time_start; old_time_stop = self.time_stop;  old_price_table = self.price_table
                self.putUtility(time_start =time_start, time_stop = time_stop, table_string=table_string)          
        if tech_id is not None:
            self.putTech(tech_id)
        if hasattr(self, 'tech') == False:
            raise Exception("tech not initialized")  
            
        [tech_data, utility_data] = self.calculate_data(mod = mod, uncertainty = uncertainty)
        [Boiler_eff, a_fuel, b_fuel, a_el, b_el, a_th,  b_th, psi_min, parasitic_load, mant_costs]  = tech_data  
        [el_price, el_price_exp, gas_price, th_demand, el_demand, gas_price_CHP] = utility_data
        
        ## calculate optimum part load    
        psi_el = (el_demand - b_el)/a_el
        psi_th = (th_demand - b_th)/a_th  
        
        part_load = np.zeros((len(psi_th))) 
        for count in range(len(part_load)):
            if psi_el[count] < psi_min:
                    part_load[count] = 0  
            elif psi_el[count] > 1:
                    part_load[count] = 1
            else:
                    part_load[count] = psi_el[count]

        ## calcualte outputs
        [op_cost_HH_pound, BAU_op_cost_HH_pound]  = self.calculate_op_cost(part_load, mod = mod, uncertainty = uncertainty)
        CHPQI  = self.calculate_CHPQI(part_load, mod = mod, uncertainty = uncertainty) 

        if time_start is not None or time_stop is not None or table_string is not None: 
                self.putUtility(time_start =old_time_start, time_stop = old_time_stop, table_string=old_price_table)  
            
        return(BAU_op_cost_HH_pound, op_cost_HH_pound, part_load, CHPQI)
    
    
    
    
        #find the part load of tech and cost considering a CHP which follows the load and is turned on only during trading hours
    def LoadFollowControlOnOff(self, tech_id= None, time_start = None, time_stop = None, table_string=None, mod=None, uncertainty=None):
        if time_start is not None or time_stop is not None or table_string is not None: 
                old_time_start = self.time_start; old_time_stop = self.time_stop;  old_price_table = self.price_table
                self.putUtility(time_start =time_start, time_stop = time_stop, table_string=table_string)       
        if tech_id is not None:
            self.putTech(tech_id)
        if hasattr(self, 'tech') == False:
            raise Exception("tech not initialized") 
            
        ### MAIN CODE    
        timestamp = self.store.timestamp      
        
        [tech_data, utility_data] = self.calculate_data(mod = mod, uncertainty = uncertainty)
        [Boiler_eff, a_fuel, b_fuel, a_el, b_el, a_th,  b_th, psi_min, parasitic_load, mant_costs]  = tech_data  
        [el_price, el_price_exp, gas_price, th_demand, el_demand, gas_price_CHP] = utility_data
        
        ## calculate optimum part load    
        psi_el = (el_demand - b_el)/a_el
        psi_th = (th_demand - b_th)/a_th  

        part_load = np.zeros((len(psi_th)))         
        Cal =calendar.Calendar(calendar.SUNDAY).yeardays2calendar(datetime.datetime.fromtimestamp(timestamp[0]*60*30).year,1)
        NewCal = [[] for x in range(12)] 
        count_month = 0                       
        for month in Cal:
            for week in month[0]:              
                for day in week:
                    if day[0] is not 0:
                        NewCal[count_month].append(day)
            count_month = count_month + 1
                
        for count in range(len(part_load)):            
            HH = 2*datetime.datetime.fromtimestamp(timestamp[count]*60*30).hour + datetime.datetime.fromtimestamp(timestamp[count]*60*30).minute/30
            Month = datetime.datetime.fromtimestamp(timestamp[count]*60*30).month -1
            Day = datetime.datetime.fromtimestamp(timestamp[count]*60*30).day
            WeekDay = NewCal[Month][Day-1][1]
            if WeekDay == 5:
                HH_open = self.store.HH_Sat_open
                HH_close = self.store.HH_Sat_close
            elif WeekDay == 6:  
                HH_open = self.store.HH_Sun_open
                HH_close = self.store.HH_Sun_close
            else:
                HH_open = self.store.HH_WD_open
                HH_close = self.store.HH_WD_close                
            
            if HH > HH_open and HH < HH_close:                
                if psi_el[count] < psi_min:
                        part_load[count] = 0  
                elif psi_el[count] > 1:
                        part_load[count] = 1
                else:
                        part_load[count] = psi_el[count]
            else:
                part_load[count] = 0        
                
        ## calcualte outputs
        
        [op_cost_HH_pound, BAU_op_cost_HH_pound ] = self.calculate_op_cost(part_load, mod = mod, uncertainty = uncertainty)
        CHPQI  = self.calculate_CHPQI(part_load, mod = mod, uncertainty = uncertainty) 
        
        if time_start is not None or time_stop is not None or table_string is not None: 
                self.putUtility(time_start =old_time_start, time_stop = old_time_stop, table_string=old_price_table)         

        return(BAU_op_cost_HH_pound, op_cost_HH_pound, part_load, CHPQI)
        
    
    ## get the operating cost from the part load
    def calculate_op_cost(self, part_load, tech_id = None, time_start = None, time_stop = None, table_string=None, mod=None, uncertainty=None):               
        if time_start is not None or time_stop is not None or table_string is not None: 
                old_time_start = self.time_start; old_time_stop = self.time_stop;  old_price_table = self.price_table
                self.putUtility(time_start =time_start, time_stop = time_stop, table_string=table_string) 
        if len(part_load) != len(self.store.p_ele):
            raise Exception("part load length do not match size of other vector")            
        if tech_id is not None:
            self.putTech(tech_id)
        if hasattr(self, 'tech') == False:
            raise Exception("tech not initialized") 

        [tech_data, utility_data] = self.calculate_data(mod = mod, uncertainty = uncertainty)
        [Boiler_eff, a_fuel, b_fuel, a_el, b_el, a_th,  b_th, psi_min, parasitic_load, mant_costs]  = tech_data  
        [el_price, el_price_exp, gas_price, th_demand, el_demand, gas_price_CHP] = utility_data

        check_psi = np.array(part_load)
        check_psi[check_psi == 0] = 1
        if min(check_psi)< (psi_min):
            raise Exception("part load less than minimum part load")      

        mask000 = part_load > 0.01
        mask011 = (a_el*part_load+b_el)*mask000 > el_demand
        mask012 = (a_th*part_load+b_th)*mask000 > th_demand   
        op_cost_HH = (a_fuel*part_load+b_fuel)*mask000*gas_price_CHP +(el_demand-(a_el*part_load+b_el)*mask000)*(1-mask011)*el_price + (th_demand - (a_th*part_load+b_th)*mask000)*(1-mask012)/Boiler_eff*gas_price - ((a_el*part_load+b_el)*mask000-el_demand)*(mask011)*el_price_exp
        op_cost_HH_pound= op_cost_HH /100
        BAU_op_cost_HH = el_demand*el_price + th_demand/Boiler_eff*gas_price 
        BAU_op_cost_HH_pound = BAU_op_cost_HH/100
        
        if time_start is not None or time_stop is not None or table_string is not None: 
                self.putUtility(time_start =old_time_start, time_stop = old_time_stop, table_string=old_price_table)
                
        return(op_cost_HH_pound, BAU_op_cost_HH_pound)

    
    
    ## get the CHPQI from the part load
    def calculate_CHPQI(self, part_load, tech_id = None, time_start = None, time_stop = None, mod=None, uncertainty=None):  
        if time_start is not None or time_stop is not None: 
                old_time_start = self.time_start; old_time_stop = self.time_stop
                self.putUtility(time_start =time_start, time_stop = time_stop)                           
        if len(part_load) != len(self.store.p_ele):
            raise Exception("part load length do not match size of other vector")        
        if tech_id is not None:
            self.putTech(tech_id)
        if hasattr(self, 'tech') == False:
            raise Exception("tech not initialized") 
            
        [tech_data, utility_data] = self.calculate_data(mod = mod, uncertainty = uncertainty)
        [Boiler_eff, a_fuel, b_fuel, a_el, b_el, a_th,  b_th, psi_min, parasitic_load, mant_costs]  = tech_data  
        [el_price, el_price_exp, gas_price, th_demand, el_demand, gas_price_CHP] = utility_data
        
        check_psi = np.array(part_load)
        check_psi[check_psi == 0] = 1
        if min(check_psi)< (psi_min):
            raise Exception("part load less than minimum part load")   
            
        mask000 = part_load > 0.01
        el_utilisation = (a_el*part_load+b_el)*mask000
        el_tot_utilisation = np.sum(el_utilisation)
        fuel_utilisation = (a_fuel*part_load+b_fuel)*mask000 
        fuel_tot_utilisation = np.sum(fuel_utilisation)
        th_utilisation = np.minimum((a_th*part_load+b_th)*mask000, th_demand)
        th_tot_utilisation = np.sum(th_utilisation)
        if sum(part_load) == 0:
            CHPQI = np.nan
        else: 
            el_efficiency_tot = el_tot_utilisation/fuel_tot_utilisation 
            th_efficiency_tot =th_tot_utilisation/fuel_tot_utilisation
            CHPQI = el_efficiency_tot*238+th_efficiency_tot*120
            
        if time_start is not None or time_stop is not None: 
                self.putUtility(time_start =old_time_start, time_stop = old_time_stop)    
            
        return(CHPQI)
    
    
        #calculate  data to be used in the models
    def calculate_data(self, mod = None, uncertainty = None):
        if hasattr(self, 'tech') == False:
            raise Exception("tech not initialized")              
        if mod is None:
            mod = [1,1,1,1]
        if uncertainty is None:
            uncertainty = [0,0,0]
        
        #Factor to convert consumption given in LHV to HHV
        K_fuel = 1 #39.8/36 (the value is already implemented in the technology used)
        
        Boiler_eff = self.boiler_eff           
        a_fuel = self.tech.a_fuel*mod[2]*K_fuel
        b_fuel = self.tech.b_fuel*mod[2]*K_fuel
        a_el = self.tech.a_el
        b_el = self.tech.b_el
        a_th =  self.tech.a_th
        b_th = self.tech.b_th
        psi_min = self.tech.psi_min    
        parasitic_load =  self.tech.parasitic_load 
        mant_costs = self.tech.mant_costs   
        ## convert CHP data from kW to kWh (for every HH) by dividing by 2 ##
        parasitic_load = parasitic_load/2; a_fuel = a_fuel/2;a_el = a_el/2;a_th = a_th/2;b_fuel = b_fuel/2;b_el = b_el/2;b_th = b_th /2
        ## need also to subtract the parasitic load form the CHP electricity production ##
        b_el = b_el-parasitic_load
        tech_data = [Boiler_eff, a_fuel, b_fuel, a_el, b_el, a_th,  b_th, psi_min, parasitic_load, mant_costs]
        el_efficiency = (a_el+b_el)/(a_fuel+b_fuel)
        
        el_price = self.store.p_ele*mod[0]
        el_price_exp = self.store.p_ele_exp
        gas_price = self.store.p_gas*mod[1]
        gas_price_CHP = (self.store.p_gas + mant_costs*el_efficiency*100)*mod[1]
        th_demand = self.store.d_gas*Boiler_eff                  ##  kWth HH  ##
        el_demand = self.store.d_ele                             ##  kWel HH  ##
        utility_data = [el_price, el_price_exp, gas_price, th_demand, el_demand, gas_price_CHP]
        return(tech_data, utility_data)
    
    def calculate_financials(self, discount_rate, tech_lifetime, year_BAU_cost, year_op_cost, Total_capex):
            year_savings = year_BAU_cost - year_op_cost
            payback = Total_capex/year_savings           
            ann_capex = -np.pmt(discount_rate, tech_lifetime, Total_capex)            
            year_cost = year_op_cost  + ann_capex
            NPV5_op_cost  = -np.npv(discount_rate, np.array([year_cost]*5))
            NPV5_BAU_cost = -np.npv(discount_rate, np.array([year_BAU_cost]*5)) 
            NPV5savings = NPV5_op_cost - NPV5_BAU_cost
            ROI = year_savings/Total_capex
            Const = (1-(1+discount_rate)**(-tech_lifetime))/discount_rate            
            Cum_disc_cash_flow = -Total_capex + Const*year_savings 
            return(year_savings, payback, NPV5savings, ROI, Cum_disc_cash_flow)
    

    ## initialise a technology 
    def putTech(self, tech_id): 
        self.tech = tc.tech(tech_id)

    def putUtility(self, time_start = None, time_stop = None, table_string=None): 
        if time_start is not None or time_stop is not None or table_string is not None:
            if time_start is not None:
                self.time_start = time_start#int((time_start-datetime.datetime(1970,1,1)).total_seconds()/60/30)  # changed for simplicity
            if time_stop is not None:
                self.time_stop = time_stop#int((time_stop-datetime.datetime(1970,1,1)).total_seconds()/60/30)                
            if table_string is not None:
                self.price_table = table_string       
            self.store.getSimplePrice(self.time_start, self.time_stop, self.price_table)
            self.store.getSimpleDemand(self.time_start, self.time_stop)   
        else:
             raise Exception("no inputs.. doing nothing")

       
    def dat_file(self, file_name, E_demand, Q_demand, e_imp, e_exp, q, tech):
    
        # Writes .dat file to be read by abstract model. Clunky, but oh well
        with open(file_name,'w') as f:
            f.write("###### \n# Author: Sebastian Gonzato \n##### \n\n# Data file \n")
              
            nt = len(E_demand) # number of time periods
            f.write("\nparam : n := %d;\n" %(nt))
            
            # Some values may be less than 0 because of bad data! Write 0 instead
            def write_var(f,i,var):
                if var >= 0:
                    f.write("%d %f \n" %(i+1,var))
                else:
                    f.write("%d %f \n" %(i+1,0))
                    
            f.write("\nparam : E_demand := \n")
            for i in range(nt):
                write_var(f,i,E_demand[i])
#                write_var(f,i,150)
                    
            f.write(";\n\nparam : Q_demand := \n")
            for i in range(nt):
                write_var(f,i,Q_demand[i])
#                write_var(f,i,90)
    			
            f.write(";\n\nparam : e_imp := \n")
            for i in range(nt):
                write_var(f,i,e_imp[i]) 
#                write_var(f,i,8) 
    		
            f.write(";\n\nparam : e_exp := \n")
            for i in range(nt):
                write_var(f,i,e_exp[i]) 
#                write_var(f,i,2) 
    			
            f.write(";\n\nparam : q := \n")
            for i in range(nt):
                write_var(f,i,q[i])
#                write_var(f,i,2.4)

            # Parameters
            # TODO: Change this to allow for addition of other technologies
            f.write(";\n\nparam : f_min f_max a_el b_el a_th b_th a_fuel b_fuel := \n")
            f.write("BOIL 0 1 0 0 870 0 1000 0 \n")
            f.write("CHP %f %f %f %f %f %f %f %f \n" %(tech[6],1,tech[2],tech[3],tech[4],tech[5],tech[0],tech[1]))
            
            # Write end of file
            f.write(";\n\nend;")

    # find the part load of tech and cost considering a CHP which follows the load
    def SebastianControl(self, tech_id, time_start = None, time_stop = None, table_string=None, uncertainty = [0,0,0], CHPQA = False,
                         mipgap = 1):
        start = time.clock() # Starting time
        if time_start is not None or time_stop is not None or table_string is not None:
            if time_start is not None:
                self.time_start = int((time_start-datetime.datetime(1970,1,1)).total_seconds()/60/30)
            if time_stop is not None:
                self.time_stop = int((time_stop-datetime.datetime(1970,1,1)).total_seconds()/60/30)                
            if table_string is not None:
                self.price_table = table_string       
            self.store.getSimplePrice(self.time_start, self.time_stop, self.price_table)
            self.store.getSimpleDemand(self.time_start, self.time_stop)            
        
        ## MAIN CODE     
        Boiler_eff = self.boiler_eff
        conn = sqlite3.connect(database_path)
        cur = conn.cursor()
        cur.execute('''SELECT * FROM Technologies WHERE id=?''', (tech_id,))
        dummy = cur.fetchall()
        
        
        # Explanation for these variables is given in Pyomo model below
        a_fuel = dummy[0][3]
        b_fuel = dummy[0][4]
        a_el = dummy[0][5]
        b_el = dummy[0][6]
        a_th = dummy[0][7]
        b_th = dummy[0][8]
        psi_min = dummy[0][9]
        parasitic_load = dummy[0][10]
        mant_costs = dummy[0][11]
        
        ni = 2 # Number of intervals for each half hour, divide demands by this number to get the demand in kWh
        
        el_price = self.store.p_ele
        el_price_exp = self.store.p_ele_exp
        gas_price = self.store.p_gas + mant_costs
        th_demand = self.store.d_gas*Boiler_eff/ni
        el_demand = self.store.d_ele/ni
        
        # Calculate business as usual (BAU) cost in pounds (hence division by 100)
        BAU_op_cost_HH = el_demand*el_price + th_demand/Boiler_eff*gas_price/100
        print("BAU ",time.clock() - start)
        
        # Find CHP part load from other control ting
        [BAU,OP,part_load, CHPQI] = self.SimpleOptiControl(tech_id = tech_id)
        
        ######################################################################
        # PYOMO OPTIMISATION MODEL
        ######################################################################
        
        # Write .dat file to be used by abstract model
        filename = "SebastianControl.dat"
        tech = [a_fuel,b_fuel,a_el,b_el-parasitic_load,a_th,b_th,psi_min]
        self.dat_file(filename, el_demand, th_demand, el_price, el_price_exp, gas_price, tech)
        
        print("Creating .dat file",time.clock() - start)
        
        # Define model class (object?)
        model = AbstractModel()
        
        # Define sets
        model.n = Param(domain=Integers)
        model.P = RangeSet(1,model.n) # Time periods set
        model.U = Set(initialize=["CHP","BOIL"])
        
        # Time period specific parameters
        model.Q_demand = Param(model.P, within=NonNegativeReals) # Heating demand [kWh]
        model.E_demand = Param(model.P, within=NonNegativeReals) # Electricity demand [kWh]
        model.e_imp = Param(model.P, within=NonNegativeReals) # Electricity price [p/kWh]
        model.e_exp = Param(model.P, within=NonNegativeReals) # Electricity price [p/kWh]
        model.q = Param(model.P, within=NonNegativeReals) # Gas price [p/kWh]
        
        # Unit specific parameters
        model.f_max = Param(model.U, within=NonNegativeReals) # Maximum part load of the units
        model.f_min = Param(model.U, within=NonNegativeReals) # Minimum part load of the units
        # The fuel consumption and electricity and heat production is defined as 
        # linear functions (variable == af + by, where f and y are defined below) of
        # the part load, hence the parameters below.
        model.a_fuel = Param(model.U)
        model.b_fuel = Param(model.U)
        model.a_el = Param(model.U)
        model.b_el = Param(model.U)
        model.a_th = Param(model.U)
        model.b_th = Param(model.U)
        
        # Functions to initialise variables
        def f_init(model,u,t):
            if u == 'CHP':
                return part_load[t-1]
            else:
                return 0
        
        def y_init(model,u,t):
            if u == 'CHP' and part_load[t-1] >= psi_min:
                return 1
            elif u == 'BOIL' and part_load[t-1] <= psi_min:
                return 1
            else:
                return 0
        
        # Variables
        model.f = Var(model.U, model.P, within=UnitInterval, initialize=f_init)
#        model.f = Var(model.U, model.P, within=UnitInterval)
        model.y = Var(model.U, model.P, within=Binary, initialize=y_init)
#        model.y = Var(model.U, model.P, within=Binary)
        model.Q_out = Var(model.P, within=NonNegativeReals)
        model.E_out = Var(model.P, within=NonNegativeReals)
        model.E_imp = Var(model.P, within=NonNegativeReals)
        model.E_exp = Var(model.P, within=NonNegativeReals)
        model.Q_ng = Var(model.P, within=NonNegativeReals)
        model.CHP_etaE = Var(within=NonNegativeReals)
        model.CHP_etaQ = Var(within=NonNegativeReals)
        
        ##############################################################################
        # TECHNOLOGY CONSTRAINTS
        ##############################################################################
        
        # The first two constraints ensure that the units can't operate below a minimum
        # value
        def f_min_part_load_rule(model,u,t):
            return model.f[u,t] >= model.f_min[u] - (1-model.y[u,t])*model.f_max[u]
        model.f_min_part_load_rule = Constraint(model.U*model.P, rule=f_min_part_load_rule)
        # Units do not cannot produce electricity if they're output is below f_min. Look at this again
        
        def f_off_rule(model,u,t):
            return model.f[u,t] <= model.y[u,t]*model.f_max[u]
        model.f_off_rule = Constraint(model.U*model.P, rule = f_off_rule)
        # Units do not produe any form of energy if off
        
        def Q_ng_rule (model,t):
            return model.Q_ng[t] == sum (model.f[u,t]*model.a_fuel[u] + model.y[u,t]*model.b_fuel[u] for u in model.U)
        model.Q_ng_rule = Constraint(model.P, rule=Q_ng_rule)
        
        def Q_out_rule (model,t):
            return model.Q_out[t] == sum (model.f[u,t]*model.a_th[u] + model.y[u,t]*model.b_th[u] for u in model.U)
        model.Q_out_rule = Constraint(model.P, rule=Q_out_rule)
        # Thermal output of all the units
        
        def E_out_rule (model,t):
            return model.E_out[t] == sum (model.f[u,t]*model.a_el[u] + model.y[u,t]*model.b_el[u] for u in model.U)
        model.E_out_rule = Constraint(model.P, rule=E_out_rule)
        # Electrical output of all the units
        
        if CHPQA == True:
            def Power_efficiency(model):
                return sum(model.f["CHP",t]*model.a_el["CHP"] + model.y["CHP",t]*model.b_el["CHP"] for t in model.P)
                    
            def Heat_efficiency(model):
                return sum(model.f["CHP",t]*model.a_th["CHP"] + model.y["CHP",t]*model.b_th["CHP"] for t in model.P)
            
            def Fuel_efficiency(model):
                return sum(model.f["CHP",t]*model.a_fuel["CHP"] + model.y["CHP",t]*model.b_fuel["CHP"] for t in model.P)
            
            def CHPQA_rule(model):
                return Power_efficiency(model)*238 + Heat_efficiency(model)*120 >= 105*Fuel_efficiency(model)
        
            model.CHPQA_rule = Constraint(rule=CHPQA_rule)
        # CHPQA constraints if required
            
        ##############################################################################
        # BALANCES
        ##############################################################################
        
        def Elec_balance_rule (model,t):
            return model.E_out[t] + model.E_imp[t] == model.E_demand[t] + model.E_exp[t]
        model.Elec_balance_rule = Constraint(model.P, rule=Elec_balance_rule)
        # Electricity balance
        
        def Heat_balance_rule (model,t):
            return model.Q_out[t] >= model.Q_demand[t]
        model.Heat_balance_rule = Constraint(model.P, rule=Heat_balance_rule)
        
        ##############################################################################
        # OBJECTIVE FUNCTION
        ##############################################################################
        
        def obj_rule(model):
            return sum (model.Q_ng[t]*model.q[t] + model.E_imp[t]*model.e_imp[t] - model.E_exp[t]*model.e_exp[t] for t in model.P)
        model.obj = Objective(rule=obj_rule)
        
        ##############################################################################
        # SOLVE
        ##############################################################################
        
#        instance = model.create_instance("simple_CHP.dat")
        instance = model.create_instance(filename)
        
        print("Creating model",time.clock() - start)
        
        solver = "glpk"
        opt = SolverFactory(solver) # Choose solver and input settings
#        opt.options['tmlim'] = 5
        
        if opt is None:
            ValueError, "Problem constructing solver " + solver
        
        opt.options['mipgap'] = mipgap # tolerance for integerisation
#        opt.options['mipgap'] = 1 # tolerance for integerisation
#        opt.options['nomip'] # Don't solve the integerisation if you feel like it
        solver_manager = SolverManagerFactory("serial")
        results = solver_manager.solve(instance, opt=opt, tee=True, timelimit=None) # Solve model, tee = True should display solver output (it doesn't)
        
        # Delete .dat file, don't need it anymore
        cmd = "del " + filename
        os.system(cmd)
        
        print("Solving model",time.clock() - start)
        
        # Save f_out and cost to array 
        f_out = np.zeros(instance.n())
        for i in instance.P:
            f_out[i-1] = instance.f["CHP",i].value
#            f_out = f_out.append(instance.f["CHP",i].value)


        return(BAU_op_cost_HH,instance.obj.expr(),f_out) #,objective_value)#,f

    ## get the carbon emissions from the part load
    def calculate_carbon(self, part_load, tech_id=None, time_start=None, time_stop=None, table_string=None, mod=None,
                          uncertainty=None):
        if time_start is not None or time_stop is not None or table_string is not None:
            old_time_start = self.time_start;
            old_time_stop = self.time_stop;
            old_price_table = self.price_table
            self.putUtility(time_start=time_start, time_stop=time_stop, table_string=table_string)
        if len(part_load) != len(self.store.p_ele):
            raise Exception("part load length do not match size of other vector")
        if tech_id is not None:
            self.putTech(tech_id)
        if hasattr(self, 'tech') == False:
            raise Exception("tech not initialized")

        [tech_data, utility_data] = self.calculate_data(mod=mod, uncertainty=uncertainty)
        [Boiler_eff, a_fuel, b_fuel, a_el, b_el, a_th, b_th, psi_min, parasitic_load, mant_costs] = tech_data
        [el_price, el_price_exp, gas_price, th_demand, el_demand, gas_price_CHP] = utility_data

        check_psi = np.array(part_load)
        check_psi[check_psi == 0] = 1
        if min(check_psi) < (psi_min):
            raise Exception("part load less than minimum part load")
        biometh_CF=0.0328
        gas_CF = 0.1840
        ele_CF = 0.370845
        mask000 = part_load > 0.01
        mask011 = (a_el * part_load + b_el) * mask000 > el_demand
        mask012 = (a_th * part_load + b_th) * mask000 > th_demand
        carbon_CHP = ((a_fuel * part_load + b_fuel) * mask000 * biometh_CF + (el_demand - (
            a_el * part_load + b_el) * mask000) * (1 - mask011) * ele_CF + (th_demand - (
            a_th * part_load + b_th) * mask000) * (
                      1 - mask012) / Boiler_eff * biometh_CF) / 1000  # - ((a_el * part_load + b_el) * mask000 - el_demand) * (mask011) * ele_CF
        BAU_carbon = (el_demand * ele_CF + th_demand / Boiler_eff * gas_CF) / 1000  # (tCO2)
        carbon_savings = (BAU_carbon - carbon_CHP)
        biomethane_usage=(a_fuel * part_load + b_fuel) * mask000 +(th_demand - (a_th * part_load + b_th) * mask000)
        if time_start is not None or time_stop is not None or table_string is not None:
            self.putUtility(time_start=old_time_start, time_stop=old_time_stop, table_string=old_price_table)

        return (np.sum(BAU_carbon), np.sum(carbon_CHP), np.sum(carbon_savings),np.sum(biomethane_usage))
