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
import Common.classStore as st # Module is in seperate folder, hence the elaboration
import Common.classTech as tc
import Connection.classClusterConnection as cc
#from gams import GamsWorkspace


#database_path = "C:\\Users\\GOBS\\Dropbox\\Uni\Other\\UROP - Salvador\\Niccolo_project\\Code\\Sainsburys.sqlite" # Path to database file
database_path = ".\\Sainsburys.sqlite" # Path to database file
# workaround for the stupid python-GAMS API
GAMS_dir_path = os.path.dirname(os.path.realpath(__file__))
GAMS_dir_path = GAMS_dir_path[0:len(GAMS_dir_path)-8]
GAMS_path = GAMS_dir_path + ".\\GAMS"


class CHPproblem:
    
    #set a bounch of properties including a store object for the problem with demand and price data
    def __init__(self, store_id):            
        self.store = st.store(store_id)
        self.price_table = 'Utility_Prices_SSL'
        default_initial_time = datetime.datetime(2016,1,1)
        default_final_time = datetime.datetime(2017,1,1)
        self.time_start= int((default_initial_time-datetime.datetime(1970,1,1)).total_seconds()/60/30)
        self.time_stop= int((default_final_time-datetime.datetime(1970,1,1)).total_seconds()/60/30)
        self.store.getSimplePrice(self.time_start, self.time_stop, self.price_table)
        self.store.getSimpleDemand(self.time_start, self.time_stop)
        self.boiler_eff = 0.87
        self.hidden_costs = 200000 + self.store.area*2
        #self.financial_lifetime = 15
        self.discount_rate = 0.09
        self.CHPQI_threshold = 105
     
    ############################################
    ### HERE are the optimisation algorithms ###    
    ############################################
    
    # Optimisation function: finds the best tech by iterating over each technology then determining which is best.    
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
                [BAU_op_cost_HH_pound, op_cost_HH_pound, opti_CHPQI , CHPQI] = self.MILPOptiControl(uncertainty = uncertainty, mod = mod)
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
            [year_savings, payback, NPV5savings, ROI, Cum_disc_cash_flow, IRR] = self.calculate_financials(discount_rate, tech_lifetime, year_BAU_cost, year_op_cost, Total_capex)
            objective = Cum_disc_cash_flow
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
                opti_Cum_disc_cash_flow = Cum_disc_cash_flow
                opti_capex = Total_capex
                opti_IRR = IRR
                operation_data  = self.calculate_CHPQI(opti_part_load, mod = mod, uncertainty = uncertainty) 


        #wrap output
        basic = [opti_tech, opti_tech_name,  opti_part_load, opti_CHPQI]
        finance = [opti_year_savings, opti_payback, opti_NPV5savings, opti_ROI, opti_Cum_disc_cash_flow, opti_capex, opti_IRR]
        #restore previous values        
        if time_start is not None or time_stop is not None or table_string is not None: 
                self.putUtility(time_start =old_time_start, time_stop = old_time_stop, table_string=old_price_table)
        
        return([basic, finance, operation_data])
    

    # Optimisation function to calculate optimal solution when CHPQI is included:
    # finds the best tech by iterating over each technology and CHPQI option then determining which is best.    
    def CHPQIOpti5NPV(self, tech_range = None, time_start = None, time_stop = None, uncertainty = None, mod = None):
        if time_start is not None or time_stop is not None: 
                old_time_start = self.time_start; old_time_stop = self.time_stop
                self.putUtility(time_start =time_start, time_stop = time_stop)          
    
        if tech_range is not None:
            array_tech = tech_range
        else:
            array_tech = range(1,20)    
                    
        hidden_costs = self.hidden_costs
        discount_rate =  self.discount_rate                 
        
        optimal_objective = -1000000
        opti_tech = -1
        opti_tech_name = 'None'
        ECA_value = 0.26
        table_standard = "Utility_Prices_SSL"
        table_CHPQI = "Utility_Prices_SSL_NoCCL"
        Table_to_restore = self.price_table   
        
        for id_tech_index in array_tech:
            for CHPQI_IO in range(2):
                tech_id = id_tech_index
                self.putTech(tech_id)
                tech_name = self.tech.tech_name
                tech_price = self.tech.tech_price*(1-ECA_value*CHPQI_IO)     
                tech_lifetime = self.tech.lifetime            
                if CHPQI_IO == 1:
                    self.putUtility(table_string=table_CHPQI)                               
                    [BAU_op_cost_HH_pound, op_cost_HH_pound, part_load, CHPQI] = self.SimpleOptiControl(uncertainty = uncertainty, mod = mod, CHPQI_IO = 1)  
                    if CHPQI < self.CHPQI_threshold:  ##then discard results
                        op_cost_HH_pound = op_cost_HH_pound*100
                else:
                    self.putUtility(table_string=table_standard) 
                    [BAU_op_cost_HH_pound, op_cost_HH_pound, part_load, CHPQI] = self.SimpleOptiControl(uncertainty = uncertainty, mod = mod)
                    
                # Calculate finantial
                numb_years = (self.time_stop-self.time_start)/2/24/365
                year_op_cost = sum(op_cost_HH_pound)/numb_years
                year_BAU_cost = sum(BAU_op_cost_HH_pound)/numb_years
                Total_capex = tech_price + hidden_costs            
                [year_savings, payback, NPV5savings, ROI, Cum_disc_cash_flow, IRR] = self.calculate_financials(discount_rate, tech_lifetime, year_BAU_cost, year_op_cost, Total_capex)
                
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
                    opti_Cum_disc_cash_flow = Cum_disc_cash_flow
                    opti_capex = Total_capex 
                    opti_IRR = IRR
                    operation_data  = self.calculate_CHPQI(opti_part_load, mod = mod, uncertainty = uncertainty) 
        
        #wrap output
        basic = [opti_tech, opti_tech_name,  opti_part_load, opti_CHPQI]
        finance = [opti_year_savings, opti_payback, opti_NPV5savings, opti_ROI, opti_Cum_disc_cash_flow, opti_capex, opti_IRR]   
        #restore previous values 
        self.putUtility(table_string=Table_to_restore)   
        if time_start is not None or time_stop is not None: 
                self.putUtility(time_start =old_time_start, time_stop = old_time_stop)        
        
        return([basic, finance, operation_data])


    #################################################################
    ### HERE are the simulation algorithm for the units operation ###    
    #################################################################

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
        op_cost_HH = (a_fuel*PL+b_fuel)*mask000*gas_price_CHP.reshape(len(gas_price_CHP),1) +(el_demand.reshape(len(el_demand),1) -(a_el*PL+b_el)*mask000)*(1-mask011)*el_price.reshape(len(el_price),1) + (th_demand.reshape(len(th_demand),1) - (a_th*PL+b_th)*mask000)*(1-mask012)/Boiler_eff*gas_price.reshape(len(gas_price),1) - ((a_el*PL+b_el)*mask000-el_demand.reshape(len(el_demand),1) )*(mask011)*el_price_exp.reshape(len(el_price_exp),1)
        part_load = PL[np.arange(PL.shape[0]),np.argmin(op_cost_HH,axis = 1)]
        
        ## calcualte outputs
        [op_cost_HH_pound, BAU_op_cost_HH_pound]  = self.calculate_op_cost(part_load, mod = mod, uncertainty = uncertainty)
       
        
        # CHPQI enforcing algorithm. could be simplified using the function below.
        #CHPQI  = self.calculate_CHPQI(part_load, mod = mod, uncertainty = uncertainty)                       
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
                if CHPQI >= self.CHPQI_threshold:
                    pass #do nothing, CHPQI is good already
                else:
                    niter = 0
                    count = 0
                    while CHPQI < self.CHPQI_threshold and niter < 300:   
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
                            if (new_CHPQI-CHPQI)/CHPQI < 0.00001:
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
        operation_data  = self.calculate_CHPQI(part_load, mod = mod, uncertainty = uncertainty) 
        CHPQI = operation_data[0]
        
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
        operation_data  = self.calculate_CHPQI(part_load, mod = mod, uncertainty = uncertainty) 
        CHPQI = operation_data[0]
        
        if time_start is not None or time_stop is not None or table_string is not None: 
                self.putUtility(time_start =old_time_start, time_stop = old_time_stop, table_string=old_price_table)         

        return(BAU_op_cost_HH_pound, op_cost_HH_pound, part_load, CHPQI)
        
        
        #find the part load of tech and cost considering a CHP which follows the load and is turned on only during trading hours
    def Greigs(self, tech_id= None, time_start = None, time_stop = None, table_string=None, mod=None, uncertainty=None):
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
                if HH > 32 and HH < 39:
                    part_load[count] = 1                
                else:
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
        operation_data  = self.calculate_CHPQI(part_load, mod = mod, uncertainty = uncertainty) 
        CHPQI = operation_data[0]
        
        if time_start is not None or time_stop is not None or table_string is not None: 
                self.putUtility(time_start =old_time_start, time_stop = old_time_stop, table_string=old_price_table)         

        return(BAU_op_cost_HH_pound, op_cost_HH_pound, part_load, CHPQI)
        
        
        
        
    ## find optimal unit operation by solving the corresponding GAMS problem. The GAMS problem can be edited to include/exclude constarints. 
    def MILPOptiControl(self, tech_id= None, MaxTimeStep = None, part_load = None, cluster = None, escCluster = None, time_start = None, time_stop = None, table_string=None, mod=None, uncertainty=None):
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
        ## initialise with a previous part load to speed up solution
        if part_load is not None:
            if len(part_load) != len(el_demand):
                raise Exception("part load length and demand length are not the same!")
        else:
            part_load = np.zeros(len(el_demand))

        GAMS_model = "CHP.gms"
        #to test and debug model use MaxTimeStep
        if MaxTimeStep is not None:
            part_load = part_load[0:MaxTimeStep]; el_price = el_price[0:MaxTimeStep]; el_price_exp = el_price_exp[0:MaxTimeStep]; gas_price = gas_price[0:MaxTimeStep]; th_demand = th_demand[0:MaxTimeStep]; el_demand = el_demand[0:MaxTimeStep]; gas_price_CHP = gas_price_CHP[0:MaxTimeStep];
                           
        ws = GamsWorkspace(GAMS_path)
        db = ws.add_database()                        
  
        
        #### GAMS problem input
        ### add sets
        time_set = np.char.mod('%d', range(len(el_demand)))
        t = db.add_set("t", 1, "")
        for time_i in time_set:
            t.add_record(time_i)        
     
        ### add parameters   
        p0 = db.add_parameter_dc("part_load", [t], "")  
        p1 = db.add_parameter_dc("el_demand", [t], "")
        p2 = db.add_parameter_dc("th_demand", [t], "")
        p3 = db.add_parameter_dc("el_price", [t], "")
        p4 = db.add_parameter_dc("el_price_exp", [t], "")
        p5 = db.add_parameter_dc("gas_price", [t], "")
        p6 = db.add_parameter_dc("gas_price_CHP", [t], "")
        
        for i in range(len(time_set)):
            time_i = time_set[i]
            p0.add_record(time_i).value = part_load[i]
            p1.add_record(time_i).value = el_demand[i]
            p2.add_record(time_i).value = th_demand[i]
            p3.add_record(time_i).value = el_price[i]
            p4.add_record(time_i).value = el_price_exp[i]
            p5.add_record(time_i).value = gas_price[i]
            p6.add_record(time_i).value = gas_price_CHP[i]  
            
            
        var_name =  ['Boiler_eff', 'a_fuel', 'b_fuel', 'a_el', 'b_el', 'a_th',  'b_th', 'psi_min', 'parasitic_load', 'mant_costs'] 
        for i in range(len(tech_data)):
            var = "C{0}".format(i)
            var = db.add_parameter(var_name[i], 0, "")
            var.add_record().value = tech_data[i]
            
        # solve
        t4 = ws.add_job_from_file(GAMS_path + "\\" + GAMS_model)          
        if cluster == 1:
            db.export(GAMS_path + "\\db.gdx")   ##create a GDX file of input to be sent to cluster
            con = cc.ClusterConnection()        ## connect to cluster
            con.SubmitGAMSJob(GAMS_model, escCluster = escCluster)   ## submit GAMS job                  
             #out_db = ws.add_database_from_gdx("output.gdx")#, database_name = "out_db")    it doesnt work. get error when trying to extract varbiable after           
        else:                    # retrieve GAMS model   
            opt = ws.add_options()
            opt.defines["gdxincname"] = db.name
            t4.run(opt, databases = db)
            
        # return solutions       WE SHOULD CREATE A METHOD HERE.
        obj = 0
        part_load0 = np.empty(shape = (len(el_demand,)))
        if cluster ==1: 
            if escCluster == 1:
                print("still need to code this")
            else:               
                conn = sqlite3.connect(GAMS_path + "\\output.db")
                cur = conn.cursor()      
                cur.execute("select name, level from scalarvariables")
                scalarvariables = cur.fetchall()
                for rec in scalarvariables:
                    if rec[0] == 'z':
                       obj = rec[1]/100
                cur.execute('''SELECT t, level FROM psi''')
                psi_var = cur.fetchall()
                i = 0
                for rec in psi_var:
                    part_load0[i] = rec[1]
                    i=i+1 
                conn.commit()
                conn.close() 
        else:  
            res_part_load = t4.out_db["psi"]
            res_out = t4.out_db["z"]        
            i = 0
            for rec in res_part_load:
                part_load0[i] = rec.level
                i=i+1               
            for rec in res_out:               
                obj = rec.level/100

        #check for consistency of solutions        
        if obj ==0:
            raise Exception("Objective function is zero..weird")                 
        cost_HH = self.calculate_op_cost(part_load0)
        obj2 =sum(cost_HH[0])
        if abs(obj - obj2)/obj > 0.001:
            raise Exception("Objective function value is different from the operational costs. It shouldn't be") 
           
        ## allow to calculate operation only on a fraction of the data to speed up solution and debugging 
        sol = self.SimpleOptiControl(tech_id = self.tech.tech_id)
        part_load2 = sol[2]
        part_load2[0:len(part_load0)] = part_load0
        part_load = part_load2
        [op_cost_HH_pound, BAU_op_cost_HH_pound ] = self.calculate_op_cost(part_load, mod = mod, uncertainty = uncertainty)
        operation_data  = self.calculate_CHPQI(part_load, mod = mod, uncertainty = uncertainty) 
        CHPQI = operation_data[0]
        
        if time_start is not None or time_stop is not None or table_string is not None: 
                self.putUtility(time_start =old_time_start, time_stop = old_time_stop, table_string=old_price_table)         

        return(BAU_op_cost_HH_pound, op_cost_HH_pound, part_load, CHPQI)

    #################################################################
    ### useful fucntion to calculate recurring quantities         ###    
    #################################################################
    
    ## get the operating cost from the part load
    def calculate_op_cost(self, part_load, tech_id = None, time_start = None, time_stop = None, table_string=None, mod=None, uncertainty=None):               
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
        if len(part_load) < len(el_demand):
            print("Note: I am calculating operational cost but part load length is less than demand length length. Recalculating demands...")
            MaxTimeStep = len(part_load)
            el_price = el_price[0:MaxTimeStep]; el_price_exp = el_price_exp[0:MaxTimeStep]; gas_price = gas_price[0:MaxTimeStep]; th_demand = th_demand[0:MaxTimeStep]; el_demand = el_demand[0:MaxTimeStep]; gas_price_CHP = gas_price_CHP[0:MaxTimeStep];
        if len(part_load) > len(el_demand):
           raise Exception("part load length is more than demand length length")         
        check_psi = np.array(part_load)
        check_psi[check_psi < 0.001] = 1
       #commented to be able to perform simulations using real part laod data
        # if min(check_psi)< (psi_min)*0.999:
        #    raise Exception("part load less than minimum part load")      

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
        check_psi[check_psi < 0.001] = 1
        if min(check_psi)< (psi_min)*0.999:
            raise Exception("part load less than minimum part load")   
            
        mask000 = part_load > 0.01
        mask011 = (a_el*part_load+b_el)*mask000 > el_demand
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
        
        avg_part_load = np.mean(part_load*mask000)
        CHP_utilisation = np.mean(part_load)
        fuel_boiler = (sum(th_demand) - th_tot_utilisation)/Boiler_eff
        ele_import = sum((el_demand-(a_el*part_load+b_el)*mask000)*(1-mask011)) 
        ele_export = sum(((a_el*part_load+b_el)*mask000-el_demand)*(mask011))
        
        if time_start is not None or time_stop is not None: 
                self.putUtility(time_start =old_time_start, time_stop = old_time_stop)    
            
        return(CHPQI, fuel_tot_utilisation, el_tot_utilisation, th_tot_utilisation, fuel_boiler, ele_import, ele_export, avg_part_load, CHP_utilisation)
      
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
        el_price = self.store.p_ele*mod[0] + self.store.cf_ele*self.store.crc/1000*100
        el_price_exp = self.store.p_ele_exp*mod[0]*mod[3] + self.store.cf_ele*self.store.crc/1000*100
        gas_price = self.store.p_gas*mod[1]  + self.store.cf_gas*self.store.crc/1000*100
        gas_price_CHP = (self.store.p_gas + mant_costs*el_efficiency*100)*mod[1] + self.store.cf_gas*self.store.crc/1000*100   #modify this to accout for biomethane
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
            cash_flow = [-Total_capex + year_savings] + [year_savings]*(tech_lifetime-1)
            IRR = np.irr(cash_flow)
            return(year_savings, payback, NPV5savings, ROI, Cum_disc_cash_flow, IRR)
    

    ################################################################################
    ### useful fucntion to intialise properties by accessing the database        ###    
    ################################################################################


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

       
