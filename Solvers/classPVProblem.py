# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:24:43 2018

@author: Alex  
"""
import sqlite3
import numpy as np
import datetime
import calendar
import os
import sys
import pandas 
scriptpath = ".\\Common" # This is written in the Windows way of specifying paths, hopefully it works on Linux?
sys.path.append(os.path.abspath(scriptpath))
import Common.classStore as st # Module is in seperate folder, hence the elaboration
import Common.classPVTech as pvtc
#from pyomo.environ import * # Linear programming module
#import pyomo as pyo
#import pyomo.environ as pyo
#from pyomo.environ import *
#from pyomo.opt import SolverFactory # Solver
import time # To time code, simply use start = time.clock() then print start - time.clock()


class PVproblem:
   
    
    def __init__(self, store_id):           
        self.store = st.store(store_id)
        self.price_table = 'Utility_Prices_Aitor'
        default_initial_time = datetime.datetime(2016,1,1)
        default_final_time = datetime.datetime(2017,1,1)
        self.time_start= int((default_initial_time-datetime.datetime(1970,1,1)).total_seconds()/60/30)
        self.time_stop= int((default_final_time-datetime.datetime(1970,1,1)).total_seconds()/60/30)
        self.store.getSimplePrice(self.time_start, self.time_stop, self.price_table)
        self.store.getSimpleDemand(self.time_start, self.time_stop)
        #put self. back 
        
        self.discount_rate = 0.09
        self.roof_max_weight = 16 #(kg/m2)
        self.roof_area= 400 #m2
        self.hidden_cost=2000 #£
        self.Roof_space_coeff=0.6 
        self.roof_available_area= self.roof_area*self.Roof_space_coeff #m2
        self.cf_ele=0.412 #kgCO2/kWh
        
        
        
    def putTechPV(self, tech_id): 
        self.tech = pvtc.tech(tech_id)

        
    def OptiPVpanels(self, method = None, tech_range = None, time_start = None, time_stop = None, table_string = None, ECA_value = 0, uncertainty = None, mod = None):
        if time_start is not None or time_stop is not None or table_string is not None: 
                old_time_start = self.time_start; old_time_stop = self.time_stop;  old_price_table = self.price_table
                self.putUtility(time_start =time_start, time_stop = time_stop, table_string=table_string) 
#        discount_rate = 0.09
#        roof_max_weight = 16 #(kg/m2)
#        roof_area= 200 #m2
#        hidden_cost=2000 #£
#        Roof_space_coeff=0.6 
#        roof_available_area= roof_area*Roof_space_coeff #m2       
        opti_savings=0
        tech_range = range(1,4) 
        df = pandas.read_excel('Irradiance-data.xlsx')
        irradiance = df['161_data'].values
        
        for tech_id in tech_range:
            # initialize
            opti_savings=0
            self.putTechPV(tech_id)
            tech_name = self.tech.PVtech_name
            tech_price = self.tech.PV_capex #(£/Wp,yr)
            tech_lifetime = self.tech.PV_lifetime
            tech_eff = self.tech.PV_eff
            tech_area = self.tech.PV_Area #(m2)
            tech_weight = self.tech.PV_Weight #(kg)
            tech_power = self.tech.PV_Nominal_Power
            elec_price=self.store.p_ele
            gas_price=self.store.p_gas
            Store_demand = self.store.d_ele
            gas_demand=self.store.d_gas
    
    
            if tech_weight / tech_area < self.roof_max_weight:
                #irradiance = np.array([0,0,1,2,3,6,3,2,1,0,0])
                Indiv_Elec_prod = (tech_eff * np.array(irradiance)*tech_area)/3600/2
                N_panel = int(self.roof_available_area / tech_area)
                Total_Elec_prod = N_panel * Indiv_Elec_prod 
                panel_price = tech_price*tech_power
               # Store_demand = np.array([0, 0, 3, 5, 6, 7, 23, 34, 120, 23, 0])
                
                mask0=(Total_Elec_prod<Store_demand).astype(int)
                mask1=(Total_Elec_prod>Store_demand).astype(int)
                Elec_grid = mask0*(Store_demand - Total_Elec_prod)
                Elec_surplus=mask1*(Total_Elec_prod-Store_demand)
                
                # Costs
                
                #policies=0.001
                #gas_demand = np.array([0, 0, 1, 3, 2, 3, 9, 15, 31, 6, 0])
                Total_capex = N_panel * panel_price + self.hidden_cost 
                Opex_savings = Total_Elec_prod *(elec_price)*10**(-2) #+policies) 
                op_cost_HH_pound = (Elec_grid-Elec_surplus)*elec_price*10**(-2) + gas_demand*gas_price*10**(-2)
                BAU_carbon=Store_demand*self.cf_ele/ 1000 # (tCO2)
                Carbon_PV=Total_Elec_prod*self.cf_ele/ 1000 # (tCO2)
                Carbon_savings=(BAU_carbon-Carbon_PV) # (tCO2)
                if np.sum(Opex_savings) > opti_savings: 
                    best_tech = tech_name
                    opti_capex=Total_capex
                    opti_ele_prod=Total_Elec_prod
                    opti_savings= np.sum(Opex_savings)
                    opti_panel=N_panel
                    opti_carbon=Carbon_savings
                else:
                    pass 
            return (best_tech, opti_savings, opti_capex,opti_ele_prod,opti_panel,opti_carbon)

    def calculate_financials(self, discount_rate, tech_lifetime, year_BAU_cost, year_op_cost, Total_capex):
        numb_years = 10 #(self.time_stop-self.time_start)/2/24/365
        year_op_cost = sum(op_cost_HH_pound)/numb_years
        year_BAU_cost = 1330 #sum(BAU_op_cost_HH_pound)/numb_years
        year_savings = year_BAU_cost - year_op_cost
        payback = Total_capex / year_savings
        ann_capex = -np.pmt(discount_rate, tech_lifetime, Total_capex)
        year_cost = year_op_cost + ann_capex
        NPV5_op_cost = -np.npv(discount_rate, np.array([year_cost] * 5))
        NPV5_BAU_cost = -np.npv(discount_rate, np.array([year_BAU_cost] * 5))
        NPV5savings = NPV5_op_cost - NPV5_BAU_cost
        ROI = year_savings / Total_capex
        Const = (1 - (1 + discount_rate) ** (-tech_lifetime)) / discount_rate
        Cum_disc_cash_flow = -Total_capex + Const * year_savings
        return (year_savings, payback, NPV5savings, ROI, Cum_disc_cash_flow)



    


            # store best value


            # output optimum and KPIs

    def SimulatePVonAllRoof(self, tech_id, N_panel ): 

        df = pandas.read_excel('Irradiance-data.xlsx')
        irradiance = df['161_data'].values
        # initialize
        opti_savings=0
        self.putTechPV(tech_id)
        tech_name = self.tech.PVtech_name
        tech_price = self.tech.PV_capex #(£/Wp,yr)
        tech_lifetime = self.tech.PV_lifetime
        tech_eff = self.tech.PV_eff
        tech_area = self.tech.PV_Area #(m2)
        tech_weight = self.tech.PV_Weight #(kg)
        tech_power = self.tech.PV_Nominal_Power
        elec_price=self.store.p_ele
        gas_price=self.store.p_gas
        Store_demand = self.store.d_ele
        gas_demand=self.store.d_gas

        max_panel_number=int(self.roof_available_area / tech_area)
        
        if N_panel>max_panel_number:
            N_panel=max_panel_number
        
        if tech_weight / tech_area < self.roof_max_weight:
                #irradiance = np.array([0,0,1,2,3,6,3,2,1,0,0])
            Indiv_Elec_prod = (tech_eff * np.array(irradiance)*tech_area)/3600/2
            #N_panel = int(self.roof_available_area / tech_area)
            Total_Elec_prod = N_panel * Indiv_Elec_prod
            Annual_Elec_prod = np.sum(Total_Elec_prod)
            panel_price = tech_price*tech_power
            # Store_demand = np.array([0, 0, 3, 5, 6, 7, 23, 34, 120, 23, 0])
                    
            mask0=(Total_Elec_prod<Store_demand).astype(int)
            mask1=(Total_Elec_prod>Store_demand).astype(int)
            Elec_grid = mask0*(Store_demand - Total_Elec_prod)
            Elec_surplus=mask1*(Total_Elec_prod-Store_demand)
                    
                    # Costs
                    
                    #policies=0.001
                    #gas_demand = np.array([0, 0, 1, 3, 2, 3, 9, 15, 31, 6, 0])
            BAU_carbon=Store_demand*self.cf_ele/ 1000 # (tCO2)
            Carbon_PV=Total_Elec_prod*self.cf_ele/ 1000 # (tCO2)
            Carbon_savings=(BAU_carbon-Carbon_PV) # (tCO2)
            Annual_Carbon_savings=np.sum(Carbon_savings)
            Total_capex = N_panel * panel_price + self.hidden_cost 
            Opex_savings = Total_Elec_prod *(elec_price)*10**(-2) #+policies) 
            Annual_Opex_savings=np.sum(Opex_savings)
            op_cost_HH_pound = (Elec_grid-Elec_surplus)*elec_price*10**(-2) + gas_demand*gas_price*10**(-2)
                   
        return (tech_name, Annual_Opex_savings, Total_capex, Annual_Elec_prod,Annual_Carbon_savings,N_panel)
                # get tech data
                # connect to database
                # retrieve data
                # store data

            #self.PVtech.eff = 9


            ###



            ###
            # find number of panels
            # find output per panels by multpy eff times irradiance
            # find totla electricity
            # find cost reduction
            # find revenue from policy

            ####


            ###
            # output some indicators, savings, capex, payback time, irr
            ####