# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 11:43:23 2018

@author: nl211
"""

import sqlite3
import numpy as np
import pandas as pd
import datetime
import calendar
import os
import Common.classStore as st # Module is in seperate folder, hence the elaboration
import Common.classTech as tc
#import Connection.classClusterConnection as cc
#from gams import GamsWorkspace



############################################
### Problem input ###    
############################################

## define problem granularity ###
time_window = 30
stores = 2
domain_decomposition = 1
year_start = 2020
year_stop = 2050

########### define scenarios #########
ele_price_increase = 0.05  # % electricity price increase each year
capex_reduction_CHP = 0.03 # % capex reduction each year

############################################
### generate coefficients ###    
############################################
year = np.linspace(year_start,year_stop, time_window+1)[0:-1]
## array of ele_price_modifier 
mod = np.power(1+ ele_price_increase, np.linspace(year_start,year_stop, time_window+1)[0:-1] - year_start)
## calcualte opexSavings and CO2 coeff
K_ele = mod
K_co2 = mod

# retrieve capex and calculate coeff
K_capex_tech1 = [2, 4] 
K_capex_tech2 = [1, 8] 



#### opex savings coefficients
#### CO2 savings coefficients
# Capex coefficent no regression
## carbon targget. 

############################################
### generate GAMS gdx file ###    
############################################
ws = GamsWorkspace(GAMS_path)
db = ws.add_database()                        
         
        
        #### GAMS problem input
        ### add sets
        time_set = np.char.mod('%d', range(len(el_demand)))
        t = db.add_set("t", 1, "")
        
        for time_i in time_set:
            t.add_record(time_i)  
            
     ### add parameters   
        p0 = db.add_parameter_dc("K_co2", [t], "")  
        p1 = db.add_parameter_dc("K_opex", [t], "")
        p2 = db.add_parameter_dc("K0_capex", [t], "")
        p3 = db.add_parameter_dc("K1_capex", [t], "")
        p4 = db.add_parameter_dc("CO2_savingTarget", [t], "")
        p5 = db.add_parameter_dc("Max_x", [t], "")
        p6 = db.add_parameter_dc("IO_modular", [t], "")
        
        for i in range(len(time_set)):
            time_i = time_set[i]
            p0.add_record(time_i).value = K_co2[i]
            p1.add_record(time_i).value = K_opex[i]
            p2.add_record(time_i).value = K0_capex[i]
            p3.add_record(time_i).value = K1_capex[i]
            p4.add_record(time_i).value = CO2_savingTarget[i]
            p5.add_record(time_i).value = Max_x[i]
            p6.add_record(time_i).value = IO_modular[i]  

            
            
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





############################################
### run GAMS model and retrieve results ###    
############################################



############################################
### plot stuff ###    
############################################

