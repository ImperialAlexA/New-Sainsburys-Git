# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 16:44:28 2018
@author: Anatole
"""

#import classPV_CHP as PC
import classPV_CHP as PC
import sqlite3
import numpy as np
import datetime
from gams import GamsWorkspace

start = datetime.datetime.now()

scenario_names = ['steady state','two degrees','slow progression','consumer power']
scenarios = [[0.045,0.025,-0.0025,-0.0075,-0.03],
             [0.07,0.035,-0.0025,-0.015,-0.06],
             [0.06,0.03,-0.0025,-0.01,-0.04],
             [0.035,0.035,-0.005,-0.0125,-0.03]] #[Steady State:[ele,gas,CHP,PV,cf], Slow progression:[],Two degrees:[],Consumer power: []]


for scen in range(0,1):
    print(scenario_names[scen])

    database_path = "Sainsburys.sqlite"
    conn = sqlite3.connect(database_path)
    cur = conn.cursor()
    
    cur.execute('''SELECT Stores_id FROM Demand_Check Where Ele= {vn1} and Gas= {vn2}'''.format(vn1=1,vn2=1))
    Index = cur.fetchall()
    Store_id_range = np.array([elt[0] for elt in Index],dtype=np.float64)
    conn.close()
    
    Store_id_range = np.delete(Store_id_range,44) # =store 2017 not included because of errors
    
    time_window = 1 
    stores = 1
    year_start = 2020
    year_stop = 2050
    NG_True_False = True #True: Natural Gas is used as fuel for CHP, False:Biomethane is used as fuel for CHP
    time_window_length=(year_stop-year_start)/time_window
    
# =============================================================================
#     gas_CF = 0.18416
#     ele_CF =   0.35156 
#     Gas_total = []
#     Ele_total = []
#     for id_store in Store_id_range[:stores]:
#         Gas_total.append(sum(BBC.CHPproblem(id_store).store.d_gas))
#         Ele_total.append(sum(BBC.CHPproblem(id_store).store.d_ele))
#     Carbon = sum(Gas_total)*gas_CF+sum(Ele_total)*ele_CF
#     
#     CO2_target = np.ones(time_window)*[0.50,0.95]*Carbon/1000
# =============================================================================
    CO2_target = np.zeros(time_window)
    tech_range = ['PV', 'CHP','dummy','ppa']
    modular = [1,0,1,1]
    split = 2 #SPlit the data for opex and carbon to generate coef of piecewise linear function
    
    #define the coefficients for ppa manually to allow for future addition of ppa to model (now all zeros)
    ppa_co2_coef = np.zeros(4) #CO2 savings=ppa_co2_coef*ppa_size
    ppa_opex_coef = np.zeros(4) #opex savings=ppa_opex_coef*ppa_size
    ppa_limit_bot_opex = np.zeros(4)
    ppa_limit_top_opex = np.ones(4)
    ppa_limit_bot_co2 = np.zeros(4)
    ppa_limit_top_co2 = np.ones(4)
    
    dummy_limit_bot_opex = np.zeros(4)
    dummy_limit_top_opex =2*np.ones(4)
    dummy_limit_bot_co2 = np.zeros(4)
    dummy_limit_top_co2 = 2*np.ones(4)
    
    
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
    
    
    Carbon_matrix =[]
    OPEX_matrix = []
    x_limit_bot_opex_matrix = []
    x_limit_top_opex_matrix = []
    x_limit_bot_co2_matrix = []
    x_limit_top_co2_matrix = []
    
    for store_id in Store_id_range[:stores]:
        a = np.where(Store_id_range == store_id)
        print('store:%d, %d/%d' %(store_id,a[0][0]+1,len(Store_id_range[:stores])))
        Carbonh = []
        OPEXh = []
        Capex_p0 = []
        Capex_p1 = []
        x_limit_bot_opex_h = []
        x_limit_top_opex_h = []
        x_limit_bot_co2_h = []
        x_limit_top_co2_h = []
        for n in range(0,time_window):
            print('Time window:%d' %year[n])
    
            solution = PC.PV_CHP(store_id,p_elec_mod=p_elec_mod[n], p_gas_mod=p_gas_mod[n], PV_price_mod= PV_mod[n], CHP_price_mod=CHP_mod[n], cf_mod=cf_mod[n]).function_approx(spl=split,NG=NG_True_False)
            OPEX_p = solution[1]
            CARBON_p = solution[2]
            CAPEX_PV_p =  solution[3]
            CAPEX_CHP_p = solution[4]
            x_limit_bot_opex = solution[5]
            x_limit_top_opex = solution[6]
            x_limit_bot_co2 = solution[7]
            x_limit_top_co2 = solution[8]
            
            Carbonh.append(np.vstack([CARBON_p,ppa_co2_coef]))
            OPEXh.append(np.vstack([OPEX_p, ppa_opex_coef]))
            Capex_p0.append([CAPEX_PV_p[1],CAPEX_CHP_p[1],0,0]) # two last entries are for dummy and ppa
            Capex_p1.append([CAPEX_PV_p[0],CAPEX_CHP_p[0],0,0])
            
            x_limit_bot_opex_h.append(np.vstack([x_limit_bot_opex,dummy_limit_bot_opex,ppa_limit_bot_opex]))
            x_limit_top_opex_h.append(np.vstack([x_limit_top_opex,dummy_limit_top_opex,ppa_limit_top_opex]))
            x_limit_bot_co2_h.append(np.vstack([x_limit_bot_co2,dummy_limit_bot_co2,ppa_limit_bot_co2]))
            x_limit_top_co2_h.append(np.vstack([x_limit_top_co2,dummy_limit_top_co2,ppa_limit_top_co2]))
            
            
        Carbon_matrix.append(Carbonh)
        OPEX_matrix.append(OPEXh)
        x_limit_bot_opex_matrix.append(x_limit_bot_opex_h)
        x_limit_top_opex_matrix.append(x_limit_top_opex_h)
        x_limit_bot_co2_matrix.append(x_limit_bot_co2_h)
        x_limit_top_co2_matrix.append(x_limit_top_co2_h)
        
    
        
    ############################################
    ### generate GAMS gdx file ###    
    ############################################
    GAMS_model = "Strategic.gms"
    ws = GamsWorkspace()
    db =ws.add_database()
    
    time_set = np.char.mod('%d', year)
    store_set = np.char.mod('%d', Store_id_range[:stores])
    tech_set = np.array(tech_range)
    split_set = np.char.mod('%d', np.arange(split**2))
    
    tech = db.add_set("tech",1,"")
    t = db.add_set("t",1,"")
    s = db.add_set("s",1,"")
    d = db.add_set("d",1,"")
    
    
    for n in tech_set:
        tech.add_record(n)
    for m in time_set:
        t.add_record(m)
    for z in store_set:
        s.add_record(z)
    for k in split_set:
        d.add_record(k)
        
    p0 = db.add_parameter_dc("K_co2", [d,tech,t,s], "")
    p1 = db.add_parameter_dc("K_opex", [d,tech,t,s], "")
    p2 = db.add_parameter_dc("K0_capex", [tech,t], "")
    p3 = db.add_parameter_dc("K1_capex", [tech,t], "")
    p4 = db.add_parameter_dc("CO2_savingTarget", [t], "")
    p5 = db.add_parameter_dc("IO_modular", [tech], "")
    p6 = db.add_parameter_dc("x_limit_bot_opex", [d,tech,t,s], "")
    p7 = db.add_parameter_dc("x_limit_top_opex", [d,tech,t,s], "")
    p8 = db.add_parameter_dc("x_limit_bot_co2", [d,tech,t,s], "")
    p9 = db.add_parameter_dc("x_limit_top_co2", [d,tech,t,s], "")
           
    for i in range(len(tech_set)):
        tech_i = tech_set[i]
        p5.add_record(tech_i).value = modular[i]
        
        for j in range(len(time_set)):
            time_j = time_set[j] 
            p2.add_record([tech_i, time_j]).value = Capex_p0[j][i]
            p3.add_record([tech_i, time_j]).value = Capex_p1[j][i]
    
            for z in range(len(store_set)):
                store_z = store_set[z]
                
                for k in range(len(split_set)):
                    split_k = split_set[k]
                    
                    p0.add_record([split_k, tech_i, time_j, store_z]).value = Carbon_matrix[z][j][i][k]      
                    p1.add_record([split_k, tech_i, time_j, store_z]).value = OPEX_matrix[z][j][i][k]
                    p6.add_record([split_k, tech_i, time_j, store_z]).value = x_limit_bot_opex_matrix[z][j][i][k]
                    p7.add_record([split_k, tech_i, time_j, store_z]).value = x_limit_top_opex_matrix[z][j][i][k]
                    p8.add_record([split_k, tech_i, time_j, store_z]).value = x_limit_bot_co2_matrix[z][j][i][k]
                    p9.add_record([split_k, tech_i, time_j, store_z]).value = x_limit_top_co2_matrix[z][j][i][k]
    
    for j in range(len(time_set)):
        time_j = time_set[j]   
        p4.add_record(time_j).value = CO2_target[j]
    
    
#    db.export("C:\\Users\\Anatole\\Documents\\GitHub\\New-Sainsburys-Git\\" +scenario_names[scen] +".gdx")

end = datetime.datetime.now()
print(end-start)