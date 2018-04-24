# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:00:37 2018

@author: Anatole
"""

import Solvers.classCHPProblemnew as BBC
import numpy as np
import matplotlib.pyplot as plt
import Solvers.classPVProblem as pb
import sqlite3


database_path = "Sainsburys.sqlite"
conn = sqlite3.connect(database_path)
cur = conn.cursor()

cur.execute('''SELECT Stores_id FROM Demand_Check Where Ele= {vn1} and Gas= {vn2}'''.format(vn1=1,vn2=1))
Index = cur.fetchall()
Store_id_range = np.array([elt[0] for elt in Index],dtype=np.float64)

Store_id_range = np.delete(Store_id_range,44) # =store 2017 not included because of errors

Cum_disc_cash_flow_PV_cat_1 =[]
Cum_disc_cash_flow_PV_cat_2 =[]
Cum_disc_cash_flow_PV_cat_3 =[]
Cum_disc_cash_flow_CHP_cat_1 =[]
Cum_disc_cash_flow_CHP_cat_2 =[]
Cum_disc_cash_flow_CHP_cat_3 =[]
Carbon_PV_1 =[]
Carbon_PV_2 =[]
Carbon_PV_3 =[]
Carbon_CHP_1 =[]
Carbon_CHP_2 =[]
Carbon_CHP_3 =[]
Cum_disc_cash_flow_PV = []
Cum_disc_cash_flow_CHP = []
Carbon_PV =[]
Carbon_CHP =[]
Cum_disc_cash_flow_south=[]
Carbon_PV_south=[]
Cum_disc_cash_flow_central=[]
Carbon_PV_central=[]
Cum_disc_cash_flow_north=[]
Carbon_PV_north=[]

for store_id in Store_id_range: 
    a = pb.PVproblem(store_id)
    a.elec_price = a.store.p_ele*11.9/9.87
    [best_tech, opti_savings, opti_capex,opti_ele_prod,opti_panel,opti_carbon,Cum_disc_cash_flow] = a.OptiPVpanels()
    
    b=BBC.CHPproblem(store_id)
    b.store.p_ele =b.store.p_ele*11.9/9.87
    [opti_tech, opti_tech_name, opti_CHPQI, opti_part_load, financials, carbon, year_BAU_cost] =b.SimpleOpti5NPV(ECA_value = 0.26)
    Area = a.store.area

    Cum_disc_cash_flow_PV.append(Cum_disc_cash_flow)
    Cum_disc_cash_flow_CHP.append(financials[-2])
    Carbon_PV.append(opti_carbon)
    Carbon_CHP.append(carbon[2])
    
    if Area <25000:
        Cum_disc_cash_flow_PV_cat_1.append(Cum_disc_cash_flow)
        Carbon_PV_1.append(opti_carbon)
        Cum_disc_cash_flow_CHP_cat_1.append(financials[-2])
        Carbon_CHP_1.append(carbon[2])
        
    elif 25000<Area<45000:
        Cum_disc_cash_flow_PV_cat_2.append(Cum_disc_cash_flow)
        Carbon_PV_2.append(opti_carbon)
        Cum_disc_cash_flow_CHP_cat_2.append(financials[-2])
        Carbon_CHP_2.append(carbon[2])
    elif Area>45000:
        Cum_disc_cash_flow_PV_cat_3.append(Cum_disc_cash_flow)
        Carbon_PV_3.append(opti_carbon)
        Cum_disc_cash_flow_CHP_cat_3.append(financials[-2])
        Carbon_CHP_3.append(carbon[2])
        
    cur.execute('''SELECT Zone FROM Stores Where id= {vn1}'''.format(vn1=store_id))
    Index = cur.fetchall()
    Location = np.array([elt[0] for elt in Index])
    if Location == "South":
        Cum_disc_cash_flow_south.append(Cum_disc_cash_flow)
        Carbon_PV_south.append(opti_carbon)
    if Location == "Central":
        Cum_disc_cash_flow_central.append(Cum_disc_cash_flow)
        Carbon_PV_central.append(opti_carbon)
    if Location == "North":
        Cum_disc_cash_flow_north.append(Cum_disc_cash_flow)
        Carbon_PV_north.append(opti_carbon)

# =============================================================================
# PV MACC with locations
# =============================================================================
Cum_disc_cash_flow = [np.average(Cum_disc_cash_flow_south),np.average(Cum_disc_cash_flow_central),np.average(Cum_disc_cash_flow_north)]
Carbon = [np.average(Carbon_PV_south),np.average(Carbon_PV_central),np.average(Carbon_PV_north)]

MAC_PV = -np.array(Cum_disc_cash_flow)/abs(np.array(Carbon))
width = Carbon
cum_width = np.cumsum(width)
ind = [width[0]/2, (cum_width[1]-cum_width[0])/2+cum_width[0], (cum_width[2]-cum_width[1])/2+cum_width[1]]

plt.figure(4)
plt.bar(ind[0], MAC_PV[0],width[0], color = 'powderblue',edgecolor = 'black',label='South')
plt.bar(ind[1], MAC_PV[1],width[1], color = 'steelblue',edgecolor = 'black',label='Central')
plt.bar(ind[2], MAC_PV[2],width[2], color = 'navy',edgecolor = 'black',label='North')
plt.xlabel('$tCO_2e$ yearly savings')
plt.ylabel('$£/tCO_2e$')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.title('MAC curves for PV vs location')

# =============================================================================
# CHP MACCC with CATEGORIES
# =============================================================================
Cum_disc_cash_flow = [np.average(Cum_disc_cash_flow_CHP_cat_3),np.average(Cum_disc_cash_flow_CHP_cat_2),np.average(Cum_disc_cash_flow_CHP_cat_1)]
Carbon = [np.average(Carbon_CHP_3),np.average(Carbon_CHP_2),np.average(Carbon_CHP_1)]

MAC_PV = -np.array(Cum_disc_cash_flow)/abs(np.array(Carbon))
width = Carbon
cum_width = np.cumsum(width)
ind = [width[0]/2, (cum_width[1]-cum_width[0])/2+cum_width[0], (cum_width[2]-cum_width[1])/2+cum_width[1]]

plt.figure(1)
plt.bar(ind[0], MAC_PV[0],width[0], color = 'powderblue',edgecolor = 'black',label='Category 3')
plt.bar(ind[1], MAC_PV[1],width[1], color = 'steelblue',edgecolor = 'black',label='Category 2')
plt.bar(ind[2], MAC_PV[2],width[2], color = 'navy',edgecolor = 'black',label='Category 1')
plt.xlabel('$tCO_2e$ yearly savings')
plt.ylabel('$£/tCO_2e$')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

plt.title('MAC curves for CHP vs Area')

# =============================================================================
# PV and CHP MACCC without CATEGORIES
# =============================================================================
Cum_disc_cash_flow = [np.average(Cum_disc_cash_flow_PV),np.average(Cum_disc_cash_flow_CHP)]
Carbon = [np.average(Carbon_PV), np.average(Carbon_CHP)]

MAC = -np.array(Cum_disc_cash_flow)/abs(np.array(Carbon))
width = Carbon
cum_width = np.cumsum(width)
ind = [width[0]/2, (cum_width[1]-cum_width[0])/2+cum_width[0]]

plt.figure(2)
plt.bar(ind[0], MAC[0],width[0], color = 'powderblue',edgecolor = 'black',label='PV')
plt.bar(ind[1], MAC[1],width[1], color = 'steelblue',edgecolor = 'black',label='CHP')
plt.xlabel('$tCO_2e$ yearly savings')
plt.ylabel('$£/tCO_2e$')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

plt.title('MAC curves')



# =============================================================================
# CATEGORIES AND PV MACC
# =============================================================================
Cum_disc_cash_flow = [np.average(Cum_disc_cash_flow_PV_cat_3),np.average(Cum_disc_cash_flow_PV_cat_2),np.average(Cum_disc_cash_flow_PV_cat_1)]
Carbon = [np.average(Carbon_PV_3),np.average(Carbon_PV_2),np.average(Carbon_PV_1)]

MAC_PV = -np.array(Cum_disc_cash_flow)/abs(np.array(Carbon))
width = Carbon
cum_width = np.cumsum(width)
ind = [width[0]/2, (cum_width[1]-cum_width[0])/2+cum_width[0], (cum_width[2]-cum_width[1])/2+cum_width[1]]

plt.figure(3)
plt.bar(ind[0], MAC_PV[0],width[0], color = 'powderblue',edgecolor = 'black',label='Category 3')
plt.bar(ind[1], MAC_PV[1],width[1], color = 'steelblue',edgecolor = 'black',label='Category 2')
plt.bar(ind[2], MAC_PV[2],width[2], color = 'navy',edgecolor = 'black',label='Category 1')
plt.xlabel('$tCO_2e$ yearly savings')
plt.ylabel('$£/tCO_2e$')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.title('MAC curves for PV vs Area')

