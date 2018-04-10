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


database_path = "Sainsburys.sqlite"
conn = sqlite3.connect(database_path)
cur = conn.cursor()

cur.execute('''SELECT Stores_id FROM Demand_Check Where Ele= {vn1} and Gas= {vn2}'''.format(vn1=1,vn2=1))
Index = cur.fetchall()
Store_id_range = np.array([elt[0] for elt in Index],dtype=np.float64)


Store_id_range = np.delete(Store_id_range,44) # =store 2017 not included because of errors

Gas = []
Ele =[]
carbon =[]
area = []

gas_CF = 0.18416
ele_CF =   0.35156 

Gas_total = []
Ele_total = []
BAU_op_cost = []

BAU_op_cost_test = []
BAU_op_cost_test2 =[]
#mod=[2.759,1.675,1,1])
for id_store in Store_id_range:
#    sol=BBC.CHPproblem(id_store).store
#    Gas_total.append(sum(BBC.CHPproblem(id_store).store.d_gas))
#    Ele_total.append(sum(BBC.CHPproblem(id_store).store.d_ele))
    BAU_op_cost.append(BBC.CHPproblem(id_store).SimpleOpti5NPV(mod=[2.759,1.675,1,1])[6])
#    BAU_op_cost_test.append(sum(sol.d_ele*sol.p_ele+sol.d_gas*sol.p_gas)/100)
#    BAU_op_cost_test2.append(pb.PVproblem(id_store).SimulatePVonAllRoof(1,1)[7])
Carbon = sum(Gas_total)*gas_CF+sum(Ele_total)*ele_CF
BAU_cost = sum(BAU_op_cost)
#BAU_cost_test=sum(BAU_op_cost_test)
#BAU_cost_test2=sum(BAU_op_cost_test2)


