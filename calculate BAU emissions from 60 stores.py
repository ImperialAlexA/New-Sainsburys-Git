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
#for id_store in Store_id_range:
#    cur.execute(
#        '''SELECT GD2016, ED2016, Carbon, Area FROM Stores Wh2ere id= {vn1}'''.format(
#            vn1=id_store))
#    Index = cur.fetchall()
#    
#    Gas.append(np.array([elt[0] for elt in Index])) #kWh
#    Ele.append(np.array([elt[1] for elt in Index])) #kWh
#    carbon.append(np.array([elt[2] for elt in Index]))
#    area.append(np.array([elt[3] for elt in Index]))

Gas = []
Ele =[]    
for id_store in Store_id_range:
    cur.execute(
        '''SELECT Ele, Gas FROM Demand Where Stores_id= {vn1}'''.format(
            vn1=id_store))
    Index = cur.fetchall()
    Gas.append(np.array([elt[0] for elt in Index])) #kWh
    Ele.append(np.array([elt[1] for elt in Index])) #kWh
    
conn.close()
