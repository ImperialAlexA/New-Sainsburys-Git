# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 10:51:19 2018

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
import decompose_fun_2 as decfun
import Common.classStore as st
import xlsxwriter
import subprocess
import classPV_CHP as PC
import os



database_path = "Sainsburys.sqlite"
conn = sqlite3.connect(database_path)
cur = conn.cursor()

cur.execute('''SELECT Stores_id FROM Demand_Check Where Ele= {vn1} and Gas= {vn2}'''.format(vn1=1,vn2=1))
Index = cur.fetchall()
Store_id_range = np.array([elt[0] for elt in Index],dtype=np.float64)
conn.close()

Store_id_range = np.delete(Store_id_range,44) # =store 2017 not included because of errors

R2_OPEX =[]
rel_err_OPEX =[]
R2_carbon =[]
rel_err_carbon =[]
for id_store in Store_id_range:
    
    a=PC.PV_CHP(id_store).error()
    R2_OPEX.append(a[1][2])
    rel_err_OPEX.append(a[1][1])
    R2_carbon.append(a[2][2])
    rel_err_carbon.append(a[2][1])
    
R2_OPEX_average= np.average(np.average(R2_OPEX,axis=1))
R2_OPEX_std= np.std(np.average(R2_OPEX,axis=1))
rel_err_OPEX_average = np.average(np.delete(np.average(rel_err_OPEX,axis=1),0,1))
rel_err_OPEX_std= np.std(np.delete(np.average(rel_err_OPEX,axis=1),0,1))  

R2_carbon_average= np.average(np.average(R2_carbon,axis=1))
R2_carbon_std= np.std(np.average(R2_carbon,axis=1))
rel_err_carbon_average = np.average(np.delete(np.average(rel_err_carbon,axis=1),0,1))
rel_err_carbon_std= np.std(np.delete(np.average(rel_err_carbon,axis=1),0,1))  

id_store = 26
a=PC.PV_CHP(id_store).error()
Carbon_error = np.average(a[1],axis=1)
Carbon_err_average = np.average(Carbon_error,axis=1)

 
#residual,rel_err,R2