# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:00:00 2018

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

database_path = "results.sqlite"
conn = sqlite3.connect(database_path)
cur = conn.cursor()

cur.execute('''SELECT * FROM s''')
store = cur.fetchall()
store_range = np.array([elt[0] for elt in store],dtype=np.int)


Size= []
for i,store in enumerate(store_range):
    print(store)
    print(i)
    cur.execute("SELECT t, level FROM x WHERE tech='PV' and s={vn1} ".format(vn1=store))
    Index = cur.fetchall()
    t = np.array([elt[0] for elt in Index],dtype=np.float64)
    Size.append(np.array([elt[1] for elt in Index],dtype=np.float64))
    plt.plot(t,Size[i],label=store)

plt.legend()

pb.PVproblem(51).Max_panel_number(1)

    

