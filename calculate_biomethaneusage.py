# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:00:25 2018

@author: Anatole
"""

import Solvers.classPVProblem as pb
import Solvers.classCHPProblemnew as BBC
import sqlite3
import numpy as np
from scipy.optimize import curve_fit
import datetime
from sklearn.metrics import mean_absolute_error, r2_score
import decompose_fun_2 as decfun
import Common.classStore as st
import re
import matplotlib.pyplot as plt

biometh_CF= 0.00039546
gas_CF = 0.18416
ele_CF = 0.35156 

database_path = "Sainsburys.sqlite"
conn = sqlite3.connect(database_path)
cur = conn.cursor()
conn.commit()
cur.execute('''SELECT * FROM Technologies''')
dummy_CHP = cur.fetchall()


sol=BBC.CHPproblem(51)
BAU_gas = []
BAU_ele =[]
biomethane_usage = []
BAU_carbon = []
savings_range =[]
CHP_tech_size_range =[]
for tech in range(1,21):
    if tech ==20:
        CHP_tech_size = [0]
        savings = 0
    else:
        
        CHP_tech_size =(list(map(int, re.findall('\d+', dummy_CHP[tech][1]))))
        savings = sol.SimpleOpti5NPV(tech_range=[tech])[5][0]-(sol.SimpleOpti5NPV(tech_range=[tech])[5][3]*gas_CF/1000+sol.SimpleOpti5NPV(tech_range=[tech])[5][4]*ele_CF/1000)
    CHP_tech_size_range.append(CHP_tech_size)
    
    biomethane_usage.append((sol.SimpleOpti5NPV(tech_range=[tech])[5][3]*gas_CF+sol.SimpleOpti5NPV(tech_range=[tech])[5][4]*ele_CF)/1000)
    
    BAU_gas= np.sum(sol.store.d_gas)
    BAU_ele= np.sum(sol.store.d_ele)
    BAU_carbon.append(sol.SimpleOpti5NPV(tech_range=[tech])[5][0])
    
    savings_range.append(savings)



#plt.plot(CHP_tech_size_range,savings_range, '.')

plt.plot(savings_range,'.')

BBC.CHPproblem(51,NG=True).SimpleOpti5NPV(tech_range= [18])[5][2]
