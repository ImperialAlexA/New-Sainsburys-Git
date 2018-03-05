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
from gams import GamsWorkspace



############################################
### Problem input ###    
############################################

## define problem granularity ###
time_window = 20
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
GAMS_model = "Strategic.gms"
ws = GamsWorkspace()
db =ws.add_database()
i = db.add_set("i",1,"")






############################################
### run GAMS model and retrieve results ###    
############################################



############################################
### plot stuff ###    
############################################

