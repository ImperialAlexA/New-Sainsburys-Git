# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:04:55 2018

@author: nl211
"""

import sqlite3
import Common.classStore as st
import Common.classTech as tc
import Solvers.classCHPProblemnew as pb
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd








store_index = 507   #Leicester North
p_id = pb.CHPproblem(store_index)


#default_initial_time = datetime.datetime(2016,1,1)
#default_final_time = datetime.datetime(2017,1,1)
#time_start= int((default_initial_time-datetime.datetime(1970,1,1)).total_seconds()/60/30)
#time_stop= int((default_final_time-datetime.datetime(1970,1,1)).total_seconds()/60/30)

p_id.store.getWeatherData(p_id.time_start, p_id.time_stop)
plt.plot(p_id.store.irr)