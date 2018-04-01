# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 21:43:37 2018

@author: Anatole
"""

import gdxpds
import pandas as pd


gdx_file = 'C:\\Users\\Anatole\\Documents\\GitHub\\New-Sainsburys-Git\\out.gdx'
dataframes = gdxpds.to_dataframes(gdx_file)
for symbol_name, df in dataframes.items():
    print("Doing work with {}.".format(symbol_name))