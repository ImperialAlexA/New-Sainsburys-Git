#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 13:32:15 2018

@author: Alex
"""

import sqlite3
import numpy as np

#database_path = "C:\\Users\\GOBS\\Dropbox\\Uni\Other\\UROP - Salvador\\Niccolo_project\\Code\\Sainsburys.sqlite" # Path to database file
 # Path to database file


class tech:
    
    def __init__(self, tech_id):
        self.tech_id = tech_id
        try:    
            conn = sqlite3.connect(".\\Sainsburys.sqlite")
            cur = conn.cursor()
        except ValueError:
            print("Cannot connect to database")
        try:    
            cur.execute('''SELECT * FROM PV_Technologies WHERE id=?''', (tech_id,))
            dummy = cur.fetchall()
            self.PV_Nominal_Power = dummy[0][3]
            self.PV_eff = dummy[0][4]
            self.PV_capex =dummy[0][5]
            self.PV_Area = dummy[0][6]
            self.PV_Weight = dummy[0][7]
            self.PV_lifetime = dummy[0][8]
            self.PVtech_name = dummy[0][1]
            self.PVtech_price = dummy[0][2]
            
        except ValueError: 
            print("Cannot retrieve store data")
            

        conn.commit()
        
