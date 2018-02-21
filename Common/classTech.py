# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 13:09:32 2017

@author: nl211
"""
import sqlite3
import numpy as np

#database_path = "C:\\Users\\GOBS\\Dropbox\\Uni\Other\\UROP - Salvador\\Niccolo_project\\Code\\Sainsburys.sqlite" # Path to database file
database_path = ".\\Sainsburys.sqlite" # Path to database file


class tech:
    
    def __init__(self, tech_id):
        self.tech_id = tech_id
        try:    
            conn = sqlite3.connect(database_path)
            cur = conn.cursor()
        except ValueError:
            print("Cannot connect to database")
        try:    
            cur.execute('''SELECT * FROM Technologies WHERE id=?''', (tech_id,))
            dummy = cur.fetchall()
            self.a_fuel = dummy[0][3]
            self.b_fuel = dummy[0][4]
            self.a_el =dummy[0][5]
            self.b_el = dummy[0][6]
            self.a_th = dummy[0][7]
            self.b_th = dummy[0][8]
            self.psi_min = dummy[0][9]
            self.parasitic_load = dummy[0][10]
            self.mant_costs = dummy[0][11]
            self.lifetime = dummy[0][12]
            self.tech_name = dummy[0][1]
            self.tech_price = dummy[0][2]
            
        except ValueError: 
            print("Cannot retrieve store data")
            

        conn.commit()    
        
        
