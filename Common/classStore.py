# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 13:09:32 2017

@author: nl211
"""
import sqlite3
import numpy as np


#database_path = "C:\\Users\\GOBS\\Dropbox\\Uni\Other\\UROP - Salvador\\Niccolo_project\\Code\\Sainsburys.sqlite" # Path to database file
database_path = "Sainsburys.sqlite" # Path to database file


# auxiliary function
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]   

#interpolate data from database to cover all the data range requested.     
def interpolate(times, values, time_start, time_stop):
    Time_vec = np.linspace(time_start , time_stop-1, num =-time_start+time_stop )     
    Temp = np.empty(len(Time_vec))
    Temp[:] = np.NaN
    #populate with known values
    mask = np.in1d(Time_vec,times)
    Temp[mask] = values
    #populate with interpolated values
    nans, x= nan_helper(Temp)
    Temp[nans]= np.interp(x(nans), x(~nans), Temp[~nans])
    #quality control
    if len(x(nans)) > len(Temp)*0.6:
                 print("nans:", len(x(nans)), "not nans:", len(x(~nans)), "... interpolated data are not reliable")  
                 raise ValueError   
    assert len(Temp) == len(Time_vec)
    return Time_vec, Temp

class store:
    
    def __init__(self, store_id):
        self.store_id = store_id
        self.HH_open = 14
        self.HH_close = 47
        try:    
            conn = sqlite3.connect(database_path)
            cur = conn.cursor()
        except ValueError:
            print("Cannot connect to database")
        try:    
            cur.execute("SELECT DNO, name FROM Stores WHERE id = ?", (store_id,))
            dummy = cur.fetchall()
            self.DNO = dummy[0][0]    
            self.Voltage = 1   #all stores are assumed to have low voltage sub connection.
            self.name = dummy[0][1]  
        except ValueError: 
            print("Cannot retrieve store data")            
#        try:    
#            cur.execute("SELECT PostCode, Lat, Lon, Area FROM Stores WHERE id = ?", (store_id,))
#            dummy = cur.fetchall()
#            self.postcode = dummy[0][0]
#            self.lat = dummy[0][1]
#            self.lon = dummy[0][2]
#            self.area = dummy[0][3]
#        except ValueError: 
#            pass  
        try:    
            cur.execute("SELECT HH_WD_open,  HH_WD_close,    HH_Sat_open,    HH_Sat_close,    HH_Sun_open,   HH_Sun_close FROM Stores WHERE id = ?", (store_id,))
            dummy = cur.fetchall()
            self.HH_WD_open = dummy[0][0]    
            self.HH_WD_close = dummy[0][1]
            self.HH_Sat_open = dummy[0][2]
            self.HH_Sat_close = dummy[0][3]
            self.HH_Sun_open = dummy[0][4]
            self.HH_Sun_close = dummy[0][5]
        except ValueError: 
            pass     
        conn.commit()    
        
        

     #get demands from time_start included to time_start excluded every HH. 
     #time_start and time-stop have to be supplied as seconds from the epoque divided 1800 (HH integer).    
    def getSimpleDemand(self,time_start,time_stop, Utilities = [1,1,0]):
        
        Ele = Utilities[0]
        Gas = Utilities[1]
        Ref = Utilities[2]        
        conn = sqlite3.connect(database_path)
        cur = conn.cursor()
        
        if Ele== 1:
            cur.execute('''SELECT Gas FROM Demand_Check Where Stores_id= ?''', (self.store_id,))
            dummy= cur.fetchone()
            try:
                if dummy[0] is not 1:
                    raise  TypeError                 
                else:
                     cur.execute('''SELECT Time_id, Gas FROM Demand Where Stores_id= ? AND Time_id > ? AND Time_id < ? ''', (self.store_id, time_start-1, time_stop))
                     RawData = cur.fetchall()
                     timeControl_start = RawData[0][0]
                     timeControl_stop = RawData[-1][0]
                     if timeControl_start == time_start and timeControl_stop  == time_stop - 1:
                         d_gas = np.array([elt[1] for elt in RawData])   
                         timestamp = np.array([elt[0] for elt in RawData])    
                         self.timestamp, self.d_gas = interpolate(timestamp, d_gas, time_start, time_stop) 
                     else:
                         print("time_id requested out of range. Gas range:", timeControl_start,  timeControl_stop, "you put:",  time_start, time_stop)  
                         raise ValueError                            
            except TypeError:
                print("We don't have the gas demand")
        
        if Gas== 1:
            cur.execute('''SELECT Ele FROM Demand_Check Where Stores_id= ?''', (self.store_id,))
            dummy= cur.fetchone()
            try:
                if dummy[0] is not 1:
                    raise  TypeError                 
                else:
                     cur.execute('''SELECT Time_id, Ele FROM Demand Where Stores_id= ? AND Time_id > ? AND Time_id < ? ''', (self.store_id, time_start-1, time_stop))
                     RawData = cur.fetchall()
                     timeControl_start = RawData[0][0]
                     timeControl_stop = RawData[-1][0]
                     if timeControl_start == time_start and timeControl_stop  == time_stop - 1:
                         d_ele = np.array([elt[1] for elt in RawData])  
                         timestamp = np.array([elt[0] for elt in RawData]) 
                         self.timestamp, self.d_ele = interpolate(timestamp, d_ele, time_start, time_stop)                      
                     else:
                         print("time_id requested out of range. Electricity range:", timeControl_start,  timeControl_stop, "you put:",  time_start, time_stop)  
                         raise ValueError   
            except TypeError:
                print("We don't have the electricity demand")                 
            conn.commit() 
        
        if Ref == 1:
            cur.execute('''SELECT Ref FROM Demand_Check Where Stores_id= ?''', (self.store_id,))
            dummy= cur.fetchone()
            try:
                if dummy[0] is not 1:
                    raise  TypeError                 
                else:
                     cur.execute('''SELECT Time_id, Ref FROM Demand Where Stores_id= ? AND Time_id > ? AND Time_id < ? ''', (self.store_id, time_start-1, time_stop))
                     RawData = cur.fetchall()
                     timeControl_start = RawData[0][0]
                     timeControl_stop = RawData[-1][0]
                     if timeControl_start == time_start and timeControl_stop  == time_stop - 1:
                         d_ref = np.array([elt[1] for elt in RawData])
                         timestamp = np.array([elt[0] for elt in RawData]) 
                         self.timestamp, self.d_ref = interpolate(timestamp, d_ref, time_start, time_stop) 
                     else:
                         print("time_id requested out of range. Refrigeration range:", timeControl_start,  timeControl_stop, "you put:",  time_start, time_stop)  
                         raise ValueError   
            except TypeError:
                print("We don't have the Refrigeration demand")          
                
            conn.commit() 
          ##get carbon factor (utilising 2016)
        self.crc = 16.1 #£/tCo2
        self.cf_ele = 0.412 #kgCO2/kWh
        self.cf_gas = 0.184 #kgCO2/kWh
        self.cf_diesel = 0.244 #kgCO2/kWh    
            
       
     #get prices from time_start included to time_start excluded every HH. 
     #time_start and time-stop have to be supplied as seconds from the epoque divided 1800 (HH integer).          
    def getSimplePrice(self, time_start, time_stop, string_table):           
        conn = sqlite3.connect(database_path)
        cur = conn.cursor()
        try:
            sql_string = '''SELECT id, Ele, Gas, Ele_exp FROM {Table_name} Where DNO= ? AND Voltage = ? AND id > ? AND id < ?'''
            sql = sql_string.format(Table_name=string_table)
            cur.execute(sql, (self.DNO-9, self.Voltage, time_start-1, time_stop))        
            RawData = cur.fetchall()
            timeControl_start = RawData[0][0]
            timeControl_stop = RawData[-1][0]
            if timeControl_start == time_start and timeControl_stop  == time_stop - 1:
                 p_ele = np.array([elt[1] for elt in RawData]) 
                 p_gas = np.array([elt[2] for elt in RawData]) 
                 p_ele_exp = np.array([elt[3] for elt in RawData])  
                 timestamp = np.array([elt[0] for elt in RawData])      
                 ## interpolate missing values
                 self.timestamp, self.p_ele = interpolate(timestamp, p_ele, time_start, time_stop) 
                 self.timestamp, self.p_gas =  interpolate(timestamp, p_gas, time_start, time_stop) 
                 self.timestamp, self.p_ele_exp = interpolate(timestamp, p_ele_exp, time_start, time_stop)                     
            else:
                 print("time_id requested out of range. Price range:", timeControl_start,  timeControl_stop, "you put:",  time_start, time_stop)  
                 raise ValueError   
        except:
            print("An error occured. Possibly selected table doesn't exist. Please chose a valid Table")     
        conn.commit()
        
    def getWeatherData(self,time_start,time_stop):    
        #get data from the closest weather station.
        self.putMIDASstation()               
        #get weather data
        conn = sqlite3.connect(database_path)
        cur = conn.cursor() 
        found = 0
        i = 0
        while found == 0:
            MIDAStry = self.MIDASall[i]
            i = i+1
            cur.execute('''SELECT Time_id, Temp, dewpoint, wetb_temp, rltv_hum  FROM Weather WHERE Station_id = ? AND Time_id > ? AND Time_id < ?''', (MIDAStry, time_start-1, time_stop ))        
            Raw_data = cur.fetchall()
            Times = np.array([elt[0] for elt in Raw_data])
            if len(Times) < 0.40*(time_stop - time_start):  # not enough values, reiterate
                 pass
            else:
                self.MIDAS = MIDAStry
                found = 1
                #remove empty string value. not clean, should be changed somehow.
                TempData =np.array([elt[1] for elt in Raw_data])
                try:  ## check if empty string are present adn process
                    TempData[TempData == ' '] = 'NaN'
                    TempData.astype(np.float)
                except:
                    pass
        
                Time_vec, Temp = interpolate(Times, TempData, time_start, time_stop)
                self.temp = Temp         
                self.timestamp = Time_vec
        
        
    def putMIDASstation(self):  
        #get closest MIDAS stations.
        conn = sqlite3.connect(database_path)
        cur = conn.cursor()
        cur.execute('''SELECT * FROM MIDAS_stations''')
        Raw_data = cur.fetchall()
        Data = np.array([elt for elt in Raw_data])
        AvData = Data[Data[:,4]==1]
        id_MIDAS = np.array(AvData[:,0])
        Lat = np.array(AvData[:,2])
        Lon = np.array(AvData[:,3])
        d = np.square(Lat- self.lat)+np.square(Lon - self.lon)
        pos = np.argsort(d)
        self.MIDAS = id_MIDAS[pos[0]]     
        self.MIDASall = id_MIDAS[pos]
      
        
    def getTSODemand():      
         pass