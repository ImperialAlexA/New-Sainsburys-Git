# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:30:10 2017

@author: nl211
"""
#import paramiko
import sys
import os
import paramiko
import yaml
import time
import shutil




#if 'Inputs.gdx' in os.listdir("Z:\\"):
#    os.remove('Z:\\Inputs.gdx')

#item_folder_path = ".\\GAMS"
#item = "db.gdx"
#item2 = "CHP.gms"
#message='gridgams24.7.4 CHP.gms'  

class ClusterConnection:

    def __init__(self):
        self.problem_dir = ".\\GAMS"
        self.clusterdir='Z:'
        self.gdx_file='db.gdx'
        self.output_file='output.gdx'
        self.output_file_db='output.db'
    
    def SubmitGAMSJob(self, GAMS_file, escCluster = None):
        self.MoveToCluster(self.problem_dir, GAMS_file)
        self.MoveToCluster(self.problem_dir, self.gdx_file)
        if self.output_file in os.listdir(self.clusterdir):
            os.remove(self.clusterdir + "\\" + self.output_file) 
            
        message='gridgams24.7.4' + " " + GAMS_file
        self.SubmitMessage(message)
        
        if escCluster == 1:
            print("job submitted, returing..")
        else:
            while (self.output_file not in os.listdir(self.clusterdir)):
                time.sleep (5)
                print("Solving in cluster.. Zzz")
            print("Problem solved")
        
            if self.output_file in os.listdir(self.problem_dir):
                os.remove(self.problem_dir + "\\"+ self.output_file) 
            if self.output_file_db in os.listdir(self.problem_dir):
                os.remove(self.problem_dir + "\\"+ self.output_file_db)    
            shutil.move(self.clusterdir + "\\" + self.output_file, self.problem_dir) 
            shutil.move(self.clusterdir + "\\" + self.output_file_db, self.problem_dir) 
            
    
    def SubmitMessage(self, message):
    
        cred = yaml.load(open('cred.yml'))
        username = cred['user']['login']
        pwd = cred['user']['password']
        ip = cred['user']['ip']
        port = cred['user']['port']
    
        ssh=paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ip,port,username,pwd)
        #print("message submitted:", message)
        stdin,stdout,stderr=ssh.exec_command(message)
    
        outlines=stdout.readlines()
        resp=''.join(outlines)
     
        print(resp)
        
        
    def MoveToCluster(self, item_folder_path, item):
        if item in os.listdir(self.clusterdir):
            os.remove(self.clusterdir + item)   
        shutil.copy(item_folder_path + "\\" + item, self.clusterdir)  
        
        

        

  
#MoveToCluster(item_folder_path, item)
#MoveToCluster(item_folder_path, item2)
#SubmitGAMSJob(message)  


#unit test for SSH submission, requires a working gams model called model.gms in your gams cluster directory
#the second test with 'message2' will move trnsportgdx1.gms and pythontest.py (currently in git repo) into Z:\ folder
#it will then create a 'testouput.gdx' file
#most of message2 contains command lines to set the correct python environment to be able to run python code in the linux cluster
#message 3 to run a standard optimisation of SSL TSO (requires insertion of Input.xlsx in cluster folder) however does not work because gdxrw not working yet in cluster. to be investigated

#if __name__ == "__main__":
#    
#    message1='gridgams24.7.4 model.gms'
#    
#    #SubmitGAMSJob(message1)
#    
#    message2='module load anaconda/2.4.1 ; module load gams/24.9.1 ; setenv PYTHONPATH ${GAMSHOME}/apifiles/Python/api ; python pythontest.py'
#    
#    clusterdir='Z:\\'
    
  #  shutil.copy("CHP.gms", clusterdir)
   # shutil.copy("pythontest.py", clusterdir)
  

    #SubmitGAMSJob(message2)
    
    
    #message3='module load anaconda/2.4.1 ; module load gams/24.9.1 ; setenv PYTHONPATH ${GAMSHOME}/apifiles/Python/api ; python gams_submission_ssl.py'
   
    
    
    #SubmitGAMSJob(message3)
#    
#    
#import time
#import shutil
#import os
#from gams import *
#l=os.getcwd()
#ws = GamsWorkspace(l)
#
#t1 = ws.add_job_from_file("Inputs.gms")
#t1.run()
#
#
#if 'Inputs.gdx' in os.listdir("Z:\\"):
#    os.remove('Z:\\Inputs.gdx')
#
#shutil.move("Inputs.gdx", "Z:")
#message='gridgams24.7.4 model.gms'
#
#
#if 'Outputs.gdx' in os.listdir("Z:\\"):
#    os.remove('Z:\\Outputs.gdx')
#    
#if 'mincost.gdx' in os.listdir('Z:\\'):
#    os.remove('Z:\\mincost.gdx')
#    os.remove('Z:\\baseline.gdx')
#    
#SubmitGAMSJob(message)
#
#
#while ('mincost.gdx' not in os.listdir("Z:\\")):
#    time.sleep (5)
#
#    
#if 'mincost.gdx' in os.listdir(l):
#    os.remove('mincost.gdx')
#    os.remove('baseline.gdx')
#    
#shutil.move("Z:\\mincost.gdx", l)    
#shutil.move("Z:\\baseline.gdx", l)  
#
#    
#t2 = ws.add_job_from_file("Outputs.gms")
#t2.run()
#
#print('optimisation results ready')   