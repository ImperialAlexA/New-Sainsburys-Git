# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:28:47 2018

@author: nl211
"""

import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def decompose(X_input,Y_input,spl):
    n_iter = 15
    dim = X_input.shape[1]
    #conduct scaling 
    X = (X_input - X_input.min(axis=0))/(X_input.max(axis=0) - X_input.min(axis=0));
    Y = (Y_input - Y_input.min(axis=0))/(Y_input.max(axis=0) - Y_input.min(axis=0));
    ##check for consistency of data input:
    #to do    
    d = spl**dim
    p_best = 0*np.ones((dim,d))
    intercept_best = 0*np.ones((1,d))
    lb_best = 0*np.ones((dim,d))
    ub_best = 0*np.ones((dim,d))    
    res_best_history = 0*np.ones((n_iter,1))
    
    res_best = 10000
    fault = 0
    for t in range(n_iter):
        try:            
            # generate randonmly the domain
            l = np.random.rand(dim,spl-1)
            l = np.sort(l,axis=1)
            lb0 = np.concatenate((0*np.ones((dim,1)),l), axis=1)
            ub0 = np.concatenate((l,np.ones((dim,1))), axis=1)
            
            
            #generate selection vector for domains
            arr_pos = np.ndarray(shape=(dim,spl**dim))
            for i in range(dim):
                u_v = np.repeat(np.arange(spl),spl**i)  
                arr_pos[i] = np.tile(u_v,(1,spl**(dim-i-1)))
            
            #fit a linear model in any domain  
            p = 0*np.ones((dim,d))
            intercept = 0*np.ones((1,d))
            lb = 0*np.ones((dim,d))
            ub = 0*np.ones((dim,d))    
            res = 0*np.ones((1,d))           
            regr = linear_model.LinearRegression()
            for i in range(d):
                f1 = arr_pos[:,i].astype(int)
                f2 = np.arange(lb0.shape[0]).astype(int)
                lb[:,i] = lb0[f2,f1] 
                ub[:,i] = ub0[f2,f1] 
                lb_IO = X > lb[:,i]
                ub_IO = X < ub[:,i]    
                mask = np.logical_and(np.all(lb_IO, axis = 1), np.all(ub_IO, axis = 1))
                
                X0 = X[mask]
                Y0 = Y[mask]
                regr.fit(X0, Y0)
                p[:,i] = regr.coef_
                intercept[0,i] = regr.intercept_
                Y_fit = regr.predict(X0)
                res[0,i] = np.sum(np.power(Y0-Y_fit,2))
                resTot = np.sum(res)
        except:    
              fault = fault + 1
              resTot = 10000
              if fault > 5:
                  print("something is worng with the problem")
                  raise 
        #store parameters if good          
        if resTot < res_best:
            res_best = resTot
            p_best = p
            intercept_best =intercept
            lb_best = lb
            ub_best = ub   
        res_best_history[t] =  res_best  
    
    div = 0*np.ones((dim,1)); div[:,0] = (X_input.max(axis=0) - X_input.min(axis=0))
    p_best = np.divide(p_best*(Y_input.max(axis=0) - Y_input.min(axis=0)),div)
    intercept_best = intercept_best*(Y_input.max(axis=0) - Y_input.min(axis=0)) + Y_input.min(axis=0) - np.transpose(np.dot(np.transpose(p_best),X_input.min(axis=0)[:,None]))
    lb_best = np.multiply(lb_best,div) + X_input.min(axis=0)[:,None]    
    ub_best = np.multiply(ub_best,div) + X_input.min(axis=0)[:,None] 
    #interect and the otehr ub lb and res
    
    return(p_best, intercept_best, lb_best,ub_best,res_best_history)
              
              

if __name__ == "__main__":
    
    #to try
    dim = 2
    spl = 2
    n_iter = 15    
    n = 4000
    X_input = np.random.rand(n,dim)  -0.5
    #Y_input = 2*X_input[:,0]*X_input[:,1] + 3  + np.random.rand(n)
    Y_input = abs(2*X_input[:,0])+abs(X_input[:,1]) #+ 3  + np.random.rand(n)
    
    [p_best, intercept_best, lb_best, ub_best, res_best_history] = decompose(X_input,Y_input,spl)
#    plt.plot(res_best_history)
    print("coeff:", p_best)
    print("intercept:", intercept_best)
    for i in range(p_best.shape[1]):
                lb_IO = X_input > lb_best[:,i]
                ub_IO = X_input < ub_best[:,i]    
                mask = np.logical_and(np.all(lb_IO, axis = 1), np.all(ub_IO, axis = 1))
                X0 = X_input[mask]
                print(X0.shape)
                Y_fit = intercept_best[:,i] +  np.dot(X0,p_best[:,i][:,None])
                if i == 0:
                   X_tot = X0
                   Y_tot = Y_fit
                else:    
                    X_tot= np.append(X_tot,X0, axis =0)
                    Y_tot=np.append(Y_tot,Y_fit)
#                fig = plt.figure()
#                ax = Axes3D(fig)
#                ax.scatter(X_input[:,0], X_input[:,1], Y_input,s = 1)
#                ax.scatter(X_tot[:,0], X_tot[:,1], Y_tot, c = 'r', s = 1)    


    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X_input[:,0], X_input[:,1], Y_input,s = 1)
    ax.scatter(X_tot[:,0], X_tot[:,1], Y_tot, c = 'r', s = 1)
   

