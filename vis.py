# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 14:39:28 2019

@author: timur
"""
from matplotlib import pyplot as plt
import numpy as np
path="data/doublecheck/5km/"
distAll={"1km/":np.array([0,0,-1000]) ,
      "2km/":np.array([0,0,-2000]),
      "5km/":np.array([0,0,-5000]),
      "707mx707m/":np.array([707,0,-707]),
      "866mx500m":np.array([500,0,-866]),
      "500mx866m":np.array([866,0,-500])
      }
dist1={"1km/":np.array([0,0,-1000]) ,
      "866mx500m/":np.array([500,0,-866]),
      "707mx707m/":np.array([707,0,-707]),
      "500mx866m/":np.array([866,0,-500])
      }
dist2={"1km/":np.array([0,0,-1000]) ,
      "2km/":np.array([0,0,-2000]),
      "5km/":np.array([0,0,-5000])
      }
def show(dist):
    for key in dist:
      try:  
        #distVal=dist[key]
        time=np.fromfile(path+key+"inpTime.txt",sep=" ")
        val=np.fromfile(path+key+"inpVal.txt",sep=" ")
        plt.plot(time[:-1],val,label=key[:-1])
        plt.legend(loc='best')
        plt.title("Electric field module, V/m")
        plt.xlabel("time,ns")
        plt.ylabel("V/m")
        #plt.show()
      except FileNotFoundError:
        print("FileNotFoundError")
    plt.show()
    plt.clf() 
    for key in dist:
      try:
        val=np.fromfile(path+key+"vals.txt",sep=" ")
        freq=np.fromfile(path+key+"freqs.txt",sep=" ")
        plt.plot(freq,val,label=key[:-1])
        plt.xscale("log")
        plt.legend(loc='best')
        plt.title("Electric field spectrum, V/m")
        plt.xlabel("frequency,Hz")
        plt.ylabel("V/m")
      except FileNotFoundError:
        print("FileNotFoundError")
    plt.show()
    plt.clf() 
show(dist2)

