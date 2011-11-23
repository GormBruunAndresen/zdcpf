#! /usr/bin/env python
from pylab import *
from scipy import *
import scipy.optimize as optimize
from numpy import concatenate as conc
import numpy as np
from cvxopt import matrix,solvers
from time import time
import os
from copy import deepcopy

colors_countries = ['#00A0B0','#6A4A3C','#CC333F','#EB6841','#EDC951'] #Ocean Five from COLOURlovers.

class node:
    def __init__(self,fileName,ID,setup,name):
        self.id = ID
        data = map(np.double,np.load(fileName))
        self.gamma = float(setup[ID][0])
        self.alpha = float(setup[ID][1])  #Alpha should be expanded to a vector.  completalpha() can be applied in update()
        self.load = data[0]
        self.nhours = len(self.load)
        self.normwind = data[1]
        self.normsolar = data[2]
        self.mean = np.mean(self.load)
        self.balancing = np.zeros(self.nhours)
        self.curtailment = np.zeros(self.nhours)
        self.baltot = []
        self.curtot = []
        self.name = name
        self.mismatch = None
        self.colored_import = None #Set using self.set_colored_i_import()
		
        self._update_()
    
    def _update_(self):
        self.mismatch=(self.getwind()+self.getsolar())-self.load
    
    def getimport(self):
        """Returns import power time series in units of MW."""
        return get_positive(get_positive(-self.mismatch) - self.balancing) #Balancing is exported if it exceeds the local residual load.
        
    def getexport(self):	
        """Returns export power time series in units of MW."""
        return get_positive(self.mismatch) - self.curtailment + get_positive(self.balancing - get_positive(-self.mismatch))
	
    def getlocalRES(self):
        """Returns the local use of RES power time series in units of MW."""
        return self.getwind() + self.getsolar() - self.curtailment  - self.getexport()
		
    def getlocalBalancing(self):
        """Returns the local use of balancing power time series in units of MW."""
        return get_positive(-self.mismatch) - self.getimport()
		
    def getwind(self):
        """Returns wind power time series in units of MW."""
        return self.mean*self.gamma*self.alpha*self.normwind
	
    def getsolar(self):
        """Returns solar power time series in units of MW."""
        return self.mean*self.gamma*(1.-self.alpha)*self.normsolar
			
    def setgamma(self,gamma):
        self.gamma = gamma
        self._update_()
    
    def setalpha(self,alpha):
        self.alpha=alpha
        self._update_()

    def set_colored_import_i(self,i,colored_import_i):
        if self.colored_import == None:
            self.colored_import = zeros((len(colored_import_i),len(self.load)))
			
        self.colored_import.transpose()[i] = colored_import_i


class Nodes:
    def __init__(self,setupnodes='setupnodes.txt',files=['N.npy','S.npy','DKW.npy','DKE.npy','DN.npy']):
        setup=np.genfromtxt(setupnodes,delimiter=',')
        self=[]
        for i in range(len(files)):
            n=node(files[i],i,setup,files[i])
            self=np.append(self,n)


