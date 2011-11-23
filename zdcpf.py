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
solvers.options['show_progress']=False

def get_positive(x):
    return x*(x>0.)  #Possibly it has to be x>1e-10.

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
        self.cache=[]
        for i in range(len(files)):
            n=node(files[i],i,setup,files[i])
            self.cache=np.append(self.cache,n)

    def __getitem__(self,x):
        return self.cache[x]

    def setgammas(self,value):
        # to change a single node's gamma, just write XX.setgamma(yy)
        # 'value' can be a single number or a vector
        if np.size(value)==1:
            for i in self.cache: i.setgamma(value)
        elif np.size(value)!=np.size(self.cache):
            print "Wrong gamma vector size. ", np.size(value,0)," were  received, ",np.size(self.cache)," were expected."
        elif i in self.cache:
            i.setgamma(value[i.id])
        else: print "Something went horribly wrong! 'class: Nodes' section"

    def setalphas(self,value):
        # to change a single node's alpha, just write XX.setgamma(yy)
        # 'value' can be a single number or a vector
        if np.size(value)==1:
            for i in self.cache: i.setalpha(value)
        elif np.size(value)!=np.size(self.cache):
            print "Wrong gamma vector size. ", np.size(value,0)," were  received, ",np.size(self.cache)," were expected."
        elif i in self.cache:
            i.setalpha(value[i.id])
        else: print "Something went horribly wrong! 'class: Nodes' section"



    def add_colored_import(self, F, node_id=None, incidence_matrix='incidence.txt', lapse=np.size(F,1)):
	
        if lapse==None:
            lapse=self.cache[0].mismatch.shape[0]
    	
        if type(incidence_matrix)==str:
            K = np.genfromtxt(incidence_matrix,delimiter=",",dtype='d') #Incidence matrix.
            K = K[1:].transpose()[1:].transpose() #Remove dummy row/column.
        else:
            K = incidence_matrix
    			
        for t in arange(lapse):
    	
            export_ = array([self.cache[i].getexport()[t] for i in arange(len(self.cache))])
            import_ = array([self.cache[i].getimport()[t] for i in arange(len(self.cache))])
    	
            FF, C = get_colored_flow(F.transpose()[t], copy(export_), incidence_matrix=K)
    	
            CC = C*kron(ones((K.shape[0],1)),import_)
    	
            #Update Node(s)
            if node_id == None:
                for node_id_ in arange(len(self.cache)):
                    self.cache[node_id_].set_colored_import_i(t,CC.transpose()[node_id_])
            else:
                self.cache[node_id].set_colored_import_i(t,CC.transpose()[node_id])

def get_colored_flow(flow, export, incidence_matrix='incidence.txt'):
    """flow: vector of flows at time t. export: vector of export at each node at time t."""
    if type(incidence_matrix)==str:
        K = np.genfromtxt(incidence_matrix,delimiter=",",dtype='d') #Incidence matrix.
        K = K[1:].transpose()[1:].transpose() #Remove dummy row/column.
    else:
        K = incidence_matrix
    Kf = K*kron(ones((K.shape[0],1)),-flow) #Modified incidence matrix that has positive values for the flow into a node.
    FF = array(mat(abs(K))*mat(Kf).transpose()) #Flow matrix with positive values indicating the positive flow into a node.
    FF = get_positive(floor(FF))
        #"Column sum" = 1
    for i in arange(FF.shape[1]):
        sum_ = (FF.transpose()[i].sum() + export[i])
        FF.transpose()[i] = FF.transpose()[i]/sum_
        export[i] = export[i]/sum_
    	
    	#Calculate color matrix	
    try:	
        C = -mat(diag(export))*inv(mat(FF)-mat(eye(FF.shape[0])))	
    except LinAlgError:
        print "Error (dfkln387c): Singular matrix"
        print mat(FF)-mat(eye(FF.shape[0]))
    
        C = zeros(FF.shape)

    return array(FF), array(C)   

def dcpowerflow(P,q,G,h,A,b):
    sol=solvers.qp(P,q,G,h,A,b)
    return sol['x']

def generatemat(incidence,constraints,copper):
    K=np.genfromtxt(incidence,delimiter=",",dtype='d')
    Nnodes=np.size(K,0)
    Nlinks=np.size(K,1)
    # These numbers include the dummy node and link
    # With this info, we create the P matrix, sized
    P1=np.eye(Nlinks+2*Nnodes)  # because a row is needed for each flow, and two for each node
    P=conc((P1[:Nlinks],P1[-2*Nnodes:]*1e-6))  # and the bal/cur part has dif. coeffs
    # Then we make the q vector, whose values will be changed all the time
    q=np.zeros(Nlinks+2*Nnodes)  # q has the same size and structure as the solution 'x'
    # Then we build the equality constraint matrix A
    # The stucture is more or less [ K | -I | I ]
    A1=conc((K,-np.eye(Nnodes)),axis=1)
    A=conc((A1,np.eye(Nnodes)),axis=1)
    # See documentation for why first row is cut
    A=np.delete(A,np.s_[0],axis=0)
    # b vector will be defined by the mismatches, in MAIN
    # Finally, the inequality matrix and vector, G and h.
    # Refer to doc to understand what the hell I'm doing, as I build G...
    g1=np.eye(Nlinks)
    G1=g1
    for i in range(Nlinks-1):
        i+=i
        G1=np.insert(G1,i+1,-G1[i],axis=0)
    G1=conc((G1,-G1[-1:]))
    # to model copper plate, we forget about the effect of G matrix on the flows
    if copper == 1:
        G1*=0
    # G1 is ready, now we make G2
    G2=np.zeros((2*Nlinks,2*Nnodes))
    # G3 is built as [ 0 | -I | 0 ]
    g3=conc((np.zeros((Nnodes,Nlinks)),-np.eye(Nnodes)),axis=1)
    G3=conc((g3,np.zeros((Nnodes,Nnodes))),axis=1)
    g4=np.eye(Nnodes)
    G4=g4
    for i in range(Nnodes-1):
        i+=i
        G4=np.insert(G4,i+1,-G4[i],axis=0)
    G4=conc((G4,-G4[-1:]))
    G5=conc((np.zeros((2*Nnodes,Nlinks+Nnodes)),G4),axis=1)
    G=conc((G1,G2),axis=1)
    G=conc((G,G3))
    G=conc((G,G5))
    # That was crazy! Now, the h vector is partly made from the constraints.txt file
    h=np.genfromtxt(constraints,dtype='d')
    # but also added some extra zeros for the rest of the constraints.
    h=np.append(h,np.zeros(3*Nnodes))
    # And that's it!
    return P,q,G,h,A

def runtimeseries(N,F,P,q,G,h,A,coop,lapse):
    if lapse==None:
        lapse=Nodes[0].mismatch.shape[0]
    Nlinks=np.size(F,0)
    Nnodes=np.size(A,0)
    start=time()
    b=matrix([[0,0,0,0,0]],tc='d')
    P_b=P[Nlinks+2:Nlinks+Nnodes+2,:]*1e6
    for t in range(lapse):
        for i in N:
            b[i.id]=i.mismatch[t]
            # from default, both curtailment and balancing have a minimum of 0.
            # in order to prevent export of curtailment, max curtailment is set to b
            h[2*Nlinks+Nnodes+5+2*i.id]=0
            if b[i.id]>0:
                h[2*Nlinks+Nnodes+5+2*i.id]=b[i.id]
        # then, we set the values of q_b and q_r for bal and cur, according to doc.
        # for Gorm's inequalities, we need f,L,delta
        f=P[0,0]
        L=Nnodes-1
        d=np.array(b)
        excess=np.dot(d.T,d>0)[0][0]
        deficit=abs(np.dot(d.T,d<0)[0][0])
        delta=min(excess,deficit)
        q_r=L*f*2*delta*0.5
        q_b=L*f*2*delta+q_r*(1.5)
        q[Nlinks+2:Nlinks+Nnodes+2]=q_b
        q[Nlinks+Nnodes+2:]=q_r
        if coop==1:
            P[Nlinks+2:Nlinks+Nnodes+2,:]=P_b*L*f*deficit*.99
        opt=dcpowerflow(P,q,G,h,A,b)   ########### Save relevant solution as flows
        for j in range(Nlinks):
            F[j][t]=opt[j+1]           
        for k in N:                ########### Save balancing at each node
            k.balancing[t]=opt[2+Nlinks+k.id]
            k.curtailment[t]=opt[3+Nlinks+Nnodes+k.id]  
        end=time()
        if (np.mod(t,547)==0) and t>0:
            print "Elapsed time is ",round(end-start)," seconds. t = ",t," out of ",lapse
            sys.stdout.flush()
    for i in N:
        i.baltot.append(sum(i.balancing))
        i.curtot.append(sum(i.curtailment))
    end=time()
    sys.stdout.flush()
    print "Calculation took ",round(end-start)," seconds."
    return N,F

def zdcpf(N,incidence='incidence.txt',constraints='constraints.txt',setupfile='setupnodes.txt',coop=0,copper=0,lapse=70128):
    P,q,G,h,A = generatemat(incidence,constraints,copper)
    Nnodes=np.size(np.genfromtxt(incidence,delimiter=','),0)-1
    Nlinks=np.size(np.genfromtxt(incidence,delimiter=','),1)-1
    F=np.zeros((Nlinks,lapse))
    P=matrix(P,tc='d')
    q=matrix(q,tc='d')
    G=matrix(G,tc='d')
    h=matrix(h,tc='d')
    A=matrix(A,tc='d')
    N,F=runtimeseries(N,F,P,q,G,h,A,coop,lapse)
    return N,F
