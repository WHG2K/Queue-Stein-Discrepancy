# -*- coding: utf-8 -*-
"""
Created on Wed May 28 10:52:09 2025

@author: Anton Braverman
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def nhpp_next(t, lambda_t,lmax):
    """generate the next arrival following time t from a non-homogeneous Poisson process with 
    time varying arrival rate lambda_t using thinning"""
    next_t = t
    while True:
        next_t += np.random.exponential(1/lmax)
        if lambda_t(next_t)/lmax > np.random.uniform():
            return next_t    

def mm1_sim(T,lambda_t,mu_t,lmax=100,mumax=100,init=0):
    """generate a trajectory from an mm1 queue on [0,T]. Return a list of event times 
    along with another list of queue lengths at these event times"""
    t = 0; Q = [init]; times = [0];
    narrival = nhpp_next(t,lambda_t,lmax)
    nservice = nhpp_next(t,mu_t,mumax)
    while t < T:
        t = min(narrival, nservice) 
        times += [t]
        if narrival<nservice:
            if Q[-1]==0:
                nservice = nhpp_next(t,mu_t,mumax)
            narrival = nhpp_next(t,lambda_t,lmax)
            Q += [Q[-1]+1]
        else: 
            if Q[-1]>0:
                Q += [Q[-1]-1]
            else:
                times.pop()
            nservice =nhpp_next(t,mu_t,mumax) 
    return times,Q
             

if __name__ == "__main__":
    T=1000000
    lmax = .1
    mumax=.1
    lambda_t = lambda t: .09
    mu_t = lambda t: .1 
    times, Q = mm1_sim(T,lambda_t,mu_t, lmax,mumax)
    
    ls = np.linspace(0,T,T+1)
    arglist = np.searchsorted(times,ls,side='right') -1
    Qvals = [Q[i] for i in arglist]
    counts = Counter(Qvals)
    total = sum(counts.values())
    values, probs = zip(*sorted((k, v / total) for k, v in counts.items()))
    plt.bar(values, probs)
    #plt.hist(Qvals,bins=75)
    #plt.plot(Qvals)
    