# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 00:09:53 2018

@author: lmaqelepo
"""
from scipy.optimize import minimize
import numpy as np
import datetime as dt
import knapsack


class Smartmeter(object):
    """
    This class represents a electricity smart meter object that monitors demand
    and supply simultaneously, that is capable of shedding load with customer 
    tier prioritization based on different tariff structures
    """
    def __init__(
                 self, ncustomers, customers_per_tier, price_per_tier, st,
                 life='10', salvage = '0'):
        """
        Initialize smart meters
            n_customers: int
                Number of customers, corresponds to number of meters
            customers_per_tier: list
                Number of customers per tier
            price_diff_tier: list
                Price of kWh of electricity per tier
        """
        if len(customers_per_tier) != len(price_per_tier):
            raise Exception('List lengths %d and %d must be equal' \
                            % (len(customers_per_tier), len(price_per_tier)))
        if np.sum(customers_per_tier) != ncustomers:
            raise Exception('Number of customers must match')
        self.ncustomers = int(ncustomers)
        self.cust_per_tier = [int(val) for val in customers_per_tier]
        self.prc_per_tier = [float(val) for val in price_per_tier]
        self.life = int(life)
        self.installdate = st
        self.installdates = [st]
        self.ET = st + dt.timedelta(days=365*self.life) + dt.timedelta(hours=23)
        self.salvage = float(salvage)
      
    def revenue(self, x, t):
        """
        Compute revenue based on demand and tarriff
        Params:
            x:- demand vector
            t:- tariffs vector
        Return:
            Revenue 
        """
        return np.sum(np.multiply(-np.array(x), t))

    def meet_load(self, timestamp, demand, supply):
        """
        Return met load that maximizes revenue based on tarriff tiers
        
        params:
            demand: array  
                vector of demand profiles
            supply: int or float
                available power to dispatch
        """
        if timestamp > self.ET:
            self.installdates.append(timestamp)
            self.ET = timestamp + dt.timedelta(days=365*self.life) + dt.timedelta(hours=23)
            
        weights = np.random.lognormal(size=self.ncustomers)
        demand = demand*weights/weights.sum()
        
        lims = [[0,d] for d in demand]
        pricepertier = []
        for p, c in zip(self.prc_per_tier, self.cust_per_tier):
            pricepertier += list(np.repeat(p, c))
        gues = np.zeros(len(demand))+0.0001
        con = lambda y: -np.sum(y) + supply
        cons = {'type':'ineq','fun':con}
        sol = minimize(self.revenue, gues, args=(pricepertier), 
                       bounds=lims, constraints=cons, tol=0.01)
        
        diff = demand - sol.x
        x = np.where(diff < 0.001, demand, 0)
        return x.sum()
    
    def mload(self, timestamp, demand, supply):
        if timestamp > self.ET:
            self.installdates.append(timestamp)
            self.ET = timestamp + dt.timedelta(days=365*self.life) + dt.timedelta(hours=23)
            
        weights = np.random.lognormal(size=self.ncustomers)
        demand = demand*weights/weights.sum()
        pricepertier = []
        for p, c in zip(self.prc_per_tier, self.cust_per_tier):
            pricepertier += list(np.repeat(p, c))
            
        idxs = knapsack.knapsack(demand, pricepertier).solve(supply)[1]
        return demand[idxs].sum()
        
    
if __name__=='__main__':
    
    start = dt.datetime(2021, 1, 1)
    Ncusts = 12
    Custpertier = [2, 5, 5]
    pricepertier = [0.3, 0.15, 0.2]
    totalload = 100
    supply = 65
    # dist = np.random.uniform(size=10)
    # dist = dist/dist.sum()
    demand = totalload#*dist
    
    smartmeter = Smartmeter(Ncusts, Custpertier, pricepertier, start)
    
    metload = smartmeter.meet_load(start, demand, supply)
    