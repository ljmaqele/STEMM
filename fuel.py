#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime as dt
import pandas as pd
import numpy as np

class FuelCost:
    
    def __init__(self, Pf0, Df, Vf, cf, tf, Vt, startdate, horizon):
        """
        

        Parameters
        ----------
        Pf0 : TYPE
            Initial fuel price per liter.
        Df : TYPE
            Annual percent fuel price drift.
        Vf : TYPE
            Annual fuel price volatility.
        cf : TYPE
            Diesel fuel consumption per hour in transit.
        tf : TYPE
            transit time in hours.
        Vt : TYPE
            Volume of transported fuel in a delivery.
        startdate : TYPE
            DESCRIPTION.
        horizon : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.Pf0 = Pf0 
        self.Df = Df
        self.Vf = Vf
        self.cf = cf
        self.tf = tf
        self.Vt = Vt
        self.startdate = startdate
        self.horizon = horizon
        
    def simfuelcost(self):
        pf = self.pfuel()
        return self.ptrans(pf)
    
    
    def pfuel(self):
        """ Evaluate fuel prices for all months """
        index = pd.date_range(start=self.startdate-dt.timedelta(hours=1), 
                              periods=12*self.horizon+1, freq='M')
        pf = np.zeros(len(index))
        for i, idx in enumerate(index):
            if i == 0:
                pf[i] = self.Pf0
            else:
                pf[i] = pf[i-1]*(1 + self.Df/12) + pf[i-1]*self.Vf*np.sqrt(1/12)*np.random.normal()
        return pd.Series(data=pf, index=index)
    
    def ptrans(self, Pf):
        return Pf.add(2*Pf*self.cf*self.tf/self.Vt)
    
if __name__ == '__main__':
    st = dt.datetime(2020, 1, 1)
    h = 5
    x = FuelCost(2, 0.03, 0.14, 0.3, 3, 200, st, h).simfuelcost()

