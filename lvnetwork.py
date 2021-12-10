#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime as dt

class LVNetwork:
    
    
    def __init__(self, Ncusts, CustLen, installdate, life):
        self.Ncusts = Ncusts
        self.CustLen = CustLen
        self.length = self.Ncusts*self.CustLen
        self.installdate = installdate
        self.installdates = [installdate]
        self.life = life
        self.endtime = installdate + dt.timedelta(days = 365*self.life) + dt.timedelta(hours=23)
        self.additionallength = {}
        
        
    def update(self, timestamp, addcusts=0):
        
        """ If the network has reached end of life, replace it """
        if addcusts > 0:
            self.additionallength[timestamp] = addcusts*self.CustLen
        
        if timestamp > self.endtime:
            self.installdates.append(timestamp)
            self.installdate = timestamp
            self.endtime = timestamp + dt.timedelta(days = 365*self.life) + dt.timedelta(hours=23)
    
        
if __name__=='__main__':
    
    Cs = 50
    Clen = 22
    inst = dt.datetime(2021, 1, 1)
    d2 = inst + dt.timedelta(days=7500)
    life = 20
    
    network = LVNetwork(Cs, Clen, inst, life)
    
    network.update(d2, 20)
    
    
