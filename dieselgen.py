# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 01:54:58 2019

@author: Lefu
"""


class DieselGen(object):
    
    def __init__(self, 
                 Prated,# Generator rated power
                 Fmarg, # marginal fuel consumption per kW
                 Fnl, # no load fuel consumption
                 minLF, # Minimum load factor
                 installdate, # Installation date
                 life, # Hours of generator lifetime 
                 runtime = 0 # hours of runtime since installation
                 ):
        self.Prated = Prated
        self.Fmarg = Fmarg
        self.Fnl = Fnl
        self.minLF =  minLF
        self.installdates = [installdate]
        self.life = life
        self.runtime = runtime
        self.powerout = []
        
    def fuel(self, power, timestamp, run=False):
        
        """ Evaluate fuel consumption """
        if self.runtime < self.life:
            if power > 0:
                if power <= self.Prated:
                    if run == True:
                        self.runtime += 1
                    self.powerout.append(power)
                    return self.Fmarg*power + self.Fnl
                else:
                    return 0
            else:
                # if run == True:
                #     self.runtime += 1
                return self.Fnl
        else:
            self.runtime = 0
            self.installdates.append(timestamp)
            return self.fuel(power, timestamp)
        
if __name__=='__main__':
    
    dgen = DieselGen(10, 0.2, 0.01, 0.3, 1, 20000)
    fl = dgen.fuel(9)