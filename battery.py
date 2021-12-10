# -*- coding: utf-8 -*-

import datetime as dt
import numpy as np
from scipy.optimize import fsolve


q = 1.602*(10**-19) # Electron Charge
k = 1.381*(10**-23) # Planck's constant

class Battery:
    '''
    Lead acid battery model
    '''
    
    def __init__(
                 self,
                 CapRate,
                 TimeRate,
                 NomBatVolt, 
                 NomVolt,
                 BatperString,
                 NStrings,
                 lifemethod,
                 installdate,
                 life,
                 efficiency = 0.85, 
                 etaA = -0.0278, 
                 c = 0.254,
                 k = 0.5181,
                 DODmax = 0.6,
                 SOC = 1, 
                 throughput=0, 
                 capfade=0,
                 cyclelife=500,
                 maxcapfade = 0.2,
                 faderate = 0.00012,
                 kA = 1.058,
                 kB = 44.16,
                 kC = 236.3,
                 chargecost = 0, 
                 tol = 0.1,
                 ):
        '''
        Initialize battery object
            Capacity: Initial storage capacity in kWh
            CapRate: rated battery capacity in Ah
            TimeRate: discharge time for CapRate
            NomVolt: nominal battery voltage
            BatperString: number of batteries per string in bank
            Nstrings: number of strings of batteries in bank
            lifemethod: method for tracking battery lifetime
            efficiency: standard conditions battery efficiency
            etaA: battery efficiency discharge rate coefficient
            c: KiBam model constant c
            k: KiBam model constant k at standard conditions
            DODmax: maximum battery depth of discharge
            SOC: battery state of charge
            throughput: lifetime throughput in Ah
            capfade: total capacity fade
            cyclelife: cycle life of battery at DODmax
            maxcapfade: total capacity fade when batteries must be replaced
            faderate: rate of capacity fade in % of capacity per Ah throughput\
                        at 25 degrees celsius
            kA: constant A in model of k dependence on temperature
            kB: constant B in model of k dependence on temperature
            kC: constant C in model of k dependence on temperature
            chargecost: average cost of energy to charge batteries
            Charge: boolean selector - 'true' for charge & 'false' for discharge
            life: expected lifetime of battery in years
            st: Date of first usage of battery
        '''
        # must choose valid lifemethod
        if lifemethod not in ['capacityfade', 'throughput']: 
            raise Exception('Invalid lifemethod input')
        self.TimeRate = int(TimeRate)
        self.NomBatVolt = float(NomBatVolt)
        self.NomVolt = float(NomVolt)
        self.BatperString = int(BatperString)
        self.NStrings = int(NStrings)
        self.CapRate = self.NStrings*float(CapRate)
        self.Capacity = self.CapRate*self.NomVolt/1000
        self.c = float(c)
        self.k = float(k)
        self.DODmax = float(DODmax)
        self.qmax0 = self.CapRate*((1-np.exp(-self.k*self.TimeRate))*(1-self.c)+\
                                   self.k*self.c*self.TimeRate)/\
                                   (self.k*self.c*self.TimeRate)
        self.qmax = self.qmax0
        self.SOC = float(SOC)
        self.q1 = self.qmax*self.c*self.SOC
        self.q2 = self.qmax*(1-self.c)*self.SOC
        self.q = self.q1 + self.q2
        self.efficiency = float(efficiency)
        self.lifemethod = lifemethod
        self.life = life
        self.throughput = float(throughput)
        self.capfade = float(capfade)
        self.cyclelife = int(cyclelife)
        self.maxthroughput = self.qmax*self.cyclelife*self.DODmax
        self.maxcapfade = float(maxcapfade)
        self.faderate = float(faderate)
        self.kA = float(kA)
        self.kB = float(kB)
        self.kC = float(kC)
        self.etaA = float(etaA)
        self.chargecost = float(chargecost)
        self.tol = float(tol)
        self.installdates = [installdate]
     
    def DispatchBat(self, power, timestamp, temp=25, timestep_size=1):
        """
        Dispatch the battery storage
        """
        k = self.k_adj(temp)
        if self.throughput < self.maxthroughput or self.capfade < self.maxcapfade:
            if power >= 0: # Discharge
                Ibat = self.Discharge(power, temp=temp, timestep_size=timestep_size)
                
                q1 = self.q1*np.exp(-k*timestep_size) \
                     + ((self.q1 + self.q2)*k*self.c-Ibat)*(1-np.exp(-k*timestep_size))/k \
                     - Ibat*self.c*(k*timestep_size - 1 +np.exp(-k*timestep_size))/k
                q2 = self.q2*np.exp(-k*timestep_size) \
                    + (self.q1 + self.q2)*(1 - self.c)*(1 - np.exp(-k*timestep_size)) \
                    - Ibat*(1 - self.c)*(k*timestep_size - 1 + np.exp(-k*timestep_size))/k
                self.q1 = q1
                self.q2 = q2
                self.SOC = (q1 + q2)/self.qmax
        
                if self.lifemethod == 'throughput': 
                    self.throughput = self.throughput + abs(Ibat)*timestep_size/2
                if self.lifemethod == 'capacityfade': 
                    self.capfade = self.capfade + abs(Ibat)*timestep_size*self.faderate*\
                                    (2**((temp - 25)/10))/(2*self.qmax)
                return Ibat*self.dischargeVolt(Ibat)/1000
            else:
                Ibat = self.Charge(power, temp=temp, timestep_size=timestep_size)
                
                q1 = self.q1*np.exp(-k*timestep_size) \
                    + ((self.q1 + self.q2)*k*self.c-Ibat)*(1-np.exp(-k*timestep_size))/k \
                    - Ibat*self.c*(k*timestep_size - 1 +np.exp(-k*timestep_size))/k
                q2 = self.q2*np.exp(-k*timestep_size) \
                        + (self.q1 + self.q2)*(1 - self.c)*(1 - np.exp(-k*timestep_size)) \
                        - Ibat*(1 - self.c)*(k*timestep_size - 1 + np.exp(-k*timestep_size))/k
                self.q1 = q1
                self.q2 = q2
                self.SOC = (q1 + q2)/self.qmax
                
                if self.lifemethod == 'throughput': 
                    self.throughput = self.throughput + abs(Ibat)*timestep_size/2
                if self.lifemethod == 'capacityfade': 
                    self.capfade = self.capfade + abs(Ibat)*timestep_size*self.faderate*\
                                    (2**((temp - 25)/10))/(2*self.qmax)
                return Ibat*self.chargeVolt(Ibat)/1000
        else:
            self.throughput = 0
            self.capfade = 0
            self.SOC = 1
            self.q1 = self.qmax*self.c*self.SOC
            self.q2 = self.qmax*(1-self.c)*self.SOC
            self.q = self.q1 + self.q2
            self.installdates.append(timestamp)
            return self.DispatchBat(power, timestamp, temp, timestep_size)
    
    def replacecost(self, power, timestep_size, temp=25):
        '''
        Only cost the discharge at double rate
        '''
        #Ibat = power*1000/self.NomVolt
        if power > 0: Ibat = self.Discharge(power, temp, timestep_size)
        else: Ibat = self.Charge(power, temp, timestep_size) 
            
        # evaluate throughput if using lifemethod throughput
        if self.lifemethod == 'throughput': 
            throughput = abs(Ibat)*timestep_size/2
            return throughput/self.maxthroughput
        # compute capacity fade if using lifemethod capacityfade
        elif self.lifemethod == 'capacityfade': 
            capacityfade = abs(Ibat)*timestep_size*self.faderate*\
                            2**((temp - 25)/10)/(2*self.qmax)
            return capacityfade/self.maxcapfade
            

    def bateff(self, Ibat):
        '''
        return battery efficiency with current Ibat
        '''
        if Ibat != 0:
            return self.etaA*np.log(abs(Ibat)/(self.CapRate/self.TimeRate)) +\
                    self.efficiency
        else:
            return self.efficiency
        
    def chargeVolt(self, Ibat):
        '''
        return battery charging voltage with current Ibat
        '''
        return self.NomVolt/((self.bateff(Ibat))**(1/2))
        
    def dischargeVolt(self, Ibat):
        '''
        return battery discharge voltage with current Ibat
        '''
        return self.NomVolt*((self.bateff(Ibat))**(1/2))
        
    def k_adj(self, temp):
        '''
        return KiBam constant k adjusted for temperature
        '''
        return self.k*self.kA*np.exp(-self.kB/(temp + 273.15 - self.kC))/0.5181
    
    def Ibatsolver(self, power): 
        '''
        return battery current when charging/discharging power
        '''
        Ibat0 = power*1000/self.NomVolt
        # define function equal to zero when battery current x is equalt to 
        # power divided by voltage
        def f(x): 
            den = self.CapRate/self.TimeRate
            if (power > 0):
                return x - power*1000/(self.NomVolt*(self.etaA*np.log(x/den) + self.efficiency)**(1/2))
            else:
                return x - power*1000/(self.NomVolt/(self.etaA*np.log(-x/den) + self.efficiency)**(1/2))
        if power == 0: # return zero current if power is zero
            return 0
        else: # solve for current such that current = power/voltage(current)
            return fsolve(f, Ibat0, xtol = self.tol)
    
    def Discharge(self, power, temp=25, timestep_size=1):
        '''
        return maximum battery discharge power
            temp: battery temperature in degrees celsius
            timestep_size: timestep size in hours
        '''
        k = self.k_adj(temp)
        Idmax = min((k*self.q1*np.exp(-k*timestep_size) \
                +(self.q1 + self.q2)*k*self.c*(1-np.exp(-k*timestep_size)))\
                    /(1 - np.exp(-k*timestep_size) + self.c*(k*timestep_size\
                      - 1 + np.exp(-k*timestep_size))),
                        (self.SOC - 1 + self.DODmax)*(self.qmax)/timestep_size)
        Ib = self.Ibatsolver(power)
        Ibat = min([Idmax, Ib])
        return Ibat
        
    def Charge(self, power, temp=25, timestep_size=1):
        '''
        return maximum battery charge power
            temp: battery temperature in degrees celsius
            timestep_size: timestep size in hours
        '''
        k = self.k_adj(temp)
        Icmax = (-k*self.c*self.qmax + k*self.q1*np.exp(-k*timestep_size) \
                + (self.q1 + self.q2)*k*self.c*(1-np.exp(-k*timestep_size))) \
                /(1 - np.exp(-k*timestep_size) + self.c*(k*timestep_size \
                  - 1 + np.exp(-k*timestep_size)))
        Ic = self.Ibatsolver(power)
        Ibat = max([Icmax, Ic])
        return Ibat
    
    def __str__(self):
        _str = ''
        return _str
    
    

if __name__=='__main__':

    caprate = 1350
    timerate = 20
    nombatV = 4
    nomV = 48
    batperstr = 12
    nstrings = 3
    lifemethod = 'capacityfade'
    installdate = dt.datetime(2020, 1, 1)
    life = 10
    
    storage = Battery(caprate, timerate, nombatV, nomV, batperstr, nstrings,
                      lifemethod, installdate, life)
    
    