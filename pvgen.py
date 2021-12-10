# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 01:57:11 2019

@author: Lefu
"""

import datetime as dt
import numpy as np


q = 1.602*(10**-19) # Electron Charge
k = 1.381*(10**-23) # Planck's constant

class PVGen(object):
    """
    
    """
    def __init__(self, 
                 Parray, # Rated DC capacity of the PV array
                 Voc0, # Open circuit voltage 
                 Isc0, # Closed circuit current 
                 Vmpp, # Maximum power point voltage at STC
                 Impp, # Maximum power point current at STC
                 NOCT, # Nominal operating cell temperature
                 tcoeff, # temperature coefficient
                 life, # solar array lifetime in years
                 endoflifecapacity, # end of life capacity
                 installdate, # Installation date
                 cablelosses = 0, # cabling losses
                 soillosses = 0, # Soiling losses
                 otherlosses = 0
                 ):
        self.Parray = Parray
        self.Voc0 = Voc0
        self.Isc0 = Isc0
        self.Vmpp = Vmpp
        self.Impp = Impp
        self.NOCT = NOCT
        self.tcoeff = tcoeff
        self.life = life
        self.eolc = endoflifecapacity
        self.installdate = installdate
        self.installdates = [installdate]
        self.starttime = installdate
        self.endtime = installdate + dt.timedelta(days = 365*self.life) + dt.timedelta(hours=23)
        self.cl = cablelosses
        self.sl = soillosses
        self.ol = otherlosses
    
    
    def Voc(self, G):
        """
        Evaluate open circuit voltage 

        Parameters
        ----------
        G : float
            Insolation incident on the generator.

        Returns
        -------
        Voc: float
            Short circuit voltage at G

        """
        return self.Voc0/(1 + 0.058*np.log(1000/G))
    
    
    def Isc(self, G):
        """
        Evaluate short circuit current 

        Parameters
        ----------
        G : float
            Insolation incident on the generator.

        Returns
        -------
        Isc: float
            Short circuit voltage

        """
        return self.Isc0*G/1000
    
    def Vth(self, T):
        """
        Evaluate thermal voltage

        Parameters
        ----------
        T : float
            Temperature

        Returns
        -------
        Vth: float
            Thermal voltage

        """
        return 1*k*(T + 273.15)/q
    
    def Rs(self, T):
        """ Evaluate module series resistance """
        vth = self.Vth(T)
        rs_init = 0
        rs = (self.Voc0 - self.Vmpp - vth\
              *np.log((self.Vmpp + vth - self.Impp*rs_init)/vth))/self.Impp
            
        while abs(rs - rs_init) > 0.001:
            rs_init = rs
            rs = (self.Voc0 - self.Vmpp - vth\
              *np.log((self.Vmpp + vth - self.Impp*rs_init)/vth))/self.Impp
        return rs
    
    def Rsh(self, T):
        """ Evaluate shunt resistance """
        rs = self.Rs(T)
        vth = self.Vth(T)
        numerator = (self.Vmpp - 1*vth)*(self.Vmpp - self.Impp*rs)
        denom = (self.Isc0 - self.Impp)*(self.Vmpp - self.Impp*rs) - self.Impp*1*vth
        return numerator/denom
    
    
    def voc(self, T, G):
        """ Evaluate normalized open circuit voltage """
        return q*self.Voc(G)/(72*1*k*(T + 273.15))
    
    def ff(self, T, G):
        """ Evaluate the fill factor """
        voc = self.voc(T, G)
        FF0 = (voc - np.log(voc + 0.72))/(voc + 1)
        Rs = self.Rs(T)
        Rsh = self.Rsh(T)
        Voc = self.Voc(G)
        Isc = self.Isc(G)
        FFs = FF0*(1 - Rs/(Voc/Isc))
        return FFs*(1 - (voc + 0.7)/voc*FFs/(Rsh/(Voc/(72*Isc))))
    
    def tcell(self, Ta, G):
        """ Evaluate cell temperature """
        return Ta + (self.NOCT - 20)*G/800
        
    def capacity(self, time):
        """ Evaluate capacity at time time """
        _end = self.installdate - dt.timedelta(hours=1)
        enddate = dt.datetime(
                            self.installdate.year+self.life,
                            _end.month,
                            _end.day,
                            _end.hour
                              )
        slope = (self.eolc - 1)/((enddate - self.installdate).days*24)
        return 1 + slope*((time - self.installdate).days*24)
        
        
    def DCPowerOut(self, G, Ta, timestamp):
        """ Evaluate DC power output """
        if timestamp < self.endtime:
            if G > 0:
                T = self.tcell(Ta, G)
                Pmpp = self.ff(T, G)*self.Voc(G)*self.Isc(G)*(1 - self.tcoeff*(T - 25))
                P = self.Parray*1000/(self.Vmpp*self.Impp)*Pmpp*(1-self.cl)*(1-self.sl)*(1-self.ol)
                cap = self.capacity(timestamp)
                return max(0, P*cap/1000)
            else:
                return 0
        else:
            self.installdates.append(timestamp)
            self.starttime = timestamp
            self.endtime = self.starttime + dt.timedelta(days = 365*self.Life) +\
                        dt.timedelta(hours=23)
            return self.dcPowerOut(Ta, G, timestamp)
    
    
    def __str__(self):
        _str = "PV array size: {} kW\n"
        _str += "Open circuit voltage: {} V\n"
        _str += "Short circuit current: {} A\n"
        _str += "Maximum power point voltage at STC: {} V\n"
        _str += "Maximum power point current at STC: {} A\n"
        _str += "Nominal Operating Cell Temperature: {} degrees C\n"
        _str += "Temperature coefficent: {}\n"
        _str += "PV module life: {} years\n"
        _str += "PV module end of life capacity: {} %\n"
        _str += "Installation date: {}\n"
        _str += "Cabling losses: {} %\n"
        _str += "Soiling losses: {} %\n"
        _str += "Other losses: {} %\n"
        _str = _str.format(self.Parray, self.Voc0, self.Isc0, self.Vmpp, self.Impp,
                          self.NOCT, self.tcoeff, self.life, self.eolc*100, 
                          self.installdate, self.cl*100, self.sl*100, self.ol*100)
        return _str
    
    def __repr__(self):
        _str = 'PVGen({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})'
        _str = _str.format(self.Parray, self.Voc0, self.Isc0, self.Vmpp, self.Impp,
                           self.NOCT, self.tcoeff, self.life, self.eolc, 
                          self.installdate, self.cl, self.sl, self.ol)
        return _str
    
    
if __name__ == '__main__':
    
    # Test the code
    pvgen = PVGen(1,
                  38, 
                  8.79, 
                  30.3, 
                  8.27, 
                  44, 
                  0.0041,
                  25,
                  0.8,
                  dt.datetime(2020, 1, 1)
                  )
    
    power = pvgen.DCPowerOut(1000, 25, dt.datetime(2020, 1, 2, 3))
    