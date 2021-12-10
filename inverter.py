# -*- coding: utf-8 -*-
"""
This module simulates a perfromance of a two way inverter (DC-AC & AC-DC)
"""
import datetime as dt


class Inverter(object):
    
    def __init__(
                 self, 
                 capacity,
                 life,
                 installdate,
                 efficiency = 0.98
                 ):
        """Inverter Object initializer:
            Capacity: Inverter rated power
            life: Inverter useful lifetime
            ST: Date at which the inverter starts operation
            age: length of time in years the inverter has been in operation
            efficiency: inverter rated efficiency
            invcapex: capital cost of an inverter per kW
            invrepcapex: cost of replacing an inverter
        """
        self.capacity = float(capacity)
        self.life = int(life)
        self.installdates = [installdate]
        self.starttime = installdate
        self.endtime = installdate + dt.timedelta(weeks=52*self.life) + dt.timedelta(hours=23) 
        self.efficiency = float(efficiency)
        
    
    def AC_Out(self, DCpower, timestamp):
        """
        Return AC power output from an inverter
        """
        if timestamp < self.endtime: 
            if DCpower >= self.capacity:
                return self.capacity*self.efficiency
            else:
                return DCpower*self.efficiency 
        else:
            self.installdates.append(timestamp)
            self.starttime = timestamp
            self.endtime = self.starttime + dt.timedelta(weeks=52*self.life) + \
                        dt.timedelta(hours=23)
            return self.AC_Out(DCpower, timestamp)

    def DC_Out(self, ACPower, timestamp):
        """
        Return DC power output from the rectifier
        """
        if timestamp < self.endtime:
            if ACPower >= self.capacity:
                return self.capacity*self.efficiency
            else:
                return ACPower*self.efficiency
        else:
            self.installdates.append(timestamp)
            self.starttime = timestamp
            self.endtime = self.starttime + dt.timedelta(days = 365*self.life) +\
                        dt.timedelta(hours=23)
            return self.DC_Out(ACPower, timestamp)
