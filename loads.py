# -*- coding: utf-8 -*-

import numpy as np
import datetime as dt
import pandas as pd
import seaborn as sns
import pickle


class Loads(object):
    
    def __init__(self, 
                 Mean,
                 STDdev,
                 basetariff,
                 targettariff,
                 start_time,
                 horizon,
                 techLoss = 0,
                 nonTechLoss = 0,
                 elasticity = 0,
                 repload = []
                 ):
        self.Mean = Mean
        self.STDdev = STDdev
        self.basetariff = basetariff
        self.targettariff = targettariff
        self.start_time = start_time
        self.end_time = dt.datetime(start_time.year + horizon, start_time.month,
                                    start_time.day) - dt.timedelta(hours=1)
        self.techLoss = techLoss
        self.nonTechLoss = nonTechLoss
        self.elasticity = elasticity
        self.repload = repload
        if not self.repload:
            fi = open('repload.pkl', 'rb')
            self.repload = pickle.load(fi)
            fi.close()
        
    def houlyloadsim(self):
        
        """ Simulate hourly load profile for the entire horizon """
        
        dailyprofile = np.array(self.dailyloadsim())
        index = pd.date_range(start=self.start_time, end=self.end_time,
                              freq='H')
        loadp = np.outer(dailyprofile, self.repload).flatten()
        
        return pd.DataFrame(loadp, index=index, columns=['load'])
        
        
    def dailyloadsim(self):
        
        """ Simulate daily loads for the entire horizon """
        
        timestamps = list(pd.date_range(start=self.start_time, end=self.end_time,
                                   freq='D'))
        return list(map(self.dailyload, timestamps))
        
    def dailyload(self, timestamp):
        """ 
        Evaluate the total demand for the day of the timestamp
        """
        if timestamp.timetuple().tm_wday in [5, 6]: # weekend
            dl = np.random.normal(self.Mean[timestamp.month - 1, 1], 
                                  self.STDdev[timestamp.month - 1, 1])
            return self.adjustDemand(dl/((1-self.techLoss)*(1-self.nonTechLoss)))
        else:
            dl = np.random.normal(self.Mean[timestamp.month - 1, 0], 
                                  self.STDdev[timestamp.month - 1, 0])
            return self.adjustDemand(dl/((1-self.techLoss)*(1-self.nonTechLoss)))
        
    def adjustDemand(self, demand):
        """
        Adjust demand for price elasticity
        """
        return demand*(1 + self.elasticity*(self.targettariff - self.basetariff)/self.basetariff)
    
    
if __name__ == '__main__':
    mean = np.array([[40, 50], [43, 53], [44, 54], [48, 58], [50, 60], [53, 63],
                     [54, 64], [44, 56], [42, 54], [39, 43], [45, 55], [45, 59]])
    stddev = mean/6
    
    basep = 1.75
    targetp = 1.75
    
    start_time = dt.datetime(2020, 3, 20)
    
    horizon = 30
    
    loads = Loads(mean, stddev, basep, targetp, start_time, horizon)
    
    houlyL = loads.houlyloadsim()