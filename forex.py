# -*- coding: utf-8 -*-

import re
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg


# Function to remove punctuation and lower case of a string
standardizenames = lambda _str: re.sub(r'[^\w\s ]', '', _str).lower().replace(" ", "")

class Forex:
    
    def __init__(self, project_country, finance_country, startdate, horizon):
        self.pc = project_country
        self.fc = finance_country
        self.startdate = startdate
        self.horizon = horizon
        
    def ExchangeRates(self):
        """ Simulate time series monthly exchange rates """
        host_inf = Inflation(self.pc, self.startdate, self.horizon)
        host_res = host_inf.simulate_inflation()
        host_cpi = host_res['CPI']
        
        host_scale = ScalingFactors(self.pc, self.startdate, self.horizon)
        host_sf = host_scale.simulate_sf()
        
        for_inf = Inflation(self.fc, self.startdate, self.horizon)
        for_res = for_inf.simulate_inflation()
        for_cpi = for_res['CPI']
        
        for_scale = ScalingFactors(self.fc, self.startdate, self.horizon)
        for_sf = for_scale.simulate_sf()
        
        return host_sf*host_cpi/(for_sf*for_cpi)
        
        
class Inflation:
    
    def __init__(self, country, startdate, horizon):
        self.country = country
        self.startdate = startdate
        self.horizon = horizon
        self.inflation = None
        self.consumerpriceindex = None
        
    def simulate_inflation(self):
        data = self.get_data()
        self._inflation(data)
        return {'INF':self.inflation, 'CPI':self.consumerpriceindex}
        
    def get_data(self):
        """ Get country specific data from the world bank data """
        cpidata = pd.read_excel("CPI Price, seas. adj..xlsx", sheet_name='monthly')
        snames = list(map(standardizenames, list(cpidata.columns)))
        cpidata.columns = snames
        _id = ''
        for _str in cpidata.columns:
            if standardizenames(self.country) in _str:
                _id = _id + _str
                break
        cpi = cpidata[_id].dropna().values
        return cpi/100
    
    def ar_params(self, data):
        """ 
        Evaluate constants QMU, QA and QV from timeseries data  
        
        It = QMU + QA * (It - QMU) + np.sqrt(QV) * N(0,1)
        
        params:
            data (1-d array) - Time series inflation data
            
        Return:
            QMU, QA, QV
            
        """
        data = np.log(data)
        y = data[1:]
        x = data[:-1]
        I = y - x
        QMU = I.mean()
        QSD = I.std()
        mod = AutoReg(I, 1, old_names=False).fit()
        QA = mod.params[1]
        
        return (QMU, QA, QSD)

    def _inflation(self, data):
        """ Evaluate timeseries inflation and consumper price index from 2011 """
        idx = np.where(data == 1)[0][0]
        selectdata = data[idx:]
        QMU, QA, QSD = self.ar_params(data)
        I = np.log(selectdata[1:]) - np.log(selectdata[:-1])
        st = dt.datetime(2010, 1, 31, 23)
        pers = 12*(self.startdate.year - 2010 + self.horizon)
        index = pd.date_range(start=st, periods=pers, freq='M')
        Iinit = I[0]
        CPIinit = selectdata[0]
        inf = []
        cpi = []
        for i, idx in enumerate(index):
            if i == 0:
                inf.append(Iinit)
                cpi.append(CPIinit)
            else:
                Iinit = QMU + QA*(Iinit - QMU) + QSD*np.random.normal()
                CPIinit = CPIinit*np.exp(Iinit)
                inf.append(Iinit)
                cpi.append(CPIinit)
        self.inflation = pd.Series(data=inf, index=index)
        self.consumerpriceindex = pd.Series(data=cpi, index=index)


class ScalingFactors:
    
    def __init__(self, country, startdate, horizon):
        self.country = country
        self.startdate = startdate
        self.horizon = horizon
        self.sf = None
        
    def simulate_sf(self):
        data = self.get_data()
        self._scalingfactor(data)
        return self.sf
        
    def get_data(self):
        XR = pd.read_excel("Exchange rate, new LCU per USD extended backward, period average.xlsx", 
                                sheet_name='monthly')
        snames = list(map(standardizenames, list(XR.columns)))
        XR.columns = snames
        def fun(x):
            try:
                return int(str(x).split('M')[0])
            except:
                return -999
        XR['year'] = list(map(fun, XR['unnamed0']))
        XR = XR[XR.year >= 2010]
        _id = ''
        for _str in XR.columns:
            if standardizenames(self.country) in _str:
                _id = _id + _str
                break
        data = XR[_id].dropna().values
        return data
    
    def ar_params(self, data):
        data = np.log(data)
        y = data[1:]
        x = data[:-1]
        XN = y - x
        XMU = data.mean()
        XSD = XN.std()
        mod = AutoReg(XN, 1, old_names=False).fit()
        XA = mod.params[1]
        return (XMU, XA, XSD)
    
    def _scalingfactor(self, data):
        XMU, XA, XSD = self.ar_params(data)
        xn = np.log(data[1:]) - np.log(data[:-1])
        st = dt.datetime(2010, 1, 31, 23)
        pers = 12*(self.startdate.year - 2010 + self.horizon)
        index = pd.date_range(start=st, periods=pers, freq='M')
        xinit = xn[0]
        XKinit = data[0]
        XK = []
        for i, idx in enumerate(index):
            if i == 0:
                XK.append(XKinit)
            else:
                XKi = np.exp(XMU + xinit)
                xinit = XA*xinit + XSD*np.random.normal()
                XK.append(XKi)
        self.sf = pd.Series(data=XK, index=index)
                
        

if __name__ == '__main__':
    
    country = 'United States'
    startdate = dt.datetime(2020, 1, 1)
    horizon = 20
    
    
    _forex = Forex('Germany', 'United States', startdate, horizon)
    _for = Forex('Lesotho', 'United States', startdate, horizon)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(4,6))
    for i in range(10):
        rates1 = _forex.ExchangeRates()
        rates1.plot.line(ax=ax0)
        rates2 = _for.ExchangeRates()
        rates2.plot.line(ax=ax1)
    ax0.set_ylabel("EUR/USD")
    ax1.set_ylabel("LSL/USD")
    ax1.set_xlabel("Year")
    
    
    # Visualize rates
    sc = ScalingFactors("Lesotho", startdate, horizon)
    d = sc.get_data()
    sc._scalingfactor(d)
    m = sc.sf
   # print(sc.ar_params(d))
    x = np.log(d[1:]) - np.log(d[:-1])
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(range(len(d)), d)
    ax2.plot(range(len(x)), x)
    
    
    iR = Inflation('Lesotho', startdate, horizon).simulate_inflation()['INF']

