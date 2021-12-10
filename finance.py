# -*- coding: utf-8 -*-

import datetime as dt
import pandas as pd
import numpy as np
import numpy_financial as npf
import utils
import forex
import fuel


class Finance:
    
    def __init__(self, 
                 dispatch_res, 
                 startdate,
                 horizon,
                 project_country,
                 finance_country,
                 fixedOM,
                 kWhtariff,
                 monthlycharge,
                 connectionfee,
                 Ncustomers,
                 metercost,
                 LVdistributioncost,
                 initialfuelprice,
                 fuelpricedrift,
                 fuelpricevolatility,
                 fuelconsumption,
                 fueltransittime,
                 fuelvolumetransported,
                 debtratio = 0.75,
                 debtduration = 'All Capex',
                 debttenor = 3,
                 debtcost = 12.5,
                 equitycost = 10,
                 costtype = 'Real',
                 pvcapex = 0,
                 batcapex = 0,
                 invcapex = 0,
                 dgencapex = 0,
                 solarPVOM = 0,
                 batteryOM = 0, 
                 dieselgensetOM = 0,
                 capexsub = 0,
                 capexsubtype = "All Capex",
                 cashgrant = 0,
                 tariffsub = 0,
                 fuelsub = 0,
                 opexsub = 0,
                 taxrate = 0.3,
                 ):
        
        self.res = dispatch_res
        self.startdate = startdate 
        self.horizon = horizon
        self.kWhtariff = kWhtariff
        self.monthlycharge = monthlycharge
        self.connectionfee = connectionfee
        self.Ncustomers = Ncustomers
        self.index = pd.date_range(start=startdate - dt.timedelta(hours=1),
                                 periods=12*horizon + 1, freq='M')
        # Simulate fuel prices over the modelling horizon
        self.fuelprice = fuel.FuelCost(initialfuelprice, fuelpricedrift, 
                                       fuelpricevolatility, fuelconsumption,
                                       fueltransittime, fuelvolumetransported,
                                       self.startdate, self.horizon).simfuelcost()
        self.debtratio = debtratio
        self.debtduration = debtduration
        self.debtcost = pd.Series(data=debtcost/(12*100), index=self.index)
        self.debttenor = debttenor
        self.equitycost = pd.Series(data=equitycost/(12*100), index=self.index)
        # If debt & equity cost are in real terms, add the inflation for finance country
        if costtype == 'Real': 
            self.inf = forex.Inflation(finance_country, startdate, horizon).simulate_inflation()['INF']
            self.debtcost = self.debtcost.append(self.inf[self.index])
            self.debtcost = self.debtcost.groupby(level=0).sum()
            self.equitycost = self.equitycost.append(self.inf[self.index])
            self.equitycost = self.equitycost.groupby(level=0).sum()
        
        self.capex = {'pvgen':pvcapex,
                      'battery':batcapex,
                      'inverter':invcapex,
                      'dgens':dgencapex}
        self.XR = forex.Forex(project_country, finance_country, self.startdate,
                              self.horizon).ExchangeRates()
        self.fixedOM = fixedOM
        self.solarPVOM = solarPVOM
        self.batteryOM = batteryOM
        self.dieselgensetOM = dieselgensetOM
        self.capexsub = capexsub
        self.capexsubtype = capexsubtype
        self.cashgrant = cashgrant
        self.tariffsub = tariffsub
        self.fuelsub = fuelsub
        self.opexsub = opexsub
        self.taxrate = taxrate
        self.metercost = metercost
        self.LVdistributioncost = LVdistributioncost
        
    
    
    def cashflows(self):
        
        """ Evaluate the Cashflows of a project """
        
        cap = self.capital().filter(items=['Total_Cap', 'Interest', 'Principal'])
        dep = self.depreciation().filter(items=['TotalDep'])
        opex = self.expenses().filter(items=['TotalOpex'])
        rev = self.revenues().filter(items=['TotalRev'])
        flows = cap.join(dep).join(opex).join(rev)
        flows['EBITDA'] = flows.TotalRev - flows.TotalOpex
        flows['EBIT'] = flows.EBITDA - flows.TotalDep
        flows['EBT'] = flows.EBIT - flows.Interest
        flows['TaxableIncome'] = self.taxableincome(flows.EBT.values)
        flows['Tax'] = flows.TaxableIncome*self.taxrate
        flows['Equity'] = flows.EBITDA - flows.Total_Cap*(1 - self.debtratio) \
            - flows.Interest - flows.Principal - flows.Tax
        flows['DSCR'] = (flows.EBITDA - flows.Tax)/(flows.Interest + flows.Principal)
        
        debttimes = pd.date_range(start=self.index[1], periods=self.debttenor*12,
                                  freq='M')
        DSCR = flows.loc[debttimes, 'DSCR'].values.mean()
        EquityNPV = npf.npv(self.equitycost, flows.Equity.values).round(2)
        return flows, DSCR, EquityNPV
    
    
    def capital(self):
        
        """ Evaluate capital costs over project lifetime """
        
        comps = self.res['components']
        caps = np.zeros((len(self.index), 5+len(comps['dgens']) + 1))
        cols = ['PVGen', 'Battery', 'Inv', 'Meters', 'LVNet'] 
        cols = cols + ['DGen_'+str(i) for i in range(len(comps['dgens']))]
        cols = cols + ['Total_Cap']
        capsdf = pd.DataFrame(data=caps, index=self.index, columns=cols)
        for key, val in comps.items():
            if key in ['pvgen', 'battery', 'inverter', 'rectifier', 'smartmeter',
                       'LVnetwork']:
                dates = [utils.lastmonthday(i - dt.timedelta(hours=1)) for i in 
                          val.installdates]
                if key == 'pvgen':
                    capsdf.loc[dates, 'PVGen'] = val.Parray*self.capex['pvgen']
                    capsdf.loc[dates, 'Total_Cap'] = capsdf.loc[dates, 'Total_Cap'] \
                        + val.Parray*self.capex['pvgen']
                elif key == 'battery':
                    capsdf.loc[dates, 'Battery'] = val.Capacity*self.capex['battery']
                    capsdf.loc[dates, 'Total_Cap'] = capsdf.loc[dates, 'Total_Cap'] \
                        + val.Capacity*self.capex['battery']
                elif key == 'inverter':
                    capsdf.loc[dates, 'Inv'] = val.capacity*self.capex['inverter']
                    capsdf.loc[dates, 'Total_Cap'] = capsdf.loc[dates, 'Total_Cap'] \
                        + val.capacity*self.capex['inverter']
                elif key == 'smartmeter':
                    capsdf.loc[dates, 'Meters'] = val.ncustomers*self.metercost
                    capsdf.loc[dates, 'Total_Cap'] = capsdf.loc[dates, 'Total_Cap']\
                        + val.ncustomers*self.metercost
                elif key == 'LVnetwork':
                    capsdf.loc[dates, 'LVNet'] = val.length*self.LVdistributioncost
                    capsdf.loc[dates, 'Total_Cap'] = capsdf.loc[dates, 'Total_Cap']\
                        + val.length*self.LVdistributioncost
            else: # diesel generators
                for idx, gen in enumerate(val):
                    dates = [utils.lastmonthday(i - dt.timedelta(hours=1)) for i in 
                          gen.installdates]
                    capsdf.loc[dates, 'DGen_'+str(idx)] = gen.Prated*self.capex['dgens']
                    capsdf.loc[dates, 'Total_Cap'] = capsdf.loc[dates, 'Total_Cap'] \
                        + gen.Prated*self.capex['dgens']
        # compute the capital subsidy
        if self.capexsubtype == 'All Capex':
            capsdf.loc[:, 'Total_Cap'] = capsdf.loc[:, 'Total_Cap']*(1 - self.capexsub)
            capsdf['CapexSubsidy'] = capsdf.loc[:, 'Total_Cap']*self.capexsub/(1 - self.capexsub)
        elif self.capexsubtype == 'Initial':
            csub = np.zeros(len(capsdf))
            csub[0] = self.capexsub
            capsdf.loc[:, 'Total_Cap'] = capsdf.loc[:, 'Total_Cap']*(1 - csub)
            capsdf['CapexSubsidy'] = capsdf.loc[:, 'Total_Cap']*csub/(1 - csub)
            
        capital = capsdf.Total_Cap.sum()
        if self.debtduration != 'All Capex':
            capital = capsdf.Total_Cap.values[0]
        debtcapital = capital*self.debtratio
        debttimes = self.debttenor*12
        per = np.arange(1, debttimes+1, 1)
        rates = self.debtcost.values[1:debttimes+1]
        interest = np.zeros(len(self.index))
        interest[1:debttimes+1] = npf.ipmt(rates, per, len(per), debtcapital)
        principal = np.zeros(len(self.index))
        principal[1:debttimes+1] = npf.ppmt(rates, per, len(per), debtcapital)
        capsdf['Interest'] = np.abs(interest)
        capsdf['Principal'] = np.abs(principal)
        return capsdf
    
    
    def depreciation(self):
        
        """ Evaluate depreciation over the lifetime of the assets """
       
        comps = self.res['components']
        deps = np.zeros((len(self.index), 5+len(comps['dgens']) + 1))
        cols = ['PVGen', 'Battery', 'Inv', 'Meters', 'LVNet'] 
        cols = cols + ['DGen_'+str(i) for i in range(len(comps['dgens']))]
        cols = cols + ['TotalDep']
        depsdf = pd.DataFrame(data=deps, index=self.index, columns=cols)
        for key, val in comps.items():
            if key in ['pvgen', 'battery', 'inverter', 'rectifier', 'smartmeter',
                       'LVnetwork']:
                dates = [utils.lastmonthday(i) for i in val.installdates]
                if key == 'pvgen':
                    pvdep = pd.Series(data=deps[:,0], index=self.index)
                    for d in dates:
                        capex = val.Parray*self.capex['pvgen']
                        capex = self._capexsub(capex, d)
                        dep = self._depreciation(capex, d, val.life)
                        pvdep = pvdep.append(dep)
                    pvdep = pvdep.groupby(level=0).sum()
                    depsdf.loc[:, 'PVGen'] = pvdep[self.index]
                    depsdf.loc[:, 'TotalDep'] = depsdf.loc[:, 'TotalDep'] + pvdep[self.index]
                if key == 'battery':
                    batdep = pd.Series(data=deps[:,1], index=self.index)
                    for d in dates:
                        capex = val.Capacity*self.capex['battery']
                        capex = self._capexsub(capex, d)
                        dep = self._depreciation(capex, d, val.life)
                        batdep = batdep.append(dep)
                    batdep = batdep.groupby(level=0).sum()
                    depsdf.loc[:, 'Battery'] = batdep[self.index]
                    depsdf.loc[:, 'TotalDep'] = depsdf.loc[:, 'TotalDep'] + batdep[self.index]
                if key == 'inverter':
                    invdep = pd.Series(data=deps[:,2], index=self.index)
                    for d in dates:
                        capex = val.capacity*self.capex['inverter']
                        capex = self._capexsub(capex, d)
                        dep = self._depreciation(capex, d, val.life)
                        invdep = invdep.append(dep)
                    invdep = invdep.groupby(level=0).sum()
                    depsdf.loc[:,'Inv'] = invdep[self.index]
                    depsdf.loc[:, 'TotalDep'] = depsdf.loc[:, 'TotalDep'] + invdep[self.index]
                if key == 'smartmeter':
                    smdep = pd.Series(data=deps[:,4], index=self.index)
                    for d in dates:
                        capex = val.ncustomers*self.metercost
                        capex = self._capexsub(capex, d)
                        dep = self._depreciation(capex, d, val.life)
                        smdep = smdep.append(dep)
                    smdep = smdep.groupby(level=0).sum()
                    depsdf.loc[:, 'Meters'] = smdep[self.index]
                    depsdf.loc[:, 'TotalDep'] = depsdf.loc[:, 'TotalDep'] + smdep[self.index]
                if key == 'LVnetwork':
                    lvdep = pd.Series(data=deps[:,5], index=self.index)
                    for d in dates:
                        capex =  val.length*self.LVdistributioncost
                        capex = self._capexsub(capex, d)
                        dep = self._depreciation(capex, d, val.life)
                        lvdep = lvdep.append(dep)
                    lvdep = lvdep.groupby(level=0).sum()
                    depsdf.loc[:, 'Meters'] = lvdep[self.index]
                    depsdf.loc[:, 'TotalDep'] = depsdf.loc[:, 'TotalDep'] + lvdep[self.index]
            else:
                for idx, gen in enumerate(val):
                    dates = [utils.lastmonthday(i) for i in gen.installdates]
                    gendep = pd.Series(data=deps[:,3+idx+1], index=self.index)
                    for d in dates:
                        capex = gen.Prated*self.capex['dgens']
                        capex = self._capexsub(capex, d)
                        h = int(np.floor(gen.life/8760))
                        dep = self._depreciation(capex, d, h)
                        gendep = gendep.append(dep)
                    gendep = gendep.groupby(level=0).sum()
                    depsdf.loc[:, 'DGen_'+str(idx)] = gendep[self.index]
                    depsdf.loc[:, 'TotalDep'] = depsdf.loc[:, 'TotalDep'] + gendep[self.index]
        return depsdf
    
    def _depreciation(self, capex, capdate, _horizon, salvage=0):
        
        """ Return depreciation over period """
        
        N = 12*_horizon
        index = pd.date_range(start=capdate, periods=N, freq='M')
        dep = (capex - salvage)/N
        return pd.Series(dep, index=index)
    
    def _capexsub(self, capex, date):
        if self.capexsubtype == 'All Capex':
            cap = capex*(1 - self.capexsub)
        elif self.capexsubtype == 'Initial':
            if date == self.startdate:
                cap = capex*(1 - self.capexsub)
            else:
                cap = capex
        return cap
    
    def expenses(self):
        
        """ Evaluate operating expenses """
        
        res = self.res['results']
        dgparams = list(res.columns.values)[11:-2]
        dftarget = res.filter(items=['PVDC', 'BAT']+dgparams)
        # Define groupby dates
        startdates = [i + dt.timedelta(hours=1) for i in self.index]
        starts = startdates[:-1]
        ends = self.index[1:]
        dataholder = np.zeros((len(starts)+1, 2 + len(dgparams)))
        # Group data
        for idx, (st, et) in enumerate(zip(starts, ends)):
            index = pd.date_range(start=st, end=et, freq='H')
            temp = dftarget.loc[index].abs().to_numpy()
            dataholder[idx+1, :] = temp.sum(axis=0)
        # Put into a dataframe
        aggregatedf = pd.DataFrame(data=dataholder, index=self.index, 
                                   columns=['PVDC', 'BAT'] + dgparams)
        # Convert dataframe into a matrix
        agmat = aggregatedf.to_numpy()
        # Isolate battery and PV power/energy
        pv_bat = agmat[:, :2]
        # Isolate diesel generator power and fuel used
        dg_mat = agmat[:, 2:]
        n = len(dgparams)
        # Dieselgenset power
        dg_power = dg_mat[:, :n//2].sum(axis=1)
        # Fuel used
        dg_fuel = dg_mat[:, n//2:].sum(axis=1)
        
        # Compute operational costs
        costs = np.zeros((len(pv_bat), 6))
        costs[1:, 0] = self.fixedOM
        costs[:, 1] = pv_bat[:, 0]*self.solarPVOM
        costs[:, 2] = pv_bat[:, 1]*self.batteryOM
        costs[:, 3] = dg_power*self.dieselgensetOM
        costs[:, 4] = dg_fuel*self.fuelprice
        costs[:, 5] = costs[:, :5].sum(axis=1)
        
        cols = ['Fixedopex', 'PVopex', 'Batopex', 'DGenopex', 'Fuelopex', 'TotalOpex']
        
        return pd.DataFrame(data=costs, columns=cols, index=self.index)
    
    def revenues(self):
        
        """ Evaluate revenues """
        
        res = self.res['results']
        dgparams = list(res.columns.values)[13:]
        targetdf = res.filter(items=['load'] + dgparams)
        
        # Define groupby dates
        startdates = [i + dt.timedelta(hours=1) for i in self.index]
        starts = startdates[:-1]
        ends = self.index[1:]
        
        # Revenues data holder
        revs = np.zeros((len(starts)+1, 8))
        # Billable load aggregated monthly
        for idx, (st, et) in enumerate(zip(starts, ends)):
            index = pd.date_range(start = st, end = et, freq='H')
            temp = targetdf.loc[index].abs().to_numpy()
            revs[idx+1, 0] = temp.sum(axis=0)[0]
            revs[idx+1, 5] = temp.sum(axis=0)[1:].sum()
            
        revs[:, 0] = revs[:, 0]*self.kWhtariff/self.XR[self.index]
        revs[1:, 1] = self.monthlycharge
        revs[0, 2] = self.Ncustomers*self.connectionfee
        revs[0, 3] = self.cashgrant
        revs[:, 4] = revs[:, 0]*self.tariffsub/self.kWhtariff
        revs[:, 5] = revs[:, 5]*self.fuelprice*self.fuelsub
        revs[1:, 6] = self.Ncustomers*self.opexsub
        revs[:, 7] = revs[:, :7].sum(axis=1)
        
        cols = ['consumption', 'fixedrev', 'connections', 'grant', 'tariff_sub',
                'fuel_sub', 'opexsub', 'TotalRev']
        return pd.DataFrame(data=revs, columns=cols, index=self.index)
    
    
    def taxableincome(self, ebt):
        inc = np.zeros(len(ebt))
        carry = 0
        for idx, eb in enumerate(ebt):
            if idx == 0:
                if eb > 0:
                    inc[idx] = eb
            else:
                if eb < 0:
                    carry += eb
                else:
                    temp = eb + carry
                    if temp >= 0:
                        inc[idx] = temp
                        carry = 0
        return inc


if __name__ == '__main__':
    
    s = dt.datetime(2020, 1, 1)
    h = 4
    f = Finance(0, s, h, 'Rwanda', 'United States', 0.75)







