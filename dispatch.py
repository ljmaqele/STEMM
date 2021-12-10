# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import copy


def dispatch(
             loads,
             meteor_data, 
             timestep,
             strategy = 'H',
             REM = 0.0,
             LM = 0.2,
             fuel_cost = 0.45,
             batrepcost = 0.2,
             pvgen = None,
             battery=None, 
             dieselgens=None,
             inverter=None, 
             smartmeter = None,
             lvnetwork = None,
             hstrategy = 'load_following'
             ):
    """
    Dispatch the power system components minimizing cost incurred
    
    Parameters:
        
        loads: list 
            power demand to be served 
            
        timestep: int/float 
            time in hours for which the load has to be sustained
            
        timestamp: datetime (year, month, day, hour)
            
        strategy: character ('H','O') 
            dispatch strategy, either heuristic or optimization 
            
        fuel_cost: float
            cost per liter of diesel
            
        OM_cost: int/float 
            operation and maintenance cost of generator per timestep
            
        bat_charge_cost: 
            cost associated with loss of power incurred when charging the 
            battery with diesel generator
            
        genStartSOC: float
            
        genStopSOC: float
            
        pvgen: Photovoltaic generator object
        
        battery: Battery object
        
        dieselgens: Diesel generator objects
        
        inverter: Inverter object
        
        rectifier: Rectifier object        
        
    Outputs:
        fuel volume used
        PV ac output
        battery power flow (in or out)
        diesel gensets power output (aggregate and individual genset)
    """
    output = {} # Specify outputs structure
    
    # Create copies of the components
    pvg = copy.deepcopy(pvgen)
    bat = copy.deepcopy(battery)
    dgens = copy.deepcopy(dieselgens)
    inv = copy.deepcopy(inverter)
    smmeter = copy.deepcopy(smartmeter)
    lvnet = copy.deepcopy(lvnetwork)
    
    # Create outputs
    size = len(meteor_data)
    PVDC = np.zeros(size)
    PVAC = np.zeros(size)
    BAT = np.zeros(size)
    BATsoc0 = np.zeros(size)
    BATsoc1 = np.zeros(size)
    Metload = np.zeros(size)
    Shedload = np.zeros(size)
    if dgens != None:
        DGENS = np.zeros((size, len(dgens)))
        FUEL = np.zeros((size, len(dgens)))
        
    for idx, timestamp in enumerate(meteor_data.Time):
        load = np.sum(loads[idx])
        T = meteor_data.Temp[meteor_data.Time == timestamp].values[0]
        G = meteor_data.G[meteor_data.Time == timestamp].values[0]
        
        # Determine load to meet or shed
        availP = availpower(pvg, bat, dgens, T, G, timestamp, load)
        meetL = smmeter.meet_load(timestamp, load, availP)
        Metload[idx] = meetL
        Shedload[idx] = load - meetL
        
        # Update the distribution network
        lvnet.update(timestamp)
        
        # 
        BATsoc0[idx] = bat.SOC
        if strategy == 'H': # Heuristic model
           if dieselgens == None:
               res = pvbatHeuristic(pvg, bat, load, G, T, timestamp, timestep,
                                     inv)
               # Update the components
               PVDC[idx] = pvg.DCPowerOut(G, T, timestamp)
               PVAC[idx] = inv.AC_Out(res['PVAC']/inv.efficiency, timestamp)
               BAT[idx] = bat.DispatchBat(res['BAT'], timestamp, T, timestep)
               BATsoc1[idx] = bat.SOC
           else:
               res = pvbatgenHeuristic(pvg, bat, inv, dgens, load, G, T,
                                       timestamp, timestep, hstrategy)
               # Update the components
               PVDC[idx] = pvg.DCPowerOut(G, T, timestamp)
               PVAC[idx] = inv.AC_Out(res['PVAC']/inv.efficiency, timestamp)
               # Update rectifier
               if res['BAT'] < 0:
                   if sum(res['DGENS']) > load:
                       inv.DC_Out(sum(res['DGENS']) - load, timestamp)
                   else:
                       inv.DC_Out(0, timestamp)
               else:
                    inv.DC_Out(0, timestamp)
               BAT[idx] = bat.DispatchBat(res['BAT'], timestamp, T, timestep)
               BATsoc1[idx] = bat.SOC
               DGENS[idx, :] = res['DGENS']
               for idg, dg in enumerate(dgens):
                   FUEL[idx, idg] = dg.fuel(res['DGENS'][idg], timestamp, run=True)
    
        elif strategy == 'O':
            res = optimal_dispatch(load, pvg, bat, inv, dgens, fuel_cost,
                                   batrepcost, timestep, timestamp, G, T)
            #Update the components
            PVDC[idx] = pvg.DCPowerOut(G, T, timestamp)
            PVAC[idx] = inv.AC_Out(res['PVAC']/inv.efficiency, timestamp)
            # Update rectifier
            if res['BAT'] < 0:
                if sum(res['DGENS']) > load:
                    inv.DC_Out(sum(res['DGENS']) - load, timestamp)
                else:
                    inv.DC_Out(0, timestamp)
            else:
                 inv.DC_Out(0, timestamp)
            BAT[idx] = bat.DispatchBat(res['BAT'], timestamp, T, timestep)
            BATsoc1[idx] = bat.SOC
            DGENS[idx, :] = res['DGENS']
            for idg, dg in enumerate(dgens):
                FUEL[idx, idg] = dg.fuel(res['DGENS'][idg], timestamp, run=True)
    disp_res = meteor_data
    disp_res['load'] = loads
    disp_res['PVDC'] = PVDC
    disp_res['PVAC'] = PVAC
    disp_res['BAT'] = BAT
    disp_res['BAT_SOC_0'] = BATsoc0
    disp_res['BAT_SOC_1'] = BATsoc1
    if dgens != None:
        for i, g in enumerate(dgens):
            disp_res['DGEN'+str(i+1)] = DGENS[:, i]
        for j, g in enumerate(dgens):
            disp_res['DGEN'+str(j+1)+'_fuel'] = FUEL[:, j]
    disp_res['MetLoad'] = Metload
    disp_res['UnmetDemand'] = Shedload
    output['results'] = disp_res
    output['components'] = {}
    output['components']['pvgen'] = pvg
    output['components']['battery'] = bat
    output['components']['inverter'] = inv
    output['components']['dgens'] = dgens
    output['components']['LVnetwork'] = lvnet
    output['components']['smartmeter'] = smmeter
    return output
        
def genfuel(x, dgens, timestamp):
    """
    Return fuel consumption of diesel generators over timestep
    
    """
    return np.sum([dg.fuel(x[i], timestamp) for i, dg in \
                   enumerate(dgens)])

def gendispatch(load, timestamp, timestep, dgens, minLF=0):
    """
    Evaluate optimized dispatch of diesel generators
    
    Parameters:
        load: demand to be met by diesel generators
        dgens: list of diesel generator objects
        
    Return:
        array of fuel used per generator
        array of power output per generator
    """
    limits = np.zeros([len(dgens),2])
    for idx, dg in enumerate(dgens):
        limits[idx,0] = minLF*dg.Prated
        limits[idx,1] = dg.Prated
    guess_sol = np.zeros(len(dgens))
    con = lambda x: np.sum(x) - load
    cons = {'type':'eq','fun':con}
    sol = minimize(genfuel, guess_sol, args = (dgens, timestamp), method='SLSQP',
                 bounds = limits, constraints = (cons), tol = 0.4)
    return sol.fun, sol.x


def pvbatHeuristic(pvgen, battery, load, G, T, timestamp, timestep, inverter):
    """
    Return dispatch of PV and battery
    """
    pvDC = pvgen.DCPowerOut(G, T, timestamp)
    if pvDC*inverter.efficiency >= load: # potential excess energy, charge the battery
        pvAC = inverter.AC_Out(load/inverter.efficiency, timestamp)
        pvDCtoBat = pvDC - load/inverter.efficiency
        batIn = battery.DispatchBat(-pvDCtoBat, timestamp, T, timestep)
        return {'pvDC':pvDC, 'pvAC':pvAC, 'batIn':batIn, 'batOut':0}
    else:
        pvac = pvDC*inverter.efficiency
        supplement = load - pvac
        Id = battery.Discharge(supplement/inverter.efficiency)
        batpow = Id*battery.dischargeVolt(Id)/1000
        if batpow >= supplement/inverter.efficiency:
            pvAC = inverter.AC_Out(pvDC, timestamp)
            batOut = battery.DispatchBat(supplement/inverter.efficiency, timestamp,
                                  T, timestep)
            return {'PVDC':pvDC, 'PVAC':pvAC, 'BAT':batOut}
        else:
            if pvDC > 0 and battery.SOC < 1:
                batIn = battery.DispatchBat(-pvDC, timestamp, T, timestep)
                return {'PVDC':pvDC, 'PVAC':0, 'BAT':batIn}
            else:
                return {'PVDC':pvDC, 'PVAC':0, 'BAT':0}


def pvgenHeuristic(load, pvgen, dieselgens, inverter, timestamp, timestep, G, T):
    """ 
    Return Dispatch of solar PV and Diesel generators 
    """
    pvDC = pvgen.DCPowerOut(G, T, timestamp)
    pvAC = inverter.AC_Out(pvDC, timestamp)
    if pvAC >= load:
        return {'PVDC':pvDC, 'PVAC':load, 'DGENS':np.zeros(len(dieselgens))}
    else:
        shortage = load - pvAC
        fuel, dgenOut = gendispatch(shortage, timestamp, timestep, dieselgens, 0.3)
        if sum(dgenOut) > shortage:
            pvAC = max(0, pvAC - sum(dgenOut) + shortage)
        return {'PVDC':pvDC, 'PVAC':pvAC, 'DGENS':dgenOut}


def pvbatgenHeuristic(pvgen, battery, inverter, dieselgens,
                      load, G, T, timestamp, timestep, hstrategy="load_following"):
    """
    Return dispatch of PV, battery and diesel gensets
    """
    pvDC = pvgen.DCPowerOut(G, T, timestamp)
    if hstrategy == "load_following":
        if pvDC*inverter.efficiency >= load: # potential excess energy, charge the battery
            pvAC = inverter.AC_Out(load/inverter.efficiency, timestamp)
            pvDCtoBat = pvDC - load/inverter.efficiency
            batIn = battery.DispatchBat(-pvDCtoBat, timestamp, T, timestep)
            return {'PVDC':pvDC, 'PVAC':pvAC, 'BAT':batIn, 
                    'DGENS':np.zeros(len(dieselgens))}
        else:
            pvac = pvDC*inverter.efficiency
            supplement = load - pvac
            Id = battery.Discharge(supplement/inverter.efficiency,  T, timestep)
            batpow = Id*battery.dischargeVolt(Id)/1000
            if batpow >= supplement:
                pvAC = inverter.AC_Out(pvDC, timestamp)
                batP = battery.DispatchBat(supplement/inverter.efficiency, timestamp,
                                      T, timestep)
                return {'PVDC':pvDC,'PVAC':pvAC,'BAT':batP,'DGENS':np.zeros(len(dieselgens))}
            else:
                if pvac > 0:
                    Ic = battery.Charge(-pvDC,  T, timestep)
                    maxbatin = Ic*battery.chargeVolt(Ic)/1000
                    if abs(maxbatin) > pvac/inverter.efficiency: #battery needs to be charged
                        fuel, dgenOut = gendispatch(load, timestamp, timestep, dieselgens)
                        batP = battery.DispatchBat(-pvDC, timestamp, T, timestep)
                        return {'PVDC':pvDC, 'PVAC':0, 'BAT':batP, 'DGENS':dgenOut}
                    else:
                        batIn = battery.DispatchBat(-pvDC, timestamp, T, timestep)
                        fuel, dgenOut = gendispatch(supplement, timestamp, timestep,
                                                    dieselgens)
                        pvAC = pvac - abs(batIn)
                        return {'PVDC':pvDC, 'PVAC':pvAC, 'BAT':batIn, 'DGENS':dgenOut}
                else:
                    batP = battery.DispatchBat(load, timestamp, T, timestep)
                    fuel, dgenOut = gendispatch(load-batP, timestamp, timestep, dieselgens)
                    return {'PVDC':pvDC, 'PVAC':0, 'BAT':batP, 'DGENS':dgenOut}
    else: # Cycle charging algorithm
        if pvDC*inverter.efficiency >= load: # potential excess energy
            pvAC = inverter.AC_Out(load/inverter.efficiency, timestamp)
            return {'PVDC':pvDC, 'PVAC':pvAC, 'BAT':0, 'DGENS':np.zeros(len(dieselgens))}
        else:
            pvac = pvDC*inverter.efficiency
            supplement = load - pvac
            Id = battery.Discharge(supplement/inverter.efficiency, T, timestep)
            batpow = Id*battery.dischargeVolt(Id)/1000
            if batpow >= supplement:
                pvAC = inverter.AC_Out(pvDC, timestamp)
                batP = battery.DispatchBat(supplement/inverter.efficiency, timestamp,
                                      T, timestep)
                return {'PVDC':pvDC,'PVAC':pvAC,'BAT':batP,'DGENS':np.zeros(len(dieselgens))}
            else:
                pvAC = inverter.AC_Out(pvDC, timestamp)
                dgenpleft = np.sum([dg.Prated for dg in dieselgens]) - supplement
                Ic = battery.Charge(-dgenpleft*inverter.efficiency, T, timestep)
                maxbatin = Ic*battery.chargeVolt(Ic)
                batP = battery.DispatchBat(maxbatin, timestamp, T, timestep)
                fuel, dgenOut = gendispatch(supplement + abs(batP), timestamp, timestep, dieselgens)
                return {'PVDC':pvDC,'PVAC':pvAC,'BAT':batP,'DGENS':dgenOut}
        


def optimal_dispatch(load, pvgen, battery, inverter, dieselgens, fuel_cost, 
                     batrepcost, timestep, timestamp, G, T):
    """
    Return optimal dispatch of power systems
    """
    # Variables
    # Maximum solar PV power
    pvp = pvgen.DCPowerOut(G, T, timestamp)
    
    # Battery bank
    Ic = battery.Charge(-load) # Maximum charge current
    Id = battery.Discharge(load) # Maximum discharge current
    blb = np.array(Ic*battery.chargeVolt(Ic)/1000).flatten()[0] # Max charge power
    bub = np.array(Id*battery.dischargeVolt(Id)/1000).flatten()[0] # Max discharge power
    
    # Define Constraints
    if pvp > load:
        cons = [{'type':'ineq', 'fun':lambda x: x[0] + x[1] + sum(x[2:]) - load},
                {'type':'eq', 'fun':lambda x: x[0] - pvp},
                {'type':'ineq', 'fun':lambda x: x[1] - blb},
                {'type':'ineq', 'fun':lambda x: bub - x[1]}]
        for i, gen in enumerate(dieselgens):
            cons.append({'type':'ineq', 'fun':lambda x: gen.Prated - x[2+i]})
            cons.append({'type':'ineq', 'fun':lambda x: x[2+i] - 0})
        init_x = np.zeros(2 + len(dieselgens))
        init_x[1] = blb
    else:
        cons = [{'type':'ineq', 'fun':lambda x: x[0] + x[1] + sum(x[2:]) - load},
                {'type':'eq', 'fun':lambda x: x[0] - pvp},
                {'type':'ineq', 'fun':lambda x: x[1] - blb},
                {'type':'ineq', 'fun':lambda x: bub - x[1]}]
        for i, gen in enumerate(dieselgens):
            cons.append({'type':'ineq', 'fun':lambda x: gen.Prated - x[2+i]})
            cons.append({'type':'ineq', 'fun':lambda x: x[2+i] - 0})
        init_x = np.zeros(2 + len(dieselgens))
        init_x[1] = bub
        
    # Objective 
    def cost(x):
        return battery.replacecost(x[1], timestep, T)*batrepcost \
            + gendispatch(np.sum(x[2:]), timestamp, timestep, dieselgens)[0]*fuel_cost
            
    res = minimize(cost, init_x, method='SLSQP', constraints=cons,
                   tol=0.1)
    X = res.x
    X[2:] = gendispatch(sum(X[2:]), timestamp, timestep, dieselgens)[1]
    if pvp >= load:
        pvAC = load
    else:
        if pvp >= abs(X[1]) and X[1] < 0:
            pvAC = pvp - abs(X[1])
        else:
            pvAC = pvp
    
    return  {'PVDC':X[0], 'PVAC':pvAC, 'BAT':X[1], 'DGENS':X[2:]}


def availpower(PVgen=None, Battery=None, dieselgens=None, T=0, G=0, ts=None, load=0):
    
    """ Evaluate available power """
    
    power = 0
    if PVgen != None:
        power += PVgen.DCPowerOut(G, T, ts)
    
    if Battery != None:
        Id = Battery.Discharge(load) # Maximum discharge current
        power += np.array(Id*Battery.dischargeVolt(Id)/1000).flatten()[0] # Max discharge power
        
    if dieselgens != None:
        for dgen in dieselgens:
            power += dgen.Prated
            
    return power

# Test 
if __name__=='__main__':
    
    
    import pvgen as pvg
    import dieselgen as dl
    import inverter as pwe
    import battery as bat
    import datetime as dt   
    import smartmeter as smt
    import lvnetwork as lvn
    
    installdate = dt.datetime(2020, 1, 1)
    
    loads = [max(0, 40 + np.random.normal(0, 1)) for i in range(24)]
     # Test the code
    pvgen = pvg.PVGen(35,45,9.5,42,8,40,0.0041,25,0.8,installdate)
    
    caprate = 450
    timerate = 20
    nombatV = 4
    nomV = 48
    batperstr = 6
    nstrings = 2
    lifemethod = 'capacityfade'
    SOC = 0.7
    
    storage = bat.Battery(caprate, timerate, nombatV, nomV, batperstr, nstrings,
                      lifemethod, installdate, 10, SOC=SOC)
    
    dgen1 = dl.DieselGen(10, 0.264, 0.0911, 0.2, installdate, 10000, 0)
    dgen2 = dl.DieselGen(5, 0.364, 0.011, 0.25, installdate, 12000, 0)
    dgens = [dgen1, dgen2]
    inv = pwe.Inverter(20, 10, installdate)
    smmter = smt.Smartmeter(30, [30], [1], installdate)
    lvnet = lvn.LVNetwork(30, 22, installdate, 30)
    G = np.random.randint(300, 900, size = 24)
    t = np.random.normal(25, 6, 24)
    ts = [dt.datetime(2018, 1, 1, i) for i in range(24)] # timestamps
    h = [i for i in range(24)]
    
    inputs = pd.DataFrame(data={'Time':ts, 'G':G, 'Temp':t})
    inputs1 = inputs.copy(deep=True)
    out1 = dispatch(loads, inputs1, 1, strategy = 'O', pvgen=pvgen, battery=storage,
                    dieselgens=dgens, inverter=inv, smartmeter=smmter,
                    lvnetwork=lvnet)
    
    inputs2 = inputs.copy(deep=True)
    out2 = dispatch(loads, inputs2, 1, strategy = 'H', pvgen=pvgen, battery=storage,
                    dieselgens=dgens, inverter=inv,  smartmeter=smmter, lvnetwork=lvnet)
    
    inputs3 = inputs.copy(deep=True)
    out3 = dispatch(loads, inputs3, 1, strategy = 'H', pvgen=pvgen, battery=storage,
                    dieselgens=dgens, inverter=inv, smartmeter=smmter, lvnetwork=lvnet, hstrategy="Cycle_charging")
    # # Figures
    # Optimization case
    fig, ax = plt.subplots(dpi=150)
    sns.lineplot(x='Time', y='BAT_SOC_0', data=out1['results'], label='Optimized', ax=ax)
    sns.lineplot(x='Time', y='BAT_SOC_0', data=out2['results'], label='Load follow', ax=ax)
    sns.lineplot(x='Time', y='BAT_SOC_0', data=out3['results'], label='Cycle charge', ax=ax)
    ax.set_xticklabels(ax.get_xticks(), rotation=90)
    