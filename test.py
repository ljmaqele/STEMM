#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pvgen as pvg
import battery as bat
import inverter as pwe
import dieselgen as dg
import dispatch as dsp
import smartmeter as smt
import lvnetwork as lvn
import meteor
import loads as ld
import finance as fn
import datetime as dt
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Start operations
t1 = time.perf_counter()

starttime = dt.datetime(2020, 1, 1)
horizon = 5

# Create Loads
mean = np.array([[40, 50], [43, 53], [44, 54], [48, 58], [50, 60], [53, 63],
                 [54, 64], [44, 56], [42, 54], [39, 43], [45, 55], [45, 59]])
stddev = mean/6
basep = 1.75
targetp = 1.75
loads = ld.Loads(mean, stddev, basep, targetp, starttime, horizon)
hourlyL = loads.houlyloadsim().load.values

# Create PVGEN
pvgen = pvg.PVGen(10,45,9.5,42,8,40,0.0041,25,0.8,starttime)

# Create battery
caprate = 450
timerate = 20
nombatV = 4
nomV = 24
batperstr = 6
nstrings = 2
lifemethod = 'capacityfade'
life = 5
SOC = 1

storage = bat.Battery(caprate, timerate, nombatV, nomV, batperstr, nstrings,
                  lifemethod, starttime, life, SOC=SOC)

# Create Diesel gensets
dgen1 = dg.DieselGen(8, 0.264, 0.0911, 0.2, starttime, 10000, 0)
dgen2 = dg.DieselGen(6, 0.364, 0.011, 0.25, starttime, 12000, 0)
dgen3 = dg.DieselGen(6, 0.464, 0.001, 0.25, starttime, 14000, 0)
dgens = [dgen1, dgen2]

# Create power electronics
inv = pwe.Inverter(5, 10, starttime)
#rec = pwe.Rectifier(5, 10, starttime)

# Create meters and network
smmter = smt.Smartmeter(30, [30], [1], starttime)
lvnet = lvn.LVNetwork(30, 22, starttime, 30)
    
t2 = time.perf_counter()
print('Start solar inputs after {} seconds'.format(t2 - t1))

# Solar inputs
si = meteor.SolarInputs(-29.066118, 27.830466, starttime, horizon)
inputs = si.evalInputs()

t3 = time.perf_counter()
print('Took {} seconds to finish solar inputs'.format(t3 - t2))

# Combine inputs
inputs['Time'] = list(inputs.index)
inputs['Temp'] = inputs['T'].values

t4 = time.perf_counter()

inputs1 = inputs.copy(deep=True)
out1 = dsp.dispatch(hourlyL, inputs1, 1, strategy = 'O', pvgen=pvgen, battery=storage,
                dieselgens=dgens, inverter=inv, rectifier=rec)

t5 = time.perf_counter()
print('Took {} seconds to run dispatch for {} with {} year(s) horizon'.format(t5-t4, 
                                                                              'Optimization',
                                                                              horizon))

inputs2 = inputs.copy(deep=True)
out2 = dsp.dispatch(hourlyL, inputs2, 1, strategy = 'H', pvgen=pvgen, battery=storage,
                dieselgens=dgens, inverter=inv, rectifier=rec,  smartmeter=smmter,
                    lvnetwork=lvnet)

t6 = time.perf_counter()
print('Took {} seconds to run dispatch for {} with {} year(s) horizon'.format(t6-t5, 
                                                                              'Hueristic-load follow',
                                                                              horizon))

inputs3 = inputs.copy(deep=True)
out3 = dsp.dispatch(hourlyL, inputs3, 1, strategy = 'H', pvgen=pvgen, battery=storage,
                dieselgens=dgens, inverter=inv, rectifier=rec, hstrategy="Cycle_charging")

t7 = time.perf_counter()
print('Took {} seconds to run dispatch for {} with {} year(s) horizon'.format(t7-t6, 
                                                                              'Hueristic-cycle charge',
                                                                              horizon))

# pvcost = 1200
# batcost = 500 
# invcost = 800 
# rectcost = 800
# dgcost = 400
# debt_ratio = 0.75
# project_c = 'Rwanda'
# finance_c = 'United States'
# fixedOM = 1
# kWhtariff = 1
# monthlycharge = 1
# connectionfee = 10
# Ncustomers = 50
# metercost = 40
# LVdistributioncost = 92
# initfuelp = 2
# fueldrift = 0.03
# fuelvol = 0.14
# fuelconsum = 0.3
# transittime = 3
# fuelvolume = 200
# capexsub = 0.3
# capexsubtype = 'All Capex'
# cashgrant = 4000
# tariffsub = 0.3
# fuelsub = 0.2
# opexsub = 0.25

# fin = fn.Finance(out2, starttime, horizon, project_c, finance_c, fixedOM, 
#                  kWhtariff, monthlycharge, connectionfee, Ncustomers, 
#                  metercost, LVdistributioncost, initfuelp,
#                  fueldrift, fuelvol, fuelconsum, transittime, fuelvolume, debt_ratio, 
#                  pvcapex=pvcost, batcapex=batcost, capexsub=capexsub, capexsubtype=capexsubtype,
#                  invcapex=invcost, rectcapex=rectcost, dgencapex=dgcost,
#                  cashgrant=cashgrant, tariffsub=tariffsub, fuelsub=fuelsub,
#                  opexsub=opexsub)

# capital = fin.capital()
# deprec = fin.depreciation()
# expenses = fin.expenses()
# revenues = fin.revenues()
# cashflows = fin.cashflows()
# tf = starttime + dt.timedelta(hours=24)

# # For Visualizations
# v1 = out1['results'][out1['results'].Time < tf]
# v2 = out2['results'][out2['results'].Time < tf]
# v3 = out3['results'][out3['results'].Time < tf]

# # Visualize battery SOC
# fig, (ax0, ax1, ax2) = plt.subplots(3,1, figsize=(10, 18))
# ax0.plot('Time', 'BAT_SOC_0', data=v1, label='Optimal')
# ax0.plot('Time', 'BAT_SOC_0', data=v2, label='H-load follow')
# ax0.plot('Time', 'BAT_SOC_0', data=v3, label='H-cycle charge')
# ax0.get_xaxis().set_visible(False)

# ax1.step('Time', 'BAT', where='mid', data=v1, label='Optimal')
# ax1.step('Time', 'BAT', where='mid', data=v2, label='H-load follow')
# ax1.step('Time', 'BAT', where='mid', data=v3, label='H-cycle charge')
# ax1.step('Time', 'load', where='mid', data=v1, ls=':', label='Load')
# ax1.step('Time', 'PVDC', where='mid', data=v1, ls='--', label='PVDC')
# ax1.step('Time', 'PVAC', where='mid', data=v1, ls='-.', label='PVAC-opt')
# ax1.step('Time', 'PVAC', where='mid', data=v2, ls='-.', label='PVAC-lf')
# ax1.step('Time', 'PVAC', where='mid', data=v3, ls='-.', label='PVAC-cc')


def carriedloss(ebt):
    loss = np.zeros(len(ebt))
    for idx, eb in enumerate(ebt):
        if idx == 0:
            if eb < 0:
                loss[idx] = eb
        else:
            if loss[idx - 1] <= 0:
                temp = loss[idx - 1] + eb
                if temp < 0:
                    loss[idx] = temp
    return loss
                

        
    