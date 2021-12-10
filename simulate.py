# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 03:41:56 2018

@author: lmaqelepo
"""

import pvgen as pvg
import dieselgen as dl
import powerelectronics as pwe
import battery as bat
import loads as lds
import meteor as met
import dispatch as disp
import datetime as dt
import numpy as np
import pandas as pd
import finance as fin
import utils
import smartmeter as sm


class Simulator:
    """
        Simulator class
        
    """
    
    def __init__(self, params_filename):
       
        self.params = pd.read_excel(params_filename, sheet_name='allparams')
        self.dgenparams = pd.read_excel(params_filename, sheet_name='gensets')
        self.smparams = pd.read_excel(params_filename, sheet_name= 'smartmeter')
        
    def create_solar_inputs(self):
        paramlist = self.get_params('meteor')
        values = self.get_values('meteor')
        gen = self.general_params()
        global solarinputs
        kargs = dict(zip(paramlist, values))
        kargs['start_time'] = gen['start']
        kargs['horizon'] = gen['horizon']
        solarinputs = met.SolarInputs(**kargs)
        
    def general_params(self):
        #genparams = self.get_params('general')
        genvalues = self.get_values('general')
        yy,mm,dd = genvalues[1].split(',')
        horizon = genvalues[0]
        startdate = dt.datetime(int(yy), int(mm), int(dd))
        return {'start':startdate, 'horizon':horizon}
        
    def get_values(self, technology):
        return self.params['param_default_value'][self.params.technology == technology].values
    
    def get_params(self, technology):
        return self.params['param_name'][self.params.technology == technology].values
        
    def create_loads_object(self):
        global loads
        loads = lds.Loads(self.mean_load, self.load_stdv, 'Lognormal')
        
    def create_pvgen_object(self):
        paramlist = self.get_params('solar')
        values = self.get_values('solar')
        gen = self.general_params()
        global pvgen
        kargs = dict(zip(paramlist, values))
        kargs['installdate'] = gen['start']
        pvgen = pvg.PVGen(**kargs)    
    
    def create_inverter_object(self):
        """
        Create inverter object
        """
        paramlist = self.get_params('inveter')
        values = self.get_values('inverter')
        gen = self.general_params()
        global inverter
        kargs = dict(zip(paramlist, values))
        kargs['installdate'] =  gen['start']
        inverter = pwe.Inverter(**kargs)     
    
    def create_rectifier_object(self):
        """
        Create rectifier object
        """
        paramlist = self.get_params('rectifier')
        values = self.get_values('rectifier')
        gen = self.general_params()
        global rectifier
        kargs = dict(zip(paramlist, values))
        kargs['installdate'] =  gen['start']
        rectifier = pwe.Rectifier(**kargs)    
    
    def create_battery_object(self):
        """
        Create battery storage object
        """
        paramlist = self.get_params('battery')
        values = self.get_values('battery')
        gen = self.general_params()
        global battery
        kargs = dict(zip(paramlist, values))
        kargs['installdate'] =  gen['start']
        battery = bat.Battery(**kargs)     
        
    def create_diesel_objects(self):
        """
        Create diesel generator objects
        """
        paramlist = self.dgenparams.param_name.values
        values = self.dgenparams.drop(columns=['ID', 'technology', 'param_name', 'param_label'])
        values = values.to_numpy()
        gen = self.general_params()
        N = values.shape[1]
        global dgens 
        dgs = []
        for i in range(N):
            kargs = dict(zip(paramlist, values[:, i]))
            kargs['installdate'] = gen['start']
            dgs.append(dl.DieselGen(**kargs))
        dgens = dgs 
        print(dgens)
    
    def create_smartmeter_object(self):
        """
        """
        paramlist = self.paras['param_name']['smartmeter'].get_values()
        values = self.paras['param_default_value']['smartmeter'].get_values()
        global smartmeter
        kargs = dict(zip(paramlist, values))
        sm_file_name = \
                    self.paras['param_default_value']['smartmeterfile']
        name = sm_file_name + '.xlsx'
        sm_df = pd.read_excel(name)
        kargs['customers_per_tier'] = sm_df['customers_per_tier'].get_values()
        kargs['price_per_tier'] = sm_df['price_per_tier'].get_values()
        smartmeter = sm.Smartmeter(**kargs)
        
    def create_financial_object(self):
        """
        Create Cashflows object
        """
        paramlist = self.paras['param_name']['financial'].get_values()
        values = self.paras['param_default_value']['financial'].get_values()
        global fn
        args = {}
        aslists = ['I0', 'QMU', 'QA', 'QV', 'fixed_cost', 'init_fuel_price']
        for idx, param in enumerate(paramlist):
            if param not in aslists:
                args[param] = values[idx]
            else:
                aslist = values[idx].split(",")
                args[param] = aslist
        args['pvgen'] = pvgen
        args['battery'] = battery
        args['inverter'] = inverter
        args['rectifier'] = rectifier
        args['dieselgens'] = dgens
        args['st'] =  self.set_start_date()
        fn = fin.CashFlows(**args)    
    
    

            
        
    """
    UTILITY FUNCTIONS
    """
    def set_start_date(self):
        paramlist = self.paras['param_default_value']['startdate'].get_values()
        global start_date
        start_date = dt.datetime(int(paramlist[2]), int(paramlist[1]),
                                 int(paramlist[0]))
        return start_date
    
    def get_power_fuel_vals(self, filename):
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)
        try:
            [row, col] = data.shape
            pvals = data[0:row//2,:]
            fvals = data[row//2:,:]
        except:
            row = len(data)
            pvals = data[0:row//2]
            fvals = data[row//2:]
        return pvals, fvals
    
    def genpower_out_list(self, genlist, genoutlist):
        """
        Return timeseries power outputs of respective diesel gensets
        
        Parameters:
            genlist: list of DieselGen objects
            genoutlist: list of gensets power output over entire
        """
        if len(genoutlist) % len(genlist) != 0:
            raise Exception('Cannot reshape list of %d into %d  equal vectors'\
                            %(len(genoutlist), len(genlist)))
        cols = len(genlist)
        rows = len(genoutlist)//cols
        return np.reshape(genoutlist, (rows,cols)) 
    
if __name__=='__main__':
    
    sim = Simulator('model_parameters.xlsx')
    
    # Create Objects
    sim.create_solar_inputs()
    
    # def run(output):
    #     np.random.seed()
    #     sim.set_start_date()
    #     sim.timelines()
    #     num_run = sim.mct//sim.cores
    #     for i in range(num_run):
    #         sim.create_loads_object()
    #         sim.create_battery_object()
    #         sim.create_pvgen_object()
    #         sim.create_inverter_object()
    #         sim.create_rectifier_object()
    #         sim.create_diesel_objects()
    #         sim.create_smartmeter_object()
    #         sim.module_incident_insolation()
    #         sim.run_dispatch()
    #         sim.create_financial_object()
    #         out = fn.runcashflows(disp_DF, 1)
    #         output.append(out)
    #     return out
    
    
    # output = []
    # Ps = [mp.Process(target=run, args = (output, )) for _ in range(sim.cores)]
    
    # for p in Ps:
    #     p.start()
        
    # for p in Ps:
    #     p.join()
        
    
    
