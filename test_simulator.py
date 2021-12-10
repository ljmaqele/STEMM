# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 03:17:41 2018

@author: lmaqelepo
"""
import time as tm
import simulate as sim
import drawdata as dd
t1 = tm.time()   

file = 'londoni.csv'
mean_load = dd.draw_load_data(file)[0]
load_stdv = dd.draw_load_data(file)[1]

sm = sim.simulate('model_parameters.xlsx', 'DLY16856.txt', mean_load, load_stdv,
                  project_country = 'Rwanda', finance_country = 'United States',
                  mct=2, inflation_data_file = 'inflation.csv')

run = sm.run()
#sm.run()
print('I took %f seconds to run' %(tm.time() - t1))