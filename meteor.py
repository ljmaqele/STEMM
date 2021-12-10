#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import requests
import calendar
import pandas as pd
import numpy as np
import datetime as dt
import pickle
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, KFold


fi = open('markov.pkl', 'rb')
markovdata = pickle.load(fi)
fi.close()

class SolarInputs:
    
    def __init__(self, latitude, longitude, start_time, horizon, markovdata=markovdata):
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.start_time = start_time
        self.horizon = int(horizon)
        self.end_time = dt.datetime(start_time.year + horizon, start_time.month,
                                    start_time.day) - dt.timedelta(hours=1)
        self.month_times = pd.date_range(start_time, periods=horizon*12, freq='M')
        self.day_times = pd.date_range(start_time, self.end_time, freq='D')
        self.hour_times = pd.date_range(self.start_time, self.end_time, freq='H')
        self.markovmats = markovdata['markovmats']
        self.ktlims = markovdata['Ktlims']
        self.ktmonth = markovdata['Ktmonth']
        
    
    def evalInputs(self):
        """
        Evaluate the hourly resolution meteorological inputs
        """
        h = self.getH() # Get average monthly insolation data from API
        T = self.getTemp() # Get daily maximum and minimum temperature from API
        kde = self.tempKDE(T) # Train maximum and minimum temperature predicition model
        Ts = self.evalTemp(kde) # Generate temperatures using the model
        mKt = self.monthlyKt(h) # Generate monthly clearness index values
        dKt = self.dailyktsim(mKt) # Generate daily clearness index values
        hKt = self.hourlyktsim(list(dKt.index), dKt.Kt) # Generate hourly clearness and insolation
        # Generate hourly 
        hKt['T'] = self.hourlyT(hKt.G.values, Ts.Tmin.values, Ts.Tmax.values, list(hKt.index))
        return hKt
    
    def getH(self):
        """
        Get monthly mean radiation on horizontal surface 
        """
        _str = "https://power.larc.nasa.gov/cgi-bin/v1/DataAccess.py?&request"
        _str = _str + "=execute&identifier=SinglePoint&parameters=ALLSKY_SFC_"
        _str = _str + "SW_DWN&userCommunity=SSE&tempAverage=CLIMATOLOGY&outputList"
        _str = _str + "=JSON&lat={}&lon={}".format(self.latitude, self.longitude)
        
        data = requests.get(_str).json()['features'][0]['properties']['parameter']\
            ['ALLSKY_SFC_SW_DWN']
        data.pop('13')
        return pd.DataFrame.from_dict(data, orient='index', columns=['H'])
    
    def getTemp(self):
        """ Get temperature data """
        _str = "https://power.larc.nasa.gov/cgi-bin/v1/DataAccess.py?&request"
        _str = _str + "=execute&identifier=SinglePoint&parameters=T2M_MAX,T2M_"
        _str = _str + "MIN&startDate=19830101&endDate=20150305&userCommunity"
        _str = _str + "=SSE&tempAverage=DAILY&outputList"
        _str = _str + "=JSON&lat={}&lon={}".format(self.latitude, self.longitude)
        
        data = requests.get(_str).json()['features'][0]['properties']['parameter']
        data = pd.DataFrame(data)
        data['mon'] = list(map(lambda s: s[4:6], data.index.values))
        return data
    
    def tempKDE(self, data):
        """ 
        Learn kernel density estimators for T2M_MAX and T2M_MIN for 
        every month
        """
        KDE = {}
        bandwidths = 10**np.linspace(-1, 1, 30)
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), 
                        {'bandwidth':bandwidths},
                        cv=KFold(n_splits=5))
        for mon in data.mon.unique():
            temp = data[data.mon == mon]
            X = temp.to_numpy()[:, 0:2]
            grid.fit(X)
            h = grid.best_params_['bandwidth']
            KDE[int(mon)] = KernelDensity(bandwidth=h, kernel='gaussian').fit(X)
        return KDE
        
    def evalTemp(self, KDE, randstate=32):
        """ 
        Evaluate daily maximum and minimum temperatures of every day of the 
        month for the entire model horizon
        """
        data = pd.DataFrame()
        for timestamp in self.month_times:
            mdays = calendar.monthrange(timestamp.year, timestamp.month)[1]
            t0 = timestamp - dt.timedelta(days=timestamp.day - 1)\
                - dt.timedelta(hours=timestamp.hour)
            tf = timestamp + dt.timedelta(days = mdays - timestamp.day)\
                - dt.timedelta(hours=timestamp.hour)
            index = pd.date_range(t0, tf, periods=mdays)
            _data = KDE[timestamp.month].sample(mdays)
            if len(data) == 0:
                data = pd.DataFrame(_data, index=index, columns=['Tmax', 'Tmin'])
            else:
                data = pd.concat([data, pd.DataFrame(_data, index=index, columns=['Tmax', 'Tmin'])])
        return data
    
    def declination(self, timestamp):
        '''
        Compute declination for date in timestamp
        '''
        N_day = timestamp.timetuple().tm_yday # find day number in year
        return 0.40928*np.sin(2*np.pi*(N_day-80)/365) # return declination in radians
    
    def hss(self, timestamp):
        '''
        Compute sunrise/sunset hour angle at timestamp for given latitude
        '''
        lat = np.deg2rad(self.latitude) # convert latitude from degrees to radians
        return np.arccos(-np.sin(lat)*np.sin(self.declination(timestamp))/(np.cos(lat)
                                 *np.cos(self.declination(timestamp)))) # return sunset hour angle
    
    def ET_DNI(self, timestamp):
        '''
        Compute extraterrestrial direct normal irradiation at timestamp
        '''
        N_day = timestamp.timetuple().tm_yday # find day number in year
        return (1 + 0.033*np.cos(2*np.pi*N_day/365))*1.367 # return ET DNI
    
    def DET_DNI(self, timestamp):
        '''
        Compute daily extraterrestrial insolation on horizontal surface for day in
        timestamp at given latitude in kWh/sqm
        '''
        lat = np.deg2rad(self.latitude) # convert latitude from degrees to radians
        res =  24/np.pi*self.ET_DNI(timestamp)*(np.sin(lat) \
                        *np.sin(self.declination(timestamp))\
                        *self.hss(timestamp) \
                        + np.cos(lat)*np.cos(self.declination(timestamp)) \
                        *np.sin(self.hss(timestamp))) # return daily ET insolation
        return res
            
    def MMET_DNI(self, timestamp):
        '''
        Compute monthly mean daily extraterrestrial insolation on horizontal surface
        for month in timestamp at given latitude
        '''
        mdays = calendar.monthrange(timestamp.year, timestamp.month)[1]
        t0 = timestamp - dt.timedelta(days=timestamp.day - 1)\
               - dt.timedelta(hours=timestamp.hour)
        tf = timestamp + dt.timedelta(days = mdays - timestamp.day)\
            - dt.timedelta(hours=timestamp.hour)
        index = pd.date_range(t0, tf, periods=mdays)
        mmET = np.sum(list(map(self.DET_DNI, list(index))))
        return mmET/mdays # return mean daily insolation for month
    
    def TdeltaETinsolation(self, starttime, endtime):
        '''
        Compute extraterrestrial insolation on horizontal surface between starttime
        and endtime at given latitude in kWh/sqm
        '''
        lat = np.deg2rad(self.latitude) # convert latitude from degrees to radians
        midtime = starttime + (endtime - starttime)/2 # find midpoint in time interval for compuation of declination
        hstart = self.hangle(starttime) # find hour angle at start time
        hend = self.hangle(endtime) # find hour angle at end time
        return 12/np.pi*self.ET_DNI(midtime)*(np.sin(lat)*np.sin(self.declination(midtime))*(hend - hstart) \
                        + np.cos(lat)*np.cos(self.declination(midtime))*(np.sin(hend)-np.sin(hstart)))
    
    def ETirradiation(self, timestamp):
        '''
        Compute extraterrestrial irradiation on horizontal surface at timestamp for
        given latitude in kW/sqm
        '''
        return self.ET_DNI(timestamp)*np.sin(self.solar_alt(timestamp))
    
    def monthlyKt(self, H_data):
        """
        Evaluate mean monthly clearness index for all months over modelling horizon
        """
        t0 = dt.datetime(self.start_time.year, 1, 1)
        yrtms = pd.date_range(start=t0, periods=12, freq='M')
        monthlymeanH = list(map(self.MMET_DNI, list(yrtms)))
        def getHData(timestamp):
            return H_data.loc[str(timestamp.month)].values[0]
        Have = list(map(getHData, list(yrtms)))
        Ktdata = pd.DataFrame(data={'H_ave':Have, 'Ho_ave':monthlymeanH}, 
                              index=yrtms)
        Ktdata['Kt'] = Ktdata['H_ave'].values/Ktdata['Ho_ave'].values
        return Ktdata
    
    def dailyktsim(self, monthlyKt):
        """
        Evaluate daily clearness index over the time horizon
        """
        # Intialize results
        tsidx = []
        ktsim = []
        Hsim = []
        ts = self.start_time # initialize timestep counter
        Kt = monthlyKt.Kt.values
        kt0 = Kt[((ts - dt.timedelta(days=1)).month - 1)] # initialize kt to Kt for month of day prior to startday
        while ts <= self.end_time: # loop over days until endday is reached
            tsidx.append(ts) # add day timestamp to list
            kt0 = self.markovsim(ts, Kt[ts.month - 1], kt0) # simulate daiy clearness index
            ktsim.append(kt0) # add kt to list
            Hsim.append(kt0*self.DET_DNI(ts)) # compute corresponding insolation and add to list
            ts = ts + dt.timedelta(days=1) # increment day
        return pd.DataFrame(data={'Kt':np.array(ktsim), 'H':np.array(Hsim)}, index=np.array(tsidx)) 
    
    
    def hourlyktsim(self, timestamps, Kts, result = 'G'):
        """
        Simulate hourly clearness index and irradiation (results='G') or
        insolation (result='H') for day in timestamp at given latitude with daily
        clearness index Kt
        """
        kth = [] # initialize list of hourly clearness index
        Hh = [] # initialize list of hourly insolation
        tsidx = [] # initialize list of time indices
        if result not in ['G', 'H']: # check if result option is valid
            raise Exception('Invalid input for result')
        for i, Kt in enumerate(Kts):
            timestamp = timestamps[i]
            phi = 0.38 + 0.06*np.cos(7.4*Kt-2.5) # compute model parameters
            lda = -0.19 + 1.12*Kt + 0.24*np.exp(-8*Kt)
            eps = 0.32 - 1.60*(Kt - 0.5)**2
            kap = 0.19 + 2.27*Kt**2 - 2.51*Kt**3
            A = 0.14*np.exp(-20*(Kt - 0.35)**2)
            B = 3*(Kt - 0.45)**2 + 16*Kt**5
            h = np.array([[i, (i-13)*np.pi/12, (i-12.5)*np.pi/12, (i-12)*np.pi/12]\
                           for i in np.arange(0,24,1)])
            hss0 = self.hss(timestamp) # compute sunset hour angle
            hssh, hssm, hsss = self.ha2hr(hss0) # compute sunset  hours, minutes, and seconds
            hsrh, hsrm, hsrs = self.ha2hr(-hss0) # compute sunrise hours, minutes and seconds
            # compute timestamps for sunrise and sunset
            tss = dt.datetime(timestamp.year, timestamp.month, timestamp.day, hssh,
                              hssm, hsss)
            tsr = dt.datetime(timestamp.year, timestamp.month, timestamp.day, hsrh,
                              hsrm, hsrs)
            y = 0 # initialize y to zero
           
            for ha in h: # loop over hours in day
                ts = dt.datetime(timestamp.year, timestamp.month, timestamp.day, 
                                 int(ha[0]), 0) # timestamp at end of timestep
                tsstart = ts - dt.timedelta(hours=1) # timestamp at start of timestep
                tsmid = ts - dt.timedelta(minutes=30) # timestamp at center of timestep
                if (ha[3] < -hss0) or (ha[1] > hss0): # if timestep is after dark, set kth and Hh to zero
                    tsidx.append(ts)
                    y = 0
                    kth.append(0)
                    Hh.append(0)
                else:
                    ts = dt.datetime(timestamp.year, timestamp.month, timestamp.day, 
                                     int(ha[0]), 0) # timestamp at end of timestep
                    tsidx.append(ts)
                    alt = self.solar_alt(tsmid) # compute solar altitude
                    kcs = 0.88*np.cos(np.pi*(ha[0]-12.5)/30) # compute clear sky radiation
                    if result=='G': # if basing calculation on irradiation
                        if alt < 0: # if altitude at end of timestep is negative, set kth and Hh to zero
                            kth.append(0)
                            Hh.append(0)
                        else: # otherwise, compute kth and Hh
                            ktm = lda + eps*np.exp(-kap/np.sin(alt))
                            std = A*np.exp(B*(1 - np.sin(alt)))
                            stdp = std*(1 - phi**2)**(1/2)
                            draw = np.random.normal(0, stdp) # draw random number from normal distribution
                            y = phi*y + draw # compute new y
                            kt = ktm + y*std # convert to kt
                            kth.append(kt)
                            Hh.append(kt*self.ETirradiation(tsmid)) # compute Hh
                    elif result=='H': # if basing calculation on insolation
                        if alt < 0: # if altitude at end of timestep is negative, compute insolation from sunrise/sunset to timestep limit
                            ktm = lda
                            std = A*np.exp(B)
                            stdp = std*(1 - phi**2)**(1/2)
                            draw = np.random.normal(0, stdp)
                            y = phi*y + draw
                            kt = ktm + y*std
                            kth.append(kt)
                            H = self.TdeltaETinsolation(np.max((tsr, tsstart)), 
                                                   np.min((tss, ts)))
                            Hh.append(kt*H)
                        else: # otherwise compute based on insolation for entire timestep
                            ktm = lda + eps*np.exp(-kap/np.sin(alt))
                            std = A*np.exp(B*(1 - np.sin(alt)))
                            stdp = std*(1 - phi**2)**(1/2)
                            draw = np.random.normal(0, stdp)
                            y = phi*y + draw
                            kt = ktm + y*std
                            kth.append(kt)
                            H = self.TdeltaETinsolation(np.max((tsr, tsstart)), 
                                                   np.min((tss, ts)))
                            Hh.append(kt*H)
        return pd.DataFrame(data={'Kt':np.array(kth), 'G':np.array(Hh)*1000}, index = np.array(tsidx))
    
    def hourlyT(self, Hh, Tmin, Tmax, tsidx):
        '''
        Compute hourly temperature based on hourly insolation and daily maximum
        and minimum temperature at given latitude indexed by time index tsidx
        '''
        if (len(Hh) != len(Tmin)*24) or (len(Hh) != len(Tmax)*24):
            raise Exception('Lengths len(Hh) =: %d, & len(Tmin) =: %d, do not \
                            match, Tmin and Tmax required for each 24 Hh datapoints\
                            '%(len(Hh), len(Tmin)*24))
        Hh24 = Hh.reshape((int(len(Hh)/24),24)) # reshape Hh array to Ndays X 24
        kxNum = Hh24.cumsum(axis=1) # numerator of kx is cumulative sum of hourly insolation
        # compute daily vector of sunrise hours
        _hsrvec = np.array(list(map(self.hss, tsidx[0::24])))*-1
        hsrvec = np.array(list(map(self.ha2hrdec, _hsrvec))) 
        # compute daily vector of sunset hours
        _hssvec = np.array(list(map(self.hss, tsidx[0::24])))
        hssvec = np.array(list(map(self.ha2hrdec, _hssvec))) 
        hrmat = np.mgrid[0:len(Tmax), 1:25][1] # create matrix of end of timestep hours
        morning = (hrmat.T - hsrvec).T # create matrix of timestep hours minus sunrise hours
        night = 1 - (hrmat.T - hssvec).T # create matrix of timestep hours minus sunset hours
        kxDen = np.concatenate((morning[:,:12],night[:,12:]),axis=1) # merge morning hours of morning matrix and after morning hours for night matrix
        kxDen[kxDen >= 1] = 1 # if kxDen is greater than or equal to one, the entire hour has sunlight
        kxDen[kxDen <= 0] = 0 # if kxDen is less than equal to zero, the entire hour has no sunlight
        sunhours = kxDen.copy() # make copy of sunhours, partial hours at sunrise/sunset represented by fraction of time with sunlight
        kxDen = ((kxDen.cumsum(axis=1).T)*np.array(list(map(self.ET_DNI, tsidx[0::24])))).T # multiply cumulative sum of sunlight hours by ET DNI
        kx = np.divide(kxNum, kxDen, where=kxDen!=0) # divide kx numerator by kx denomenator, leave as zero when denomenator is zero
        kxmax = np.max(kx, axis=1) # find maximum kx during each day
        tkxmax = np.argmax(kx, axis=1) + 1 # find timestep hour when kx is at maximum
        slpb = (Tmax - Tmin)/kxmax # compute daily temperature slope before tkxmax
        slpa = 1.7*slpb # compute daily temperature slope after tkxmax
        morn_filt = morning < 0 # create mask for morning hours
        night_filt = night < 1 # create mask for night hours 
        before_filt = (~morn_filt)*((hrmat.T <= tkxmax).T)*(~night_filt) # create mask for hours when slpb applies
        after_filt = (~morn_filt)*(~night_filt)*(~before_filt) # create mask for hours when slpa applies
        Thourly = (((kx.T)*slpb + Tmin).T)*before_filt # compute hourly temperature before tkxmax
        Thourly = Thourly + ((Tmax - (kxmax - kx.T)*slpa).T)*after_filt # compute hourly temperature after tkxmax
        Tss = np.choose(np.argmax((Thourly > 0)*hrmat,axis=1), Thourly.T) # compute sunset temperature
        nslp = (Tss - np.append(Tmin[1:],Tmin[0])) \
                /(24 + np.append(hsrvec[1:],hsrvec[0]) - hssvec) # compute night time temperature slope
        mslp = np.append(nslp[-1], nslp[:-1]) # compute morning temperature slope
        Thourly = Thourly + ((Tmin - (hrmat.T - hsrvec)*mslp).T)*morn_filt # compute morning hourly temperature
        Thourly = Thourly + ((Tss - (hrmat.T - hssvec)*nslp).T)*night_filt # compute night time hourly temperature
        return Thourly.flatten() # return flattened array of hourly temperature
        
    
    
    def rowselector(self, Kt, kt0):
        '''
        Select Markov matrix and row of selected Markov matrix based on monthly
        clearness index Kt and previous day's clearness index kt0
        '''
        Ktidx = np.argmax((Kt > self.ktmonth[:,0])*(Kt <= self.ktmonth[:,1])) # find index of Kt transition matrix
        ktmin = self.ktlims[Ktidx,0] # find lower kt limit for selected transition matrix
        ktmax = self.ktlims[Ktidx,1] # find upper kt limit for selected transition matrix
        step = (ktmax - ktmin)/10 # compute kt step in transition matrix
        ktlow = np.arange(ktmin, ktmax, step) # create array of row lower limits
        kthigh = ktlow + step # create array of row upper limits
        ktidx = np.argmax((kt0 > ktlow)*(kt0 <= kthigh)) # select row in transition matrix
        return (Ktidx, ktidx) # return tuple with transition matrix index and row index
    

    def markovsim(self, timestamp, Kt, kt0):
        '''
        Simulate daily Kt using Markov model with monthly clearness index
        Kt and previous day clearness index kt0
        '''
        Ktidx, ktidx = self.rowselector(Kt, kt0) # find transition matrix and row indices
        ktmin = self.ktlims[Ktidx,0] # find kt lower limit
        ktmax = self.ktlims[Ktidx,1] # find kt upper limit
        step = (ktmax - ktmin)/10 # calculate kt step size
        krange = np.arange(ktmin, ktmax + step, step) # define array with transition matrix row kt boundaries
        row = self.markovmats[Ktidx, ktidx] # select transition matrix and corresponding row
        rowsum = np.insert(np.cumsum(row), 0, 0) # compute cumulative probabilities across row
        draw = np.random.rand() # draw random number between 0 and 1
        return np.interp(draw, rowsum, krange) # construct cumulative distribution
                                               # function and return new kt value
                                               # corresponding to random draw
        
    def hangle(self, timestamp):
        '''
        Compute hour angle from timestamp
        '''
        return np.pi/12*(timestamp.hour + timestamp.minute/60
                         + timestamp.second/3600 - 12) # compute hour angle

    def solar_alt(self, timestamp):
        '''
        Compute solar altitude at timestamp for given latitude (in degrees)
        '''
        lat = np.deg2rad(self.latitude) # convert latitude from degrees to radians
        h = self.hangle(timestamp) # compute hour angle
        return np.arcsin(np.sin(lat)*np.sin(self.declination(timestamp))+np.cos(lat)*
                         np.cos(self.declination(timestamp))*np.cos(h)) # return solar altitude in radians

    def ha2hr(self, ha):
        '''
        Convert hour angle to hour, minutes, seconds
        '''
        hourdec = ha*12/np.pi+12 # compute decimal hour
        hour = np.floor(hourdec) # find integer hour
        minute = np.floor((hourdec - hour)*60) # find integer minutes
        second = np.floor((hourdec - hour - minute/60)*3600) # find rounded integer seconds
        return (int(hour), int(minute), int(second)) # return tuples with hours, minutes and seconds

    def ha2hrdec(self, ha):
        h, m, s = self.ha2hr(ha) # convert hour angle to hours minutes seconds
        return h + m/60 + s/3600 # compute decimal hour
    
    def __str__(self):
        _str = ''
        _str += "Latitude: {}\n"
        _str += "Longitude: {}\n"
        _str += "Start date: {}\n"
        _str += "Horizon: {}"
        _str = _str.format(self.latitude, self.longitude, self.start_time, self.horizon)
        return _str
    
    def __repr__(self):
        _str = 'SolarInputs({}, {}, {}, {})'
        _str = _str.format(self.latitude, self.longitude, self.start_time, self.horizon)
        return _str
    
if __name__=='__main__':
    
    date = dt.datetime(2020, 3, 20)
    si = SolarInputs(-29.066118, 27.830466, date, 30)
    #inputs = si.evalInputs()
    # h = si.getH()
    # x = si.getTemp()
    # kde = si.tempKDE(x)
    # sample = si.evalTemp(kde)
    # monthlyKt = si.monthlyKt(h)
    # dailyKt = si.dailyktsim(monthlyKt)
    # hourlyKt = si.hourlyktsim(list(dailyKt.index), dailyKt.Kt)
    # houtlyT = si.hourlyT(hourlyKt.Hh.values, sample.Tmin.values, sample.Tmax.values, list(hourlyKt.index))
    