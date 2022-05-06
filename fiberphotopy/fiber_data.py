import numpy as np
import pandas as pd
import random
import time
import scipy.signal as signal
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from fp_utils import FiberPhotopy, timer


class FiberData(FiberPhotopy):
    """Extract fiberphotometry data from Doric system."""
    vars().update(FiberPhotopy.FIBER)
    
    def __init__(self,
                 filepath,
                 name      = 'default',
                 rat_ID    = None,
                 alignement=0,
                autoinherit=False,
                 **kwargs):
        start = time.time()
        super().__init__(**kwargs)
        self.alignement = alignement
        self.filepath = filepath
        self._print(f'Importing {filepath}...')
        self.rat_ID = rat_ID
        #part1 = timer(start,'part1')
        if name == 'default':
            self.name = self.filepath.split('/')[-1].split('.csv')[0]
            if self.rat_ID:
                self.name += rat_ID
        self.number_of_recording = 0
        #part2 = timer(part1,'adding names')
        self.df = self._read_file(filepath,alignement=alignement)
        #part3 = timer(part2,'reading df')
        self.full_time = np.array(self.df['Time(s)'])
        self.raw_columns = list(self.df.columns)
        self.ncl = self.SYSTEM['DORIC']
        self.data = self._extract_data()
        self.columns = list(self.data.columns)
        self.cut_time = np.array(self.data[self.ncl['Time(s)']])
        self.sampling_rate = 1/np.median(np.diff(self.cut_time))
        #part4 = timer(part3,'a bunch of things')
        if self.split_recordings:
            self.recordings = self._split_recordings()
        else:
            self.recordings = {1: self.data}
        #part5 = timer(part4,'splitting the recording')
        self.rec_intervals  = [tuple([self.recordings[recording][self.ncl['Time(s)']].values[index] for index in [0,-1]]) for recording in range(1,self.number_of_recording+1)]
        #part6 = timer(part5,'getting record intervals')
        self.peaks = {}
        self._print('Analyzing peaks...')
        for r in self.recordings.keys():
            #try:
            data = self.norm(rec=r,add_time=True)
            t = data[:,0] ; s = data[:,1]
            self.peaks[r] = self._detect_peaks(t,s,plot=False)
            #except ValueError:
            #    self.peaks[r] = 'cannot calculate peaks'
        #part7 = timer(part6,'Peak analysis')
        self._print(f'Importing of {filepath} finished in {time.time() - start} seconds')

    def __repr__(self):
        """Give general information about the recording data."""
        general_info = f"""\
File                 : {self.filepath}
Rat_ID               : {self.rat_ID}
Number of recordings : {self.number_of_recording}
Data columns         : {self.columns}
Total lenght         : {self.full_time[-1] - self.full_time[0]} s
Kept lenght          : {self.cut_time[-1] - self.cut_time[0]} ({abs(self.full_time[0]-self.cut_time[0])} seconds trimmed)
Global sampling rate : {self.sampling_rate} S/s
"""
        return general_info

    def _read_file(self,filepath,alignement=0):
        """Read file and convert in the desired unit if needed."""
        df = pd.read_csv(filepath,usecols=["AIn-1 - Demodulated(Lock-In)","AIn-2 - Demodulated(Lock-In)","Time(s)"],dtype=np.float64)#,engine='pyarrow')
        # if not self.file_unit:
        #     if 29 <= df['Time(s)'].iloc[-1] <= 28_740:
        #         self.file_unit = 's'
        #     elif df['Time(s)'].iloc[-1] > 29_000:
        #         self.file_unit = 'ms'
        #     else:
        #         print('Please specify units')
        #         return
        # unit_values = {'s':1,'ms':1000}
        #ratio = unit_values[self.file_unit]/unit_values[self.user_unit]
        # df['Time(s)'] = df['Time(s)']/ratio + alignement
        #print(df,alignement)
        df['Time(s)'] = df['Time(s)'] + alignement
        return df

    def _extract_data(self):
        """Extract raw fiber data from Doric system."""
        idx = len(self.full_time)-len(self.full_time[self.full_time>self.trim_recording])
        return pd.DataFrame({self.ncl[i]: self.df[i].to_numpy()[idx:] for i in self.ncl if i in self.raw_columns})

    def _split_recordings(self):
        """Cut at timestamp jumps (defined by a step greater than N times the mean sample space (defined in config.yaml)."""
        time       = self.cut_time
        jumps      = list(np.where(np.diff(time)> self.split_treshold * np.mean(np.diff(time)))[0] + 1)
        indices    = [0] + jumps + [len(time)-1]
        ind_tuples = [(indices[i],indices[i+1]) for i in range(len(indices)-1) if indices[i+1]-indices[i] > self.min_sample_per_rec ]
        self.number_of_recording = len(ind_tuples)
        self._print(f"Found {self.number_of_recording} separate recordings.")
        return {ind_tuples.index((s,e))+1 : self.data.iloc[s:e,:] for s,e in ind_tuples}

    def to_csv(self,
               recordings   = 'all',
               auto         = True,
               columns      = None,
               column_names = None,
               prefix       = 'default'):
        """Export data to csv.

        - recording     (integer/list/'all') : default behaviour is exporting all recordings (if multiple splitted recordings exist); alternatively a single or a few splits can be chosen
        - auto          (True/False)         : automatically export signal and isosbestic separately with sampling_rate
        - columns       ([<names>])          : (changes auto to False) export specific columns with their timestamps (columns names accessible with <obj>.columns
        - columns_names ([<names>])          : change default name for columns in outputted csv (the list must correspond to the column list)
        - prefix        ('string']           : default prefix is 'raw filename + recording number' (followed by data column name)
        """
        if prefix == 'default': prefix = self.name
        if recordings == 'all': recordings = list(self.recordings.keys())
        sig_nom  = self.ncl["AIn-1 - Demodulated(Lock-In)"]
        ctrl_nom = self.ncl["AIn-2 - Demodulated(Lock-In)"]
        time_nom = self.ncl["Time(s)"]
        if auto and not columns:
            nomenclature = {sig_nom : 'signal', ctrl_nom : 'control'}
            for rec in recordings:
                time = self.get(time_nom,rec)
                for dataname in nomenclature.keys():
                    df = pd.DataFrame({'timestamps'     : time,
                                        'data'          : self.get(dataname,rec),
                                        'sampling_rate' : [1/np.diff(time)] + [np.nan]*(len(time)-1) }) # timestamps data sampling rate
                    df.to_csv(f'{prefix}_{rec}_{nomenclature[dataname]}.csv',index=False)
        else:
            recordings = self._list(recordings)
            columns = self._list(columns)
            column_names = self._list(column_names)
            for r in recordings:
                time = self.get(time_nom,r)
                for c in columns:
                    df = pd.DataFrame({'timestamps'     : time,
                                        'data'          : self.get(c,r),
                                        'sampling_rate' : [1/np.diff(time)] + [np.nan]*(len(time)-1) })
                    df.to_csv(f'{prefix}_{r}_{c}.csv',index=False)

    def _find_rec(self,timestamp):
        """Find recording number corresponding to inputed timestamp."""
        rec_num  = self.number_of_recording
        time_nom = self.ncl['Time(s)']
        return [i for i in range(1,rec_num+1) if self.get(time_nom,recording=i)[0] <= timestamp <= self.get(time_nom,recording=i)[-1]]

    def get(self,
            column,
            recording = 1,
            add_time  = False,
            as_df     = False):
        """Extracts a data array from a specific column of a recording (default is first recording)
        - add_time (False) : if True returns a 2D array with time included
        - as_df    (False) : if True returns data as a data frame (add_time will automatically set to True)"""
        time_nom = self.ncl['Time(s)']
        data = np.array(self.recordings[recording][column])
        time = np.array(self.recordings[recording][time_nom])
        if as_df:
            return pd.DataFrame({time_nom:time, column:data})
        if add_time:
            return np.vstack((time,data)).T
        else:
            return data

    def downsample(self):
        print("I don't exist yet :(")

    def TTL(self,ttl,rec=1):
        """Output TTL timestamps."""
        ttl             =  self.get(self.ncl[f"DI/O-{ttl}"],rec)
        time            =  self.get(self.ncl['Time(s)'],rec)
        ttl[ttl <  0.01] =  0
        ttl[ttl >= 0.01] =  1
        if (ttl == 1).all(): return [time[0]]
        if (ttl == 0).all(): return []
        index           =  np.where(np.diff(ttl) == 1)[0]
        return [time[i] for i in index]

    def norm(self,
             rec      = 1,
             method   = 'default',
             add_time = True):
        """Normalize data with specified method"""
        sig  = self.get(self.ncl["AIn-1 - Demodulated(Lock-In)"],rec)
        ctrl = self.get(self.ncl["AIn-2 - Demodulated(Lock-In)"],rec)
        tm    = self.get(self.ncl["Time(s)"],rec)
        if method == 'default': method = self.default_norm
        self._print(f"Normalizing recording {rec} with method '{method}'")
        if method == 'F':
            coeff = np.polynomial.polynomial.polyfit(ctrl,sig,1)
            #fitted_control = np.polynomial.polynomial.polyval(coeff,ctrl)
            #coeff = np.polyfit(ctrl,sig,1) /!\ coeff order is inversed with np.polyfit vs np.p*.p*.polyfit
            fitted_control = coeff[0] + ctrl*coeff[1]
            normalized = (sig - fitted_control)/fitted_control
        if method == 'Z':
            S = (sig - sig.mean())/signal.std()
            I = (ctrl - ctrl.mean())/ctrl.std()
            normalized = S-I
        if method == 'raw' or not method:
            normalized = np.vstack((sig,ctrl))
        if add_time:
            return np.vstack((tm,normalized)).T
        else:
            return normalized


    def _detect_peaks(self,t,s,window='default',distance='default',plot=True,figsize=(30,10),zscore='full',bMAD='default',pMAD='default'):
        """Detect peaks on segments of data:

        window:   window size for peak detection, the median is calculated for each bin
        distance: minimun distance between peaks, limits peak over detection
        zscore:   full if zscore is to be computed on the whole recording before splitting in bins, or bins if after
        plot,figsize: parameters for plotting
        """
        if window   == 'default': window   = self.peak_window
        if distance == 'default': distance = self.peak_distance
        if zscore   == 'default': zscore   = self.peak_zscore
        if bMAD     == 'default': bMAD     = int(self.peak_baseline_MAD)
        else: bMAD = int(bMAD)
        if pMAD     == 'default': pMAD     = int(self.peak_peak_MAD)
        else: pMAD = int(pMAD)
        self._print('')
        dF = s
        # calculate zscores
        if zscore == 'full':
            s = (s - s.mean())/s.std()
        # distance
        distance = round(float(distance.split('ms')[0])/(np.mean(np.diff(t))*1000))
        if distance == 0: distance = 1
        # find indexes for n second windows
        t_points = np.arange(t[0],t[-1],window)
        if t_points[-1] != t[-1]: t_points = np.concatenate((t_points,[t[-1]]))
        indexes = [np.where(abs(t-i) == abs(t-i).min())[0][0] for i in t_points]
        # create time bins
        bins   = [pd.Series(s[indexes[i-1]:indexes[i]],index=t[indexes[i-1]:indexes[i]]) for i in range(1,len(indexes))]
        dFbins = [pd.Series(dF[indexes[i-1]:indexes[i]],index=t[indexes[i-1]:indexes[i]]) for i in range(1,len(indexes))]
        if zscore == 'bins':
            bins = [(b - b.mean())/b.std() for b in bins]
        # find median for each bin and remove events >2MAD for baselines
        baselines = [b[b < np.median(b) + bMAD*np.median(abs(b - np.median(b)))] for b in bins]
        # calculate peak tresholds for each bin, by default >3MAD of previsoult created baseline
        tresholds = [np.median(b)+pMAD*np.median(abs(b-np.median(b))) for b in baselines]
        # find peaks using scipy.signal.find_peaks with thresholds
        peaks = []
        for n,bin_ in enumerate(bins):
            b = pd.DataFrame(bin_).reset_index()
            indices, heights = signal.find_peaks(b.iloc[:,1],height=tresholds[n],distance=distance)
            peaks.append(pd.DataFrame({'time'      : [b.iloc[i,0] for i in indices],
                                       'dF/F'      : [dFbins[n].iloc[i] for i in indices],
                                       'zscore' : list(heights.values())[0]}))
        if plot:
            plt.figure(figsize=figsize)
            peak_tresholds = [pd.Series(t,index=baselines[n].index) for n,t in enumerate(tresholds)]
            bin_medians    = [pd.Series(np.median(b),index=bins[n].index) for n,b in enumerate(bins)]
            bin_mad        = [pd.Series(np.median(abs(b - np.median(b))),index=bins[n].index) for n,b in enumerate(bins)]
            for n,i in enumerate(bins):
                c = random.choice(list('bgrcmy'))
                plt.plot(i,alpha=0.6,color=c)
                plt.plot(baselines[n],color=c,label=n*'_'+'signal <2MAD + median')
                plt.plot(bin_medians[n],color='k',label=n*'_'+'signal median')
                plt.plot(bin_medians[n]+bin_mad[n]*2,color='darkgray',label=n*'_'+'>2MAD + median')
                plt.plot(peak_tresholds[n],color='r',label=n*'_'+'>3MAD + baseline')
            for n,p in enumerate(peaks):
                plt.scatter(p.loc[:,'time'],p.loc[:,'zscore'])
            plt.legend()
        return pd.concat(peaks,ignore_index=True)


    def plot_transients(self,value='zscore',figsize=(20,20),rec='all',colors='k',alpha=0.3,**kwargs):
        """Show graphical representation of detected transients with their amplitude."""
        if rec == 'all': rec = self.number_of_recording
        fig,axes = plt.subplots(rec,figsize=figsize)
        if type(axes) != np.ndarray:
            axes.grid(which='both')
            data = self.peaks[1]
            for i in data.index:
                axes.vlines(data.loc[i,'time'],ymin=0,ymax=data.loc[i,value],colors=colors,alpha=alpha,**kwargs)
        else:
            for n,ax in enumerate(axes):
                ax.grid(which='both')
                data = self.peaks[n+1]
                for i in data.index:
                    ax.vlines(data.loc[i,'time'],ymin=0,ymax=data.loc[i,value],colors=colors,alpha=alpha,**kwargs)

    def peakFA(self,a,b):
        r = 0
        for n,i in enumerate(self.rec_intervals):
            if (i[0]<=a<i[1]) and (i[0]<b<=i[1]):
                r = n+1
        if r == 0: return
        data = self.peaks[r][(self.peaks[r]['time'] > a) & (self.peaks[r]['time'] < b)]
        return {'frequency'   : len(data)/(b-a),
                'mean zscore' : data['zscore'].mean(),
                'mean dF/F'   : data['dF/F'].mean(),
                'max zscore'  : data['zscore'].max(),
                'max dF/F'    : data['dF/F'].max(),
                'data'        : data}
