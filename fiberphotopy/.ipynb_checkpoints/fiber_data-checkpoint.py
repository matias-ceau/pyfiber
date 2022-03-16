import numpy as np
import pandas as pd
from fp_utils import FiberPhotopy

class FiberData(FiberPhotopy):
    """Extract fiberphotometry data from Doric system."""

    def __init__(self,
                 filepath,
                 name      = 'default',
                 rat_ID    = None,
                 alignement=0,
                 **kwargs):
        super().__init__('fiber',**kwargs)
        self.alignement = alignement
        self.filepath = filepath
        self.rat_ID = rat_ID
        if name == 'default':
            self.name = self.filepath.split('/')[-1].split('.csv')[0]
            if self.rat_ID:
                self.name += rat_ID
        self.number_of_recording = 0
        self.df = self._read_file(filepath,alignement=alignement)
        self.full_time = np.array(self.df['Time(s)'])
        self.raw_columns = list(self.df.columns)
        self.ncl = self.config['DORIC']
        self.data = self._extract_data()
        self.columns = list(self.data.columns)
        self.cut_time = np.array(self.data[self.ncl['Time(s)']])
        self.sampling_rate = 1/np.median(np.diff(self.cut_time))
        if self.split_recordings:
            self.recordings = self._split_recordings()
        else:
            self.recordings = {1: self.data}
        self.rec_intervals  = [tuple([self.recordings[recording][self.ncl['Time(s)']].values[index] for index in [0,-1]]) for recording in range(1,self.number_of_recording)]

    def __repr__(self):
        """Give general information about the recording data."""
        general_info = f"""\
File                 : {self.filepath}
Rat_ID               : {self.rat_ID}
Number of recordings : {self.number_of_recording}
Data columns         : {self.columns}
Total lenght         : {self.full_time[-1] - self.full_time[0]} {self.user_unit}
Kept lenght          : {self.cut_time[-1] - self.cut_time[0]} ({abs(self.full_time[0]-self.cut_time[0])} seconds trimmed)
Global sampling rate : {self.sampling_rate} S/{self.user_unit}
Original file unit   : {self.file_unit}"""
        return general_info

    def _read_file(self,filepath,alignement=0):
        """Read file and convert in the desired unit if needed."""
        df = pd.read_csv(filepath)
        if self.file_unit and self.user_unit:
            if self.file_unit == self.user_unit:
                return df
        elif not self.file_unit:
            if 29 <= df['Time(s)'].iloc[-1] <= 28_740:
                self.file_unit = 's'
            elif df['Time(s)'].iloc[-1] > 29_000:
                self.file_unit = 'ms'
            else:
                print('Please specify units')
                return
        unit_values = {'s':1,'ms':1000}
        ratio = unit_values[self.file_unit]/unit_values[self.user_unit]
        df['Time(s)'] = df['Time(s)']/ratio + alignement
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
        ind_tuples = [(indices[i],indices[i+1]) for i in range(len(indices)-1)]
        self.number_of_recording = len(ind_tuples)
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
        return [i for i in range(1,rec_num) if self.get(time_nom,recording=i)[0] <= timestamp <= self.get(time_nom,recording=i)[-1]]

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
        """Outputs TTL timestamps"""
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
             method   = 'F',
             add_time = True):
        """Normalizes data with specified method"""
        signal  = self.get(self.ncl["AIn-1 - Demodulated(Lock-In)"],rec)
        control = self.get(self.ncl["AIn-2 - Demodulated(Lock-In)"],rec)
        time    = self.get(self.ncl["Time(s)"],rec)
        if method == 'F':
            fit = np.polyfit(control,signal,1)
            fitted_control = control*fit[0] + fit[1]
            normalized = (signal - fitted_control)/fitted_control
        if method == 'Z':
            S = (signal - signal.mean())/signal.std()
            I = (control - control.mean())/control.std()
            normalized = S-I
        if method == 'raw' or not method:
            normalized = np.vstack((signal,control))
        if add_time:
            return np.vstack((time,normalized)).T
        else:
            return normalized
