import behavioral_data
import fiber_data
from fp_utils import FiberPhotopy

import numpy as np
import pandas as pd
import time
import os
from math import ceil
import matplotlib.pyplot as plt
from scipy import integrate,stats,signal
import seaborn as sns
#from scipy.stats import sem

class RatSession(FiberPhotopy):
    """Create object containing both fiber recordings and behavioral files from a single session."""

    def __init__(self,behavior,fiber,rat_ID=None,**kwargs):
        super().__init__('all',**kwargs)
        heritage = self.__dict__
        self.rat_ID = rat_ID
        if type(behavior) == behavioral_data.BehavioralData:
            self.behavior = behavior
        else:
            self.behavior = behavioral_data.BehavioralData(behavior,autoinherit=heritage)
        if type(fiber) == fiber_data.FiberData:
            if fiber.alignement == self.behavior.rec_start:
                self.fiber = fiber
            else:
                self.fiber = fiber_data.FiberData(fiber.filepath,alignement=self.behavior.rec_start,autoinherit=heritage)
        else:
            self.fiber = fiber_data.FiberData(fiber,alignement=self.behavior.rec_start,autoinherit=heritage)
        self.analyses = {}

    def _sample(self,time_array,event_time,window):
        """Take a sample of the recording, based on one event and desired preevent and postevent duration."""
        start  = event_time - window[0]
        end    = event_time + window[1]
        start_idx = np.where(abs(time_array -      start) == min(abs(time_array -      start)))[0][0]
        event_idx = np.where(abs(time_array - event_time) == min(abs(time_array - event_time)))[0][0]
        end_idx   = np.where(abs(time_array -        end) == min(abs(time_array -        end)))[0][-1]
        return (start_idx , event_idx, end_idx)

    def _recorded_timestamps(self,events,window,**kwargs):
        """Return timestamps using BehavioralData timestamps function and applying it to analyzable_events."""
        if type(events) == str:
            recorded_events    = self.events(recorded=True,window=window)[events]
        elif type(events) == list:
            recorded_events = np.concatenate([self.events(recorded=True,window=window)[k] for k in events])
        else:
            print('You must input data as string')
            return
        return self.behavior.timestamps(events=recorded_events,**kwargs)

    def events(self,**kwargs):
        """Return all events ; wrapper for behavioral data function."""
        return self.behavior.events(**kwargs)

    def intervals(self,**kwargs):
        """Return all intervals ; wrapper for behavioral data function."""
        return self.behavior.intervals(**kwargs)

    def analyze_perievent(self,
                          event_time,
                          window     = 'default',
                          norm       = 'default'):
        """Return Analysis object, related to defined perievent window."""
        res = Analysis(self.rat_ID)
        res.event_time     = event_time
        res.fiberfile      = self.fiber.filepath
        res.behaviorfile   = self.behavior.filepath
        if norm == 'default':
            res.normalisation = self.default_norm
        else:
            try:
                res.normalisation   = {'F' : 'delta F/F', 'Z' : 'Z-scores'}[norm]
            except KeyError:
                print("Invalid choice for signal normalisation !\nZ-score differences: norm='Z'\ndelta F/f: stand='F'")
                return None
        res.event_time = event_time
        try:
            res.rec_number = self.fiber._find_rec(event_time)[0] # locates recording containing the timestamp
        except IndexError:
            print('No fiber recording at this timestamp:')
            print(event_time)
            print(self.fiber.filepath,self.behavior.filepath)
            return None
        if window == 'default':
            res.window    = self.perievent_window
        else:
            res.window    = window
        res.recordingdata = self.fiber.norm(rec=res.rec_number,method=res.normalisation)
        res.rawdata       = self.fiber.norm(rec=res.rec_number,method='raw')
        start_idx,event_idx,end_idx = self._sample(res.recordingdata[:,0],event_time,res.window)
        res.data          = res.recordingdata[start_idx:end_idx+1]
        res.raw_signal    = res.rawdata[start_idx:end_idx+1][:,1]
        res.raw_control   = res.rawdata[start_idx:end_idx+1][:,2]
        res.signal        = res.data[:,1]
        res.time          = res.data[:,0]
        res.sampling_rate = 1/np.diff(res.time).mean()
        res.postevent     = res.data[end_idx-event_idx:][:,1]
        res.pre_raw_sig   = res.raw_signal[:end_idx-event_idx]
        res.post_raw_sig  = res.raw_signal[end_idx-event_idx:]
        res.post_raw_ctrl = res.raw_control[end_idx-event_idx:]
        res.post_time     = res.data[end_idx-event_idx:][:,0]
        res.preevent      = res.data[:end_idx-event_idx][:,1]
        res.pre_time      = res.data[:end_idx-event_idx][:,0]
        res.zscores       = (res.signal - res.preevent.mean()) / res.preevent.std()
        res.pre_zscores   = res.zscores[:end_idx-event_idx]
        res.post_zscores  = res.zscores[end_idx-event_idx:]
        res.rob_zscores   = (res.signal - np.median(res.preevent))/stats.median_abs_deviation(res.preevent)
        res.pre_Rzscores  = res.rob_zscores[:end_idx-event_idx]
        res.post_Rzscores = res.rob_zscores[end_idx-event_idx:]
        res.preAVG_Z      = res.pre_zscores.mean()
        res.postAVG_Z     = res.post_zscores.mean()
        res.preAVG_RZ     = res.pre_Rzscores.mean()
        res.postAVG_RZ    = res.post_Rzscores.mean()
        res.pre_raw_AUC   = integrate.simpson(res.pre_raw_sig, res.pre_time)
        res.post_raw_AUC  = integrate.simpson(res.post_raw_sig, res.post_time)
        res.preAUC        = integrate.simpson(res.preevent, res.pre_time)
        res.postAUC       = integrate.simpson(res.postevent, res.post_time)
        res.preZ_AUC      = integrate.simpson(res.pre_zscores, res.pre_time)
        res.postZ_AUC     = integrate.simpson(res.post_zscores, res.post_time)
        res.preRZ_AUC     = integrate.simpson(res.pre_Rzscores, res.pre_time)
        res.postRZ_AUC    = integrate.simpson(res.post_Rzscores, res.post_time)
        self.analyses.update({f'rec{res.rec_number}_{res.event_time}_{res.window}' : res})
        return res

    def update_window(self,new_window):
        """Change perievent window."""
        self.perievent_window     = new_window
        self.analyzable_events  = self.behavior.events(recorded=True,window=self.perievent_window)
        self.recorded_intervals = self.behavior.intervals(recorded=True,window=self.perievent_window)

    def plot(self,what='events'):
        """Plot either events or intervals that happen within recording timeframe."""
        if what == 'events':
            data = {k:v for k,v in self.analyzable_events.items() if v.size>0}
        elif what == 'intervals':
            data = {k:v for k,v in self.recorded_intervals.items() if v.size>0}
        else:
            print("Choose either 'intervals' or 'events'")
        self.behavior.figure(obj=list(data.values()),label_list=list(data.keys()))

class Analysis:
    """Give results of perievent analysis relative to one event from a session."""

    _savgol = FiberPhotopy._savgol

    def __init__(self,rat_ID):
        """Initialize Analysis object."""
        pass

    def __repr__(self):
        """Represent Analysis. Show attributes."""
        return '\n'.join(['<obj>.'+i for i in self.__dict__.keys()])

    def plot(self,
             data,
             ylabel      = None,
             xlabel      = 'time',
             plot_title  = None,
             figsize     = (20,10),
             event       = True,
             event_label = 'event',
             linewidth   = 2,
             smooth      = 'savgol',
             unsmoothed  = True,
             smth_window = 'default'):
        """Visualize data, by default smoothes data with Savitski Golay filter (window size 250ms)."""
        try:
            data = self.__dict__[data]
        except KeyError:
            print(f'Input type should be a string, possible inputs:\n{self._possible_data()}')
        time = self.time
        if smooth:
            time_and_data = self.smooth(data,method=smooth,window=smth_window)
            s_time = time_and_data[:,0]
            s_data = time_and_data[:,1]
        if len(data) == len(time):
            fig = plt.figure(figsize=figsize)
            if unsmoothed == True:
                plt.plot(time,data,c='r',alpha=0.5)
            if smooth:
                plt.plot(s_time,s_data,c='k')
            plt.xlabel = xlabel
            plt.ylabel = ylabel
            plt.suptitle(plot_title)
            if event:
                plt.axvline(self.event_time,c='k',label=event_label,lw=linewidth)

    def _possible_data(self):
        d = {k:v for k,v in self.__dict__.items() if type(v) == np.ndarray}
        l = [f"'{k}'" for k in d.keys() if d[k].shape == self.time.shape]
        return '\n'.join(l)

    def smooth(self,
               data,
               method    = 'savgol',
               window    = 'default',
               polyorder = 3,
               add_time  = True):
        """Return smoothed data, possible methods: Savitsky-Golay filter, rolling average."""
        if type(data) == str:
            data = self.__dict__[data]
        if type(window) == str:
            if window[-2:] == 'ms':
                window = ceil(float(window[:-2])/1000 * self.sampling_rate)
            if window == 'default':
                window = ceil(self.sampling_rate/4) #250 ms
        if method == 'savgol':
            smoothed = self._savgol(data,window,polyorder)
            if add_time:
                return np.vstack((self.time,
                                  smoothed)).T
        if method == 'rolling':
            smoothed = pd.Series(data).rolling(window=window).mean().iloc[window-1:].values
            if add_time:
                return np.vstack((pd.Series(self.time).rolling(window=window).mean().iloc[window-1:].values,
                                  smoothed)).T
        return smoothed


class MultiSession(FiberPhotopy):
    """Group analyses or multiple events for single subject."""

    def __init__(self,folder=None,session_list=None,debug=800):
        super().__init__('all')
        start = time.time()
        if folder:
            self.rat_sessions = self._import_folder(folder)
        elif session_list:
            self.rat_sessions = self._import_sessions(session_list)
        if debug:
            print('Analysing interinfusion intervals...')
            removed = []
            for session,obj in self.rat_sessions.items():
                interval = obj.behavior.debug_interinj()
                if interval < debug:
                    removed.append((session,interval))
            if len(removed)>0:
                for session,interval in removed:
                    self.rat_sessions.pop(session)
                    self.removed = removed
                    print(f"Session {session} removed, interinfusion = {interval} ms")
            else:
                print('No sessions removed.')
                self.removed = 'No session removed'
        self.names = list(self.rat_sessions.keys())
        if folder: self.multibehavior = behavioral_data.MultiBehavior(folder)
        print(f'Extraction finished, {int(len(self.names)*2)} files in {time.time() - start} seconds')

    def _import_folder(self,folder):
        sessions = {}
        if type(folder) == str:
            rats = os.listdir(folder)
            for r in rats:
                print(f"\nImporting folder {r}...")
                for file in os.listdir(folder+'/'+r):
                    path = folder+'/'+r+'/'+file
                    if '.csv' in path:
                        f = path
                    elif '.dat' in path:
                        b  = path
                sessions[r] = RatSession(behavior=b ,fiber=f)
            return sessions

    def _import_sessions(self,sesslist):
        sessions = {}
        for s in sesslist:
            if s.ID != None:
                name = s.ID
            else:
                name = s.fiber.filepath
            sessions[name] = s

    def show_rates(self,**kwargs):
        self.multibehavior.show_rate(**kwargs)

    def analyze(self,events,
                window='default',
                norm='default',
                **kwargs):
        result = MultiAnalysis()
        if window=='default': window=self.perievent_window
        result.window = window
        result.rat_sessions = self.rat_sessions
        result.dict = {}
        for k,v in self.rat_sessions.items():
            timestamps = v._recorded_timestamps(events=events,window=window,**kwargs)
            result.dict[k]  = [v.analyze_perievent(i,norm=norm,window=window) for i in timestamps]
            for obj in result.dict[k]:
                if obj:
                    for att in obj.__dict__.keys():
                        if result.__dict__.get(att):
                            result.__dict__[att].append(obj.__dict__[att])  #append
                        else:
                            result.__dict__[att] = [obj.__dict__[att]]   #create

        result.epoch = []
        for n,t in enumerate(result.time):
            result.epoch.append(t - result.event_time[n])
        result.update()
        return result

    def compare_behavior(self,attribute):
        obj_behav = [self.rat_sessions[k].behavior for k in self.rat_sessions.keys()]
        end = max(rat[1].behavior.end for rat in self.rat_sessions.items())
        timebins = np.arange(0,round(end)+1,1)
        cumul = [np.zeros(timebins.shape) for i in range(len(obj_behav))]
        names = [] ; endpoints = []
        for n,o in enumerate(obj_behav):
            a,b = np.histogram(o.__dict__[attribute],bins=timebins)
            cumul[n] = np.array([np.sum(a[:n]) for n in range(len(a))])
            names.append(self.names[n])
            endpoints.append(round(o.end))
        plt.figure(figsize=(20,10))
        plt.title(attribute)
        sns.heatmap(np.array(cumul),yticklabels=self.names)
        plt.figure(figsize=(20,10))
        pltcumul = [cumul[n][:endpoints[n]] for n in range(len(endpoints))]
        for n,i in enumerate(pltcumul):
            plt.plot(i,label=names[n])
        plt.legend()
        return np.array(cumul)

class MultiAnalysis(FiberPhotopy):

    def __init__(self):
        super().__init__('all')
        self.exclude_list = []


    def possible_data(self):
        """Return dictionnary of possible data to plot"""
        comparator = [i.shape for i in self.epoch]
        possible = []
        for k,v in self.__dict__.items():
            if type(v) == list:
                if len(v) > 0:
                    if type(v[0]) == np.ndarray:
                        if [i.shape for i in v] == comparator:
                            possible.append(k)
        return possible

    def exclude(self,list_of_sessions):
        return

    def update(self,
               nb_of_points='default'):
        """Recalculate mean values for all relevant values (perievent data). "nb of points" is the desired number of points for the new aligned data. Default is mean number of points of the data."""
        if nb_of_points == 'default':
            self.nb_of_points = round(np.mean([i.shape for i in self.time]))
        self.EPOCH = np.linspace(-1*self.window[0],
                                 self.window[1],
                                 self.nb_of_points)
        for value in self.possible_data():
            self.__dict__['interpolated_'+value] = []
            for a,b in zip(self.epoch,self.__dict__[value]):
                self.__dict__['interpolated_'+value].append(np.interp(self.EPOCH,a,b))
            self.__dict__[value.upper()]= sum(self.__dict__['interpolated_'+value])/len(self.__dict__['interpolated_'+value])

    def plot(self,
             data='signal',
             smooth_data=True,
             smooth_mean=True,
             data_window=500,
             mean_window=500,
             label=None,
             **kwargs):
        """Visualize specified data."""
        cfg = {'figsize':(20,10),'c':'k','linewidth':1,'alpha':0.3}
        if data not in self.__dict__.keys():
            print(f"You need to choose among {self.possible_data()}")
            return
        else:
            mean_data = self.__dict__[data.upper()]
            data_list = self.__dict__[data]
        for key in cfg.keys():
            if key in kwargs.keys():
                cfg[key] = kwargs[key]
        plt.figure(figsize=(cfg['figsize']))

        for n,z in enumerate(data_list):
            if label: label = list(self.rat_sessions.keys())[n]
            if smooth_data:
                z = self._savgol(z,data_window)
            plt.plot(self.epoch[n],z,alpha=cfg['alpha'],label=label)
            if label:
                plt.legend()
        if smooth_mean:
            mean_data = self._savgol(mean_data,mean_window)
        plt.plot(self.EPOCH,mean_data,c='k',label='mean')
        plt.axvline(0,alpha=1,linewidth=cfg['linewidth'],c=cfg['c'])
