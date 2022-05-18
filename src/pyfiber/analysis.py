import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from scipy import integrate, stats
import seaborn as sns
from typing import List, Tuple, Union, Any
import matplotlib.style as st
st.use('ggplot')

from .behavior import Behavior, MultiBehavior
from .fiber import Fiber
from ._utils import PyFiber as PyFiber
__all__ = ['Session', 'MultiSession', 'Analysis', 'MultiAnalysis']

Intervals = List[Tuple[float,float]]
Events = np.ndarray

class Session(PyFiber):
    """Create object containing both fiber recordings and behavioral files.
    
    From a single session/subject.
    """

    vars().update(PyFiber.FIBER)
    vars().update(PyFiber.BEHAVIOR)

    def __init__(self, behavior, fiber, ID=None, **kwargs):
        super().__init__(**kwargs)
        self.ID = ID
        if type(behavior) == Behavior:
            self.behavior = behavior
        else:
            self.behavior = Behavior(behavior)
        if type(fiber) == Fiber:
            if fiber.alignement == self.behavior.rec_start:
                self.fiber = fiber
            else:
                self.fiber = Fiber(
                    fiber.filepath, alignement=self.behavior.rec_start)
        else:
            self.fiber = Fiber(fiber, alignement=self.behavior.rec_start)
        self.analyses = {}

    def __repr__(self): 
        """Representation of Session object."""
        return f"""\
<Session(fiber    = "{self.fiber.filepath}",
            behavior = "{self.behavior.filepath}")"""

    def _sample(self, time_array, event_time, window):
        """Take a sample of the recording.

    Based on one event and desired preevent and postevent duration.
    """
        start = event_time - window[0]
        end = event_time + window[1]
        start_idx = np.where(abs(time_array - start) ==
                             min(abs(time_array - start)))[0][0]
        event_idx = np.where(abs(time_array - event_time)
                             == min(abs(time_array - event_time)))[0][0]
        end_idx = np.where(abs(time_array - end) ==
                           min(abs(time_array - end)))[0][-1]
        return (start_idx, event_idx, end_idx)

    def _recorded_timestamps(self, events, window, **kwargs):
        """Return timestamps.
   
    Uses Behavior objecys timestamps method function
    and applying it to analyzable_events.
    """
        if type(events) == str:
            recorded_events = self.events(recorded=True, window=window)[events]
        elif type(events) == list:
            recorded_events = np.concatenate(
                [self.events(recorded=True, window=window)[k] for k in events])
        else:
            self._print('You must input data as string')
            return
        return self.behavior.timestamps(events=recorded_events, **kwargs)

    def events(self, **kwargs):
        """Return all events ; wrapper for behavioral data function."""
        return self.behavior.events(**kwargs)

    def intervals(self, **kwargs):
        """Return all intervals ; wrapper for behavioral data function."""
        return self.behavior.intervals(**kwargs)

    def analyze(self,
                event_time,
                window='default',
                norm='default'):
        """Return Analysis object, related to defined perievent window."""
        res = Analysis(self.ID)
        res.event_time = event_time
        res.fiberfile = self.fiber.filepath
        res.behaviorfile = self.behavior.filepath
        if norm == 'default':
            res.normalisation = self.default_norm
        else:
            try:
                res.normalisation = {'F': 'delta F/F', 'Z': 'Z-scores'}[norm]
            except KeyError:
                self._print(
                    """Invalid choice for signal normalisation !
                    nZ-score differences: norm='Z'\ndelta F/f: stand='F'""")
                return None
        res.event_time = event_time
        try:
            # locates recording containing the timestamp
            res.rec_number = self.fiber._find_rec(event_time)[0]
        except IndexError:
            self._print(f"""\
No fiber recording at timestamp: {event_time}
for {self.fiber.filepath},{self.behavior.filepath}""")
            return None
        if window == 'default':
            res.window = self.perievent_window
        else:
            res.window = window
        res.recordingdata = self.fiber.norm(
            rec=res.rec_number, method=res.normalisation)
        res.rawdata = self.fiber.norm(rec=res.rec_number, method='raw')
        start_idx, event_idx, end_idx = self._sample(
            res.recordingdata[:, 0], event_time, res.window)
        res.data = res.recordingdata[start_idx:end_idx+1]
        res.raw_signal = res.rawdata[start_idx:end_idx+1][:, 1]
        res.raw_control = res.rawdata[start_idx:end_idx+1][:, 2]
        res.signal = res.data[:, 1]
        res.time = res.data[:, 0]
        res.sampling_rate = 1/np.diff(res.time).mean()
        res.postevent = res.data[end_idx-event_idx:][:, 1]
        res.pre_raw_sig = res.raw_signal[:end_idx-event_idx]
        res.post_raw_sig = res.raw_signal[end_idx-event_idx:]
        res.post_raw_ctrl = res.raw_control[end_idx-event_idx:]
        res.post_time = res.data[end_idx-event_idx:][:, 0]
        res.preevent = res.data[:end_idx-event_idx][:, 1]
        res.pre_time = res.data[:end_idx-event_idx][:, 0]
        res.zscores = (res.signal - res.preevent.mean()) / res.preevent.std()
        res.pre_zscores = res.zscores[:end_idx-event_idx]
        res.post_zscores = res.zscores[end_idx-event_idx:]
        res.rob_zscores = (res.signal - np.median(res.preevent)) / \
            stats.median_abs_deviation(res.preevent)
        res.pre_Rzscores = res.rob_zscores[:end_idx-event_idx]
        res.post_Rzscores = res.rob_zscores[end_idx-event_idx:]
        res.preAVG_Z = res.pre_zscores.mean()
        res.postAVG_Z = res.post_zscores.mean()
        res.preAVG_RZ = res.pre_Rzscores.mean()
        res.postAVG_RZ = res.post_Rzscores.mean()
        res.pre_raw_AUC = integrate.simpson(res.pre_raw_sig, res.pre_time)
        res.post_raw_AUC = integrate.simpson(res.post_raw_sig, res.post_time)
        res.preAUC = integrate.simpson(res.preevent, res.pre_time)
        res.postAUC = integrate.simpson(res.postevent, res.post_time)
        res.preZ_AUC = integrate.simpson(res.pre_zscores, res.pre_time)
        res.postZ_AUC = integrate.simpson(res.post_zscores, res.post_time)
        res.preRZ_AUC = integrate.simpson(res.pre_Rzscores, res.pre_time)
        res.postRZ_AUC = integrate.simpson(res.post_Rzscores, res.post_time)
        self.analyses.update(
            {f'rec{res.rec_number}_{res.event_time}_{res.window}': res})
        return res

    def update_window(self, new_window):
        """Change perievent window."""
        self.perievent_window = new_window
        self.analyzable_events = self.behavior.events(
            recorded=True, window=self.perievent_window)
        self.recorded_intervals = self.behavior.intervals(
            recorded=True, window=self.perievent_window)

    def plot(self, what='events'):
        """Plot either events or intervals.

        Tests if they happen within recording timeframe.
        """
        if what == 'events':
            data = {k: v for k, v in self.analyzable_events.items()
                    if v.size > 0}
        elif what == 'intervals':
            data = {k: v for k, v in self.recorded_intervals.items()
                    if v.size > 0}
        else:
            self._print("Choose either 'intervals' or 'events'")
        self.behavior.figure(obj=list(data.values()),
                             label_list=list(data.keys()))


class Analysis(PyFiber):
    """Give results of perievent analysis.

    Relative to one event from a session.
    """

    _savgol = PyFiber._savgol

    def __init__(self, ID):
        super().__init__()
        pass

    def __repr__(self):
        """Representation of Analysis object."""
        return f"<Analysis.object> // {self.window}, {self.event_time}"

    def plot(self,
             data,
             ylabel=None,
             xlabel='time',
             plot_title=None,
             figsize=(20, 10),
             event=True,
             event_label='event',
             linewidth=2,
             smooth='savgol',
             unsmoothed=True,
             smth_window='default'):
        """Visualize data.

        By default smoothes data with Savitski Golay filter
        (window size 250ms).
        """
        try:
            data = self.__dict__[data]
        except KeyError:
            self._print(
                f"""Input type should be a string, possible inputs:
                    {self._possible_data()}""")
        time = self.time
        if smooth:
            time_and_data = self.smooth(
                data, method=smooth, window=smth_window)
            s_time = time_and_data[:, 0]
            s_data = time_and_data[:, 1]
        if len(data) == len(time):
            fig = plt.figure(figsize=figsize)
            if unsmoothed is True:
                plt.plot(time, data, c='r', alpha=0.5)
            if smooth:
                plt.plot(s_time, s_data, c='k')
            plt.xlabel = xlabel
            plt.ylabel = ylabel
            plt.suptitle(plot_title)
            if event:
                plt.axvline(self.event_time, c='k',
                            label=event_label, lw=linewidth)
            plt.show()

    def _possible_data(self):
        d = {k: v for k, v in self.__dict__.items() if type(v) == np.ndarray}
        l = [f"'{k}'" for k in d.keys() if d[k].shape == self.time.shape]
        return '\n'.join(l)

    def smooth(self,
               data,
               method='savgol',
               window='default',
               polyorder=3,
               add_time=True):
        """Return smoothed data.

        Possible methods:
            - Savitsky-Golay filter
            - rolling average.
            """
        if type(data) == str:
            data = self.__dict__[data]
        if type(window) == str:
            if window[-2:] == 'ms':
                window = np.ceil(float(window[:-2])/1000 * self.sampling_rate)
            if window == 'default':
                window = np.ceil(self.sampling_rate/4)  # 250 ms
        if method == 'savgol':
            smoothed = self._savgol(data, window, polyorder)
            if add_time:
                return np.vstack((self.time,
                                  smoothed)).T
        if method == 'rolling':
            smoothed = pd.Series(data).rolling(
                window=window).mean().iloc[window-1:].values
            if add_time:
                return np.vstack(
                    (pd.Series(self.time).rolling(
                        window=window
                        ).mean().iloc[window-1:].values, smoothed)
                    ).T
        return smoothed


class MultiSession(PyFiber):
    """Group analyses or multiple events for single subject."""
    vars().update(PyFiber.FIBER)
    vars().update(PyFiber.BEHAVIOR)
#TODO CHANGE DEBUG
    def __init__(self, folder, debug=0.800, verbosity=True):
        super().__init__()
        print('folder', folder)
        self._verbosity = verbosity
        self.folder = folder
        start = time.time()
        self._import_folder(folder)
        if debug:
            self._print('Analysing interinfusion intervals...')
            self.removed = []
            for session, obj in self.sessions.items():
                interval = obj.behavior._debug_interinj()
                if interval < debug:
                    self.removed.append((session, interval))
            if len(self.removed) > 0:
                for session, interval in self.removed:
                    self.sessions.pop(session)
                    self._print(
                        f"Session {session} removed,\
                        interinfusion = {interval} s")
            else:
                self._print('No sessions removed.')
                self.removed = 'No session removed'
        self.names = list(self.sessions.keys())
        self._getID()
        self._print(
            f"""Extraction finished, {int(len(self.names))} sessions
                ({int(len(self.names)*2)} files)
                    in {time.time() - start} seconds"""
                    )

    def __repr__(self):
        """Representation fo MultiSession object."""
        return f"""<MultiSession object> // folder: {self.folder}"""

    def __getitem__(self, item):
        """Return session objective with corresponding filename."""
        return self.sessions[item]

    def _import_folder(self, folder):
        sessions = {}
        if type(folder) == str:
            rats = os.listdir(folder)
            for r in rats:
                self._print(f"\nImporting folder {r}...")
                for file in os.listdir(os.path.join(folder, r)):
                    path = os.path.join(folder, r, file)
                    if '.csv' in path:
                        f = path
                    elif '.dat' in path:
                        b = path
                sessions[r] = Session(behavior=b, fiber=f)
            self.sessions = sessions

    def _getID(self):
        sep = self.GENERAL['folder_nomenclature']['separator']
        attr = self.GENERAL['folder_nomenclature']['meanings']
        naming = self.GENERAL['folder_nomenclature']['naming'].split(sep)
        ID = self.GENERAL['folder_nomenclature']['ID']
        vals = [b.replace('@', '') for b in [a for a in naming]]
        vals = [[a.replace(i, '') for i, a in zip(vals, b)]
                for b in [i.split(sep) for i in self.names]]
        detailedID = [{k: v
                       for k, v in zip(attr, a) if k != ''}
                      for a in vals
                      ]
        self.ID = [sep.join([str(d[a]) for a in ID]) for d in detailedID]
        df = pd.concat([pd.DataFrame({k: pd.Series(v)
                                      for k, v in d.items()})
                        for d in detailedID],
                       ignore_index=True)
        df['ID'] = self.ID
        df['session'] = self.names
        self.details = df

    def show_rates(self, **kwargs):
        """Show rates for specified events."""
        MultiBehavior(self.folder).show_rate(**kwargs)

    def analyze(self, events,
                window='default',
                norm='default',
                sessions='all',
                **kwargs):
        """Analyze specific events.

        events            : ex: 'np1'      
        window='default'  : peri-event window
        (default is specified in config.yaml)
        norm='default'    : normalization method used
    """
        # Generate new MultiAnalysis instance
        result = MultiAnalysis()
        
        # Parameters
        if sessions == 'all':
            sessions = self.sessions
        else:
            sessions = {k: v for k, v in self.sessions.items()
                        if k in sessions}

        if window == 'default':
            window = self.perievent_window
 
        result.WINDOW = window
        result.dict = {}
        result.key = []

        # Create 'key' and 'dict' attribute.
        # 'key' correspond to the parent session for each event
        # dict correspond to each session and their list of Analysis instances
        # (one for each timestamp)

        for k, v in sessions.items():
            timestamps = v._recorded_timestamps(
                events=events, window=window, **kwargs
                )
            result.dict[k] = [v.analyze(i, norm=norm, window=window)
                              for i in timestamps]
            result.key.append(len(result.dict[k])*[k])

        result.key = [j for k in [i for i in result.key if i] for j in k]
        result._objects = np.concatenate(list(result.dict.values())).tolist()
     
        # remove empty lists of analysis (no recording at timestamp or
        # no events) and update key accordingly
        sess_df = pd.DataFrame({'key': result.key, 'objects': result._objects})
        
        # drop and alert on empty analysis lists
        idx = sess_df[sess_df.objects.isnull()].index
        rvd = ', '.join(sess_df.loc[idx,'key'].tolist())
        self._print(f'Only some timestamp, or no timestamps for {rvd}') 

        sess_df = sess_df.drop(index=idx)
        result.key = sess_df.key.tolist()
        result._objects = sess_df.objects.tolist()
        
        # find all non private attribute of the Analysis instances
        result._attribute_names = [
            i for i in np.unique(
                np.concatenate([list(vars(a).keys()) 
                                for a in result._objects]))
            if i[0] != '_']
        
        # group subject attribute together in instance attribute 
        result._attribute_dict = {
            k: [d[k] for d in [
                vars(o) for o in result._objects]
                ]
            for k in result._attribute_names
            }
        
        result.__dict__.update(result._attribute_dict)
        
        # calculate epochs for all time series by subtracting the corresponding
        # event time
        result.epoch = [t - result.event_time[n]
                        for n, t in enumerate(result.time)]
        
        result.update()
        if type(events) == str:
            result.event_name = events
        elif type(events) == list:
            if len(events):
                if type(events[0]) == str:
                    result.event_name = str(events)
        else:
            result.event_name = 'custom events, see times.'
        return result

    def compare_behavior(self, attribute):
        """Compare behavior (input should be event name)"""
        obj_behav = [self.sessions[k].behavior for k in self.sessions.keys()]
        end = max(rat[1].behavior.end for rat in self.sessions.items())
        timebins = np.arange(0, round(end)+1, 1)
        cumul = [np.zeros(timebins.shape) for i in range(len(obj_behav))]
        names = []
        endpoints = []
        for n, o in enumerate(obj_behav):
            a, b = np.histogram(o.__dict__[attribute], bins=timebins)
            cumul[n] = np.array([np.sum(a[:n]) for n in range(len(a))])
            names.append(self.names[n])
            endpoints.append(round(o.end))
        plt.figure(figsize=(20, 10))
        plt.title(attribute)
        sns.heatmap(np.array(cumul), yticklabels=self.names)
        plt.figure(figsize=(20, 10))
        pltcumul = [cumul[n][:endpoints[n]] for n in range(len(endpoints))]
        for n, i in enumerate(pltcumul):
            plt.plot(i, label=names[n])
        plt.legend()
        plt.show()
        return np.array(cumul)


class MultiAnalysis(PyFiber):
    vars().update(PyFiber.FIBER)
    vars().update(PyFiber.BEHAVIOR)

    def __init__(self):
        super().__init__()
        self.exclude_list = []

    def __repr__(self): return f"""<MultiAnalysis object>
    event:             {self.event_name}
    window:            {self.WINDOW}
    number of events:  {len(self.key)}
    key:               {', '.join(self.key)}"""

    def __getitem__(self, item):
        return self.dict[item]

    @property
    def data(self):
        return pd.DataFrame({k: v for k, v in self._attribute_dict.items() if type(v[0]) != np.ndarray}, index=self.key)

    @property
    def full_data(self):
        return pd.DataFrame(self._attribute_dict, index=self.key)

    def possible_data(self):
        """Return dictionnary of possible data to plot"""
        comparator = [i.shape for i in self.epoch]
        possible = []
        for k, v in self.__dict__.items():
            if type(v) == list:
                if len(v) > 0:
                    if type(v[0]) == np.ndarray:
                        if [i.shape for i in v] == comparator:
                            possible.append(k)
        return possible

    def exclude(self, list_of_sessions):
        return

    def update(self,
               nb_of_points='default'):
        """Recalculate mean values for all relevant values (perievent data). "nb of points" is the desired number of points for the new aligned data. Default is mean number of points of the data."""
        if nb_of_points == 'default':
            self.nb_of_points = round(np.mean([i.shape for i in self.time]))
        self.EPOCH = np.linspace(-1*self.WINDOW[0],
                                 self.WINDOW[1],
                                 self.nb_of_points)
        for value in self.possible_data():
            self.__dict__['interpolated_'+value] = []
            for a, b in zip(self.epoch, self.__dict__[value]):
                self.__dict__['interpolated_' +
                              value].append(np.interp(self.EPOCH, a, b))
            self.__dict__[value.upper()] = sum(
                self.__dict__['interpolated_'+value])/len(self.__dict__['interpolated_'+value])

    def plot(self,
             data='signal',
             smooth_data=True,
             smooth_mean=True,
             data_window=500,
             mean_window=500,
             label=None,
             **kwargs):
        """Visualize specified data."""
        cfg = {'figsize': (20, 10), 'c': 'k', 'linewidth': 1, 'alpha': 0.3}
        if data not in self.__dict__.keys():
            self._print(f"You need to choose among {self.possible_data()}")
            return
        else:
            mean_data = self.__dict__[data.upper()]
            data_list = self.__dict__[data]
        for key in cfg.keys():
            if key in kwargs.keys():
                cfg[key] = kwargs[key]
        plt.figure(figsize=(cfg['figsize']))

        for n, z in enumerate(data_list):
            if label:
                label = list(self.sessions.keys())[n]
            if smooth_data:
                z = self._savgol(z, data_window)
            plt.plot(self.epoch[n], z, alpha=cfg['alpha'], label=label)
            if label:
                plt.legend()
        if smooth_mean:
            mean_data = self._savgol(mean_data, mean_window)
        plt.plot(self.EPOCH, mean_data, c='k', label='mean')
        plt.axvline(0, alpha=1, linewidth=cfg['linewidth'], c=cfg['c'])
        plt.show()
