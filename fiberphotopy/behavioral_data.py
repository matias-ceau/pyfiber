import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fp_utils import FiberPhotopy
import info

class BehavioralData(FiberPhotopy):
    """Input : imetronic behavioral data file path."""

    def __init__(self,
                 filepath,
                 custom='all',
                 **kwargs):
        """Initialize BehavioralData object, timestamp arrays are lowercase, periods are uppercase."""
        super().__init__('behavior',**kwargs)
        self.df               = pd.read_csv(filepath,skiprows=12,delimiter='\t',header=None,names=['TIME','F','ID','_P','_V','_L','_R','_T','_W','_X','_Y','_Z'])
        self.custom           = custom
        self.filepath         = filepath
        self.start            = self.df.iloc[0,0]
        self.end              = self.df.iloc[-1,0]            # last timestamp (in ms) automatically changed to user_unit (see fp_utils.py)
        self.time_ratio       = self._time_ratio()
        #RAW EVENT TIMESTAMPS
        self.hled_on          = self._extract(1, 1,'_P',1)     # houselight on (start of ND period) (1,1)
        self.hled_off         = self._extract(1, 1,'_P',0)     # houselight on
        self.led1_on          = self._extract(1, 2,'_P',1)     # CS = administration light on (1,2)
        self.led1_off         = self._extract(1, 2,'_P',0)     # led1 off (possibly unnecessary switch off commands depending of exercises)
        self.led2_on          = self._extract(1, 3,'_P',1)
        self.led2_off         = self._extract(1, 3,'_P',0)
        self.np1              = self._extract(3, 1,'_V',1)     # NP active detected (3,1)
        self.np2              = self._extract(3, 2,'_V',1)     # NP inactive detected (3,2)
        self.inj1             = self._extract(6, 1,'_L',1)     # injection (6,1) (NB: first pump turn)
        self.ttl1_on          = self._extract(15,1,'_L',1)
        self.ttl1_off         = self._extract(15,1,'_L',0)
        if self.ttl1_on != np.array([]):
            self.rec_start    = self.ttl1_on[:1]
        else:
            self.rec_start    = np.array([])
        for i in ['hled','led1','led2','ttl1']:
            self._attr_interval(i)
        self.DARK             = self._intersection(self.HLED_OFF,self.LED1_OFF,self.LED2_OFF)
        self.TO_DARK          = self._interval_is_close_to(self.DARK,self.inj1)
        self.TIMEOUT          = self._union(self.LED1_ON,self.TO_DARK)
        self.NOTO_DARK        = self._intersection(self.DARK,self._non(self.TIMEOUT,end=self.end))
        # SWITCHES
        self._custom_events_intervals()

    def info(self):
        """Return information about object."""
        print(info.behavior_help)

    def __repr__(self):
        """Return __repr__."""
        return f"""\
GENERAL INFORMATION:
******************************************************************************************************************************************************************
    Filename            : {self.filepath}
    Rat ID              : {self.rat_ID}
    Experiment duration : {self.end} ({self.user_unit})
    Time unit           : {self.user_unit} (File original unit: {self.file_unit}
    Fixed ratio         : FR{self.fixed_ratio}

Useful commands:
    - <obj>.summary() : shows graphical summary of the dat file
    - <obj>.df        : shows raw IMETRONIC data
    - <obj>.elements  : list of elements that can be used in the main functions

Main functions:
    - <obj>.timestamps(param)  : outputs timestamps based on user-defined criteria and optionally outputs a csv of timestamps
    - <obj>._internal_graph(element)     : shows graphical representation of the element
    - <obj>.get(param)         : outputs slice from full dat file (formatted as a DataFrame)

FULL_HELP: <obj>.info"""

######################### HELPER FUNCTIONS ###########################
    def _extract(self,family,subtype,column,value):
        """Extract timestamps for counters from imetronic file, first and second values indentify datatype, additional column precise event of interest (start,end, etc.)."""
        return     self.df[(self.df['F']    ==  family)  &
                          (self.df['ID']   ==  subtype) &
                          (self.df[column] ==  value)]['TIME'].to_numpy()/self.time_ratio

    def _time_ratio(self):
        """Read file and returns ratio with which to divide timedata."""
        if self.file_unit and self.user_unit:
            if self.file_unit == self.user_unit:
                pass
        elif not self.file_unit:
            minimum = self.experiment_duration['min']
            maximum = self.experiment_duration['max']
            if minimum <= self.end <= maximum:
                self.file_unit = 's'
            elif self.end > maximum:
                self.file_unit = 'ms'
            else:
                print('Please specify units')
                pass
        unit_values = {'s':1,'ms':1000}
        ratio = unit_values[self.file_unit]/unit_values[self.user_unit]
        self.start
        self.end /= ratio
        return ratio
    
    
    def _attr_interval(self,string):
        on = self.__dict__[string+'_on']
        off = self.__dict__[string+'_off']
        on_int = self._interval(on,off,self.end)
        self.__dict__[string.upper()+'_ON'] = on_int
        self.__dict__[string.upper()+'_OFF'] = self._non(on_int,self.end)

    def _interval(self,on,off,end):
        on  = set([i for i in on if i not in off])
        off =set([i for i in off if i not in on])
        on_series = pd.Series([1]*len(set(on)),index=on,dtype='float64')
        off_series = pd.Series([0]*len(set(off)),index=off,dtype='float64')
        s = pd.concat((on_series,off_series)).sort_index()
        status, intervals, current = 0, [], [None,None]
        for n in s.index:
            if status == 0 and s[n] == 1:
                status = 1
                current[0] = n
            if status == 1 and s[n] == 0:
                status = 0
                current[1] = n
                intervals.append(current)
                current = [None,None]
        if current != [None,None]:
            current[1] = end
            intervals.append(current)
        return [tuple(i) for i in intervals if i[0]-i[1] != 0]


    def _interval_is_close_to(self,intervals,events,closed='right'):
        """Return a list of intervals each near of at least one event from the specified event array (near being defined by treshold)."""
        result = [] ; intervals = pd.arrays.IntervalArray.from_tuples(intervals,closed=closed)
        for n in range(len(intervals)):
            if (abs(events -  intervals[n].left) < self.close_interval).any():
                result.append((intervals[n].left,intervals[n].right))
        return result

    def _set_element(self,event_array,intervals,is_element=True):
        """Return events (inputed as list or array) that are elements (is_element=True) or not (is_element=False) of intervals (tuple list or pd.intervalarray)."""
        if type(intervals) != pd.core.arrays.interval.IntervalArray:
            intervals = pd.arrays.IntervalArray.from_tuples(intervals,closed='left')
        if is_element:
            return np.array([event for event in event_array if intervals.contains(event).any()])
        else:
            return np.array([event for event in event_array if not intervals.contains(event).any()])

    def _set_operations(self,A,B,operation):
        """Do basic set operations with two sets lits, A and B, given as list of interval limits.

        Operation :
        - union               A U B
        - intersection        A n B
        """
        if (A == []) or (B == []):
            if operation == 'intersection':
                return []
            if operation == 'union':
                if A == []:
                    return B
                if B == []:
                    return A
        if A == B: return A
        left  = np.sort( list(set([i[0] for i in A] + [i[0] for i in B])) )
        right = np.sort( list(set([i[1] for i in A] + [i[1] for i in B])) )
        pA = pd.arrays.IntervalArray.from_tuples(A,closed="both")
        pB = pd.arrays.IntervalArray.from_tuples(B,closed="both")
        if operation == 'intersection':
            result = list(zip([l for l in left if (pA.contains(l).any() and pB.contains(l).any())],
                              [r for r in right if (pA.contains(r).any() and pB.contains(r).any())]))
        if operation == 'union':
            result = list(zip([l for l in left if not(pA.contains(l).any() and pB.contains(l).any())],
                              [r for r in right if not(pA.contains(r).any() and pB.contains(r).any())]))
        return [(left,right) for left,right in result if left-right != 0]

    def _union(self,*sets):
        """Find union of sets, using self._set_operations(union)."""
        if len(sets) == 1: return sets[0] #intersection of an ensemble with itself is itself
        union = self._set_operations(sets[0],sets[1],'union')
        if len(sets) == 2: return union
        else:
            for i in range(2,len(sets)):
                union = self._set_operations(union,sets[i],'union')
        return union

    def _intersection(self,*sets):
        """Find intersection of sets, using self._set_operations(intersection)."""
        if len(sets) == 1: return sets[0] #intersection of an ensemble with itself is itself
        intersection = self._set_operations(sets[0],sets[1],'intersection')
        if len(sets) == 2: return intersection
        else:
            for i in range(2,len(sets)):
                intersection = self._set_operations(intersection,sets[i],'intersection')
        return intersection

    def _non(self,A,end):
        """Return non A for a set A inputed as list of tuples defining time interval limits."""
        if A == []:
            return [(self.start,self.end)]
        sides = [i for a in A for i in a]
        if sides[-1] < end: sides.append(end)
        if sides[0] != 0:
            sides.insert(0,0)
            return list(zip(sides[::2],sides[1::2]))
        else:
            return list(zip(sides[1::2],sides[2::2]))

    def _translate(self,obj):
        """Translate strings into corresponding arrays."""
        if type(obj) == str:
            try: return self.__dict__[obj]
            except KeyError: print(f'Acceptable keys: {self.elements.keys()}')
        else: return obj

    def _internal_selection(self,obj):
        """Transform string into corresponding data."""
        if type(obj) == str:
            obj = [self._translate(obj)]                         # events arrays
        if type(obj) == np.ndarray:
            return [obj]                              # events arrays
        if type(obj) in [list,tuple]:
            return [self._translate(i) for i in obj]
        else:
            return []

    def _graph(self, ax,
                 obj,
                 label = None,
                 color = None,
                 demo  = True,
                 unit  = 'min',
                 x_lim = 'default'):
        factor = {k:v*({'s':1,'ms':1000}[self.user_unit]) for k,v in {'ms' : 0.001, 's': 1, 'min': 60, 'h': 3.6*10**3}.items()}[unit]
        if x_lim == 'default': x_lim = (self.start/factor,self.end/factor)
        data = self._translate(obj)
        # Choosing label
        if type(obj) == str and label == None:
            label = obj
        elif label:
            pass
        else:
            label == None
        # Plotting intervals
        if type(data) == list:
            if not label:
                label_list = [i for i in [k for k,v in self.__dict__.items() if type(v)==list] if data == self.__dict__[i]]
                if len(label_list) == 1: label = label_list[0]
            if label in self.elements.keys():
                if not color: color = self.elements[label][-1]
                if demo: label = self.elements[label][1]
            if len(data) == 0:
                ax.axvspan(0,0,label=label)
            for n,interval in enumerate(data):
                if color: color = self._list(color)[n%len(self._list(color))]
                ax.axvspan(interval[0]/factor,interval[1]/factor, label='_'*n+label,color=color)
        # Plotting events
        elif type(data) == np.ndarray:
            if not label:
                label = [k for k,v in self.events.items() if np.array_equal(data,v)][0]
            if label in self.elements.keys():
                if not color: color = self.elements[label][-1]
                if demo: label = self.elements[label][1]
            ax.eventplot(data/factor,linelengths=1,lineoffsets=0.5,colors=color,linewidth=0.6,label=label)
        else:
            pass
        ax.legend()
        ax.set_xlim(x_lim)
        ax.set_ylim((0,1))
        ax.axes.yaxis.set_visible(False)

    def _custom_events_intervals(self):
        if self.custom in ['fiber','all']:
            self.switch_d_nd      = np.array([i for i in [start for start,end in self.HLED_ON] if i in [end for start,end in self.LED2_ON]])
            self.switch_to_nd     = np.array([i for i in [start for start,end in self.HLED_ON] if i in [end for start,end in self.TIMEOUT]])
            self.switch_nd_d      = np.array([i for i in [end for start,end in self.HLED_ON] if i in [start for start,end in self.LED2_ON]])
            for i in range(self.fixed_ratio):
                self.__dict__[f'np1_{i+1}'] = self._set_element(self.np1,self._non(self.TIMEOUT,self.end))[i::self.fixed_ratio]
            for n,t in enumerate(self.HLED_OFF):
                self.__dict__[f'D_{n+1}']   = [t] 
            for n,t in enumerate(self.HLED_ON):
                self.__dict__[f'ND_{n+1}']  = [t] 
        if self.custom in ['movement','all']:
            self.x      = self.get(idtuple=(9,1))['_X'].to_numpy()
            self.y      = self.get(idtuple=(9,1))['_Y'].to_numpy()
            self.xytime = self.get(idtuple=(9,1))['TIME'].to_numpy()

 ###################### USER FUNCTIONS ##########################

    def movement(self,values=False,plot=True,figsize=(20,10),cmap='seismic'):
        """Show number of crossings."""
        if self.custom not in ['all','movement']: return
        array = np.zeros((max(self.y),max(self.x)))
        for i in range(len(self.x)):
            array[ self.y[i] -1,
                   self.x[i] - 1] += 1
        fig,ax = plt.subplots(1,figsize=figsize)
        ax.imshow(array,cmap=cmap,vmin=0,vmax=np.max(array))
        ax.invert_yaxis()
                

    def get(self,name=None,idtuple=None):
        """Extract dataframe section corresponding to selected counter name ('name') or tuple ('idtuple')."""
        if type(self.config) != str:
            nomenclature = {}
            for i in [self.config['IMETRONIC'][d] for d in self.config['IMETRONIC'].keys()]: nomenclature.update(i)
            if name:
                name = name.upper()
                if name in nomenclature.keys():
                    return self.df[(self.df['F']  == int(nomenclature[name][0])) & (self.df['ID'] == int(nomenclature[name][1]))]
        else:
            print('No configuration file, use idtuple')
        if idtuple: return self.df[(self.df['F']  == int(idtuple[0])) & (self.df['ID'] == int(idtuple[1]))]
        else: return self.df


    def figure(self,obj,
              figsize = 'default',h=0.8,hspace=0,label_list=None,color_list=None, **kwargs):
        """Take either string, single events/intervals (or list of either), or data_dictionnary as input and plots into one graph."""
        if type(obj) == dict:
            label_list  = list(obj.keys())
            obj_list    = [b[0] for a,b in obj.items()]
            color_list  = [b[1] for a,b in obj.items()]
        else:
            obj_list = self._list(obj)
            #
        if not label_list: label_list = [None]*len(obj_list)
        if not color_list: color_list = [None]*len(obj_list)
        # plotting
        if figsize == 'default': figsize = 20,h*len(obj_list)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(len(obj_list), hspace=hspace)
        axes = gs.subplots(sharex=True, sharey=True)
        if len(obj_list) == 1:
            self._graph(axes, obj_list[0],label=label_list[0], color=color_list[0], **kwargs)
        else:
            for n,ax in enumerate(axes):
                self._graph(ax, obj_list[n], label=label_list[n], color=color_list[n], **kwargs)


    def summary(self, demo=True,**kwargs):
        """Return a graphical summary of main events and intervals (can be configured in config.yaml)."""
        full_list = [i for i in self.elements.keys() if i in self.__dict__.keys()]
        sel_list = [i for i in full_list if self.elements[i][0]]
        self.figure(sel_list,**kwargs)

    def timestamps(self,
                   events,
                   interval      = 'all',
                   intersection  = [],
                   exclude       = [],
                   user_output=False):
        """events        : timestamp array, list of timestamp arrays, keyword or list of keywords (ex: 'np1')
           interval      : selected interval or list of intervals
           intersection  : intersection of inputed intervals
           exclude       : array or list of arrays to exclude
           to_csv        : True/False output csv timestamp file
           filename      : selected filemame for csv
           graph         : True/False visualise selection"""
        events_data = np.sort(np.concatenate(self._internal_selection(events)))
        if interval == 'all':
            interval_data = [(self.start,self.end)]
        else:
            interval_data = self._union(*self._internal_selection(interval))
        selected_interval = interval_data
        if intersection != []:
            intersection_data = self._intersection(*self._internal_selection(intersection))
            selected_interval = self._intersection(selected_interval,intersection_data)
        else:
            intersection_data = []
        if exclude  != []:
            exclude_data = self._union(*self._internal_selection(exclude))
            selected_interval = self._intersection(selected_interval,self._non(exclude_data,end=self.end))
        else:
            exclude_data = []
        selected_timestamps = self._set_element(events_data,selected_interval,is_element=True)
        if user_output:
            return events_data, interval_data, intersection_data, exclude_data, selected_interval, selected_timestamps
        else:
            return selected_timestamps
    
    def export_timestamps(self,
                          events,
                          interval      = 'all',
                          intersection  = [],
                          exclude       = [],
                          to_csv        = True,
                          graph         = True,
                          filename      = 'default',
                          start_TTL1    = False,
                          **kwargs):
        """Create list of timestamps and export them."""
        events_data, interval_data, intersection_data, exclude_data, selected_interval, selected_timestamps = self.timestamps(events,interval,intersection,exclude,user_output=True)   
        if graph:
            data = {f"Event(s):     {','.join([self.elements[i][1] for i in self._list(events)])}"      : (events_data,               'k'),#'r'),
                    f"Interval(s):  {','.join([self.elements[i][1] for i in self._list(interval)])}"    : (interval_data,             'g'),#'g'),
                    f"Intersection: {','.join([self.elements[i][1] for i in self._list(intersection)])}": (intersection_data,  'darkgrey'),#'#069AF3'),
                    f"Excluded:     {','.join([self.elements[i][1] for i in self._list(exclude)])}"     : (exclude_data,              'r'),#'k'),
                    "Selected interval(s):"                                                             : (selected_interval,         'y'),#'orange'),
                    "Selected timestamp(s):"                                                            : (selected_timestamps,       'g')}#'darkorange')}
            data_dict = {k:v for k,v in data.items() if v[0] != []}
            self.figure(data_dict,**kwargs)
        if start_TTL1:
            start = self.rec_start
        else:
            start = 0
        result = (selected_timestamps)-start
        if to_csv:
            if filename == 'default':
                filename = f"{self.filepath.split('/')[-1].split('.dat')[0]}.csv"
            else:
                if filename[-3:] != 'csv': filename += f'{self.rat_ID}.csv'
            pd.DataFrame({'timestamps': result}).to_csv(filename,index=False)
        return result
    
    def events(self,recorded=False,window=(0,0),window_unit='default'):
        """Retrieve list of events. Optionally only those which can be used in a perievent analysis (ie during the recording period and taking into account a perievent window)."""
        events = {k:v for k,v in self.__dict__.items() if type(v)==np.ndarray}
        if window_unit == 'default': window_unit = self.user_unit
        if not recorded:
            return events
        else:
            if window_unit != self.user_unit:
                if self.user_unit == 'ms' and window_unit == 's': window = [i/1000 for i in window]
                elif self.user_unit == 's' and window_unit == 'ms': window = [i*1000 for i in window]
            recorded_and_window = [(a+window[0],b-window[1]) for a,b in self.TTL1_ON]    
            return {k: self._set_element(v,recorded_and_window,is_element=True) for k,v in events.items()}
        
    def intervals(self,recorded=False,window=(0,0),window_unit='default'):
        """Retrieve list of intervals."""
        intervals = {k:v for k,v in self.__dict__.items() if type(v)==list}
        if window_unit == 'default': window_unit = self.user_unit
        if not recorded:
            return intervals
        else:
            if window_unit != self.user_unit:
                if self.user_unit == 'ms' and window_unit == 's': window = [i/1000 for i in window]
                elif self.user_unit == 's' and window_unit == 'ms': window = [i*1000 for i in window]
            recorded_and_window = [(a+window[0],b-window[1]) for a,b in self.TTL1_ON]  
            return {k: self._intersection(v,recorded_and_window) for k,v in intervals.items()}


    