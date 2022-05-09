import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from fp_utils import FiberPhotopy
import info
import seaborn as sns

class BehavioralData(FiberPhotopy):
    """Input : imetronic behavioral data file path."""
    vars().update(FiberPhotopy.BEHAVIOR)
    def __init__(self,
                 filepath,
                 **kwargs):
        """Initialize BehavioralData object, timestamp arrays are lowercase, periods are uppercase."""
        self.__dict__.update({k:v for k,v in locals().items() if k not in ('self','__class__','kwargs')})
        super().__init__(**kwargs)
        self._print(f'IMPORTING {filepath}...')
        start = time.time()
        self.df               = pd.read_csv(filepath,skiprows=12,delimiter='\t',header=None,names=['TIME','F','ID','_P','_V','_L','_R','_T','_W','_X','_Y','_Z'])
        self.df['TIME']      /= self.behavior_time_ratio
        self.start            = self.df.iloc[0,0]
        self.end              = self.df.iloc[-1,0]            # last timestamp (in ms) automatically changed to user_unit (see fp_utils.py)
        self.rec_start = None
        self._create_attributes()
        self._print(f'Importing finished in {np.round(time.time() - start,3)} seconds\n')

    @property    
    def raw(self):
        with open(self.filepath) as f:
            print(''.join(f.readlines()))
    
    @property
    def total(self):
       return pd.DataFrame({k : v.shape[0] for k,v in self.events().items()},index=['count']).T 
    
    @property
    def data(self):
        user_value = self._verbosity
        self._verbosity = False
        events = self.events().keys()
        intervals = self.intervals().keys()
        values = np.vstack([[self.timestamps(events=e,interval=i).shape[0] for e in self.events().keys()] for i in self.intervals()])
        df     = pd.DataFrame(values,index=intervals,columns=events).T
        self._verbosity = user_value
        return pd.concat((self.total,df),axis=1)
    
    def __repr__(self):
        """Return __repr__."""
        return f"""\
GENERAL INFORMATION:
******************************************************************************************************************************************************************
    Filename            : {self.filepath}
    Rat ID              : {self.rat_ID}
    Experiment duration : {self.end} (s)
    Time unit           : converted to seconds (ratio: {self.behavior_time_ratio})
    Fixed ratio         : FR{self.fixed_ratio}

(Need help?: <obj>.help"""

######################### HELPER FUNCTIONS ###########################

######################################################################################################
          #RAW EVENT TIMESTAMP
    def _create_attributes(self):
        # simple events
        for e,param in self.BEHAVIOR['imetronic_events'].items():
            self._print(f"Detecting {e+'...':<30}  {param})")
            if param[0] == 'conditional':
                (f,i),(c,v) = param[1:]
                self.__dict__[e] = self._extract(f,i,c,v)
            if param[0] == 'simple':
                (f,i),c = param[1:]
                self.__dict__[e] = self.get((f,i))[c].to_numpy()
        # local function, translates from string to data, including special nomenclature       
        def t(inp):
            if type(inp) == str:
                return self._translate(inp)
            if type(inp) in [float,int]:
                return inp
            elif type(inp) == list:
               return [t(i) for i in inp]
            else:
                print('inp',inp)

        # simple intervals, only need events    
        for e,param in self.BEHAVIOR['basic_intervals'].items():
            self._print(f"Detecting {e+'...':<30}  {param})")
            if param[0] == 'ON_OFF':
                w,(a,b) = param[1:]
                ON = self._interval(t(a),t(b),self.end)
                if w == 'on'  or w == 'both': self.__dict__[e+'_ON'] = ON
                if w == 'off' or w == 'both': self.__dict__[e+'_OFF'] = self._non(ON,self.end)
        # custom events an intervals, function defined first as generation is sequential and depend on a specific order       
        def _custom_gen(e,param):
            p,a,*b = param
            self._print(f"Detecting {e+'...':<30}  {param})")
            if p == 'INTERSECTION':  self.__dict__[e] = self._intersection(*t(a))
            if p == 'NEAR_EVENT':    self.__dict__[e] = self._interval_is_close_to(**{k:v for k,v in zip(['intervals','events','nearness'],t([a]+b))})
            if p == 'DURATION':      self.__dict__[e] = self._select_interval_by_duration(t(a),b)
            if p == 'UNION':         self.__dict__[e] = self._union(*t(a))
            if p == 'boundary': 
                self.__dict__[e] = np.array([i if a == 'start' else j for i,j in t(*b)])  
            if p == 'combination':   self.__dict__[e] = np.unique(np.sort(np.concatenate(t(a))))
            if p == 'indexed':       self.__dict__[e] = t(a)[b[0]-1 : b[0]]
            if p == 'iselement':     self.__dict__[e] =  self._set_element(t(a), t(*b))
            if p == 'timerestricted':self.__dict__[e] = t(a)[(t(a) > b[0][0])&(t(a) < b[0][1])]
            if p == 'generative':    self.__dict__.update({e.replace('_n',f'_{str(i+1)}') : t(a)[i::b[0]] for i in range(b[0])})
            if p == 'GENERATIVE':    self.__dict__.update({e.replace('_n',f'_{str(i+1)}') : [n] for i,n in enumerate(t(a))})
        for e,param in self.BEHAVIOR['custom'].items(): _custom_gen(e,param)
            
 ######################################################################################################           
    def _extract(self,family,subtype,column,value):
        """Extract timestamps for counters from imetronic file, first and second values indentify datatype, additional column precise event of interest (start,end, etc.)."""
        return     self.df[(self.df['F']    ==  family)  &
                          (self.df['ID']   ==  subtype) &
                          (self.df[column] ==  value)]['TIME'].to_numpy()

    def _select_interval_by_duration(self,interval,condition):
        if condition[0] == '<': return [(a,b) for a,b in interval if (b-a)<condition[1]]
        elif condition[0] == '>': [(a,b) for a,b in interval if (b-a)>condition[1]] 
        elif condition[0] == '=': [(a,b) for a,b in interval if (b-a)==condition[1]]
        else: print('Invalid input for select by interval duration.')
        
    def _interval(self,on,off,end):
        on  = list(set([i for i in on if i not in off]))
        off = list(set([i for i in off if i not in on]))
        # if not len(on): return []
        # if not len(off): return [(on[0]),end]
        # while off[0] < on[0]:
        #     off = off[1:]
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


    def _interval_is_close_to(self,intervals,events,nearness,closed='right'):
        """Return a list of intervals each near of at least one event from the specified event array (near being defined by treshold)."""
        result = [] ; intervals = pd.arrays.IntervalArray.from_tuples(intervals,closed=closed)
        for n in range(len(intervals)):
            if (abs(events -  intervals[n].left) < nearness).any():
                result.append((intervals[n].left,intervals[n].right))
        return result

    def _set_element(self,event_array,intervals,is_element=True,boolean=False):
        """Return events (inputed as list or array) that are elements (is_element=True) or not (is_element=False) of intervals (tuple list or pd.intervalarray)."""
        if type(intervals) != pd.core.arrays.interval.IntervalArray:
            intervals = pd.arrays.IntervalArray.from_tuples(intervals,closed='both')#'left')
        if is_element:
            res = np.array([event for event in event_array if intervals.contains(event).any()])
        else:
            res = np.array([event for event in event_array if not intervals.contains(event).any()])
        if not boolean: return res
        if boolean and is_element:     return len(res) >  0
        if boolean and not is_element: return len(res) == 0
        else: self._print(f'Something went wrong with _set_element method ({self.__class__})') 
            

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
        if len(sets) == 1:
            return sets[0] #intersection of an ensemble with itself is itself
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
        """Return non A for any set A inputed as list of tuples defining time interval limits."""
        if len(A) == 0:
            return [(self.start,self.end)]
        if A == [(self.start,self.end)]: return []
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
            if '~' in obj:
                obj = obj[1:] ; non = True
            else: non = False
            try:
                if non:
                    return self._non(self.__dict__[obj],self.end)
                else:
                    return self.__dict__[obj]
            except KeyError: (f'Acceptable keys: {self.elements.keys()}')
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
                 x_lim = 'default',
                 alpha = 1):
        factor = {'ms' : 0.001, 's': 1, 'min': 60, 'h': 3.6*10**3}[unit]
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
                if not label: label = ''
                ax.axvspan(interval[0]/factor,interval[1]/factor, label='_'*n+label,color=color,alpha=alpha)
        # Plotting events
        elif type(data) == np.ndarray:
            if not label:
                label = [k for k,v in self.events().items() if np.array_equal(data,v)][0]
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


 ###################### USER FUNCTIONS ##########################

    def _debug_interinj(self):
        """Return interinfusion time (between 2 consecutive pump activations)."""
        a = self.get((6,1))
        return np.mean(abs(a['TIME'][a['_L'] == 1].to_numpy() - a['TIME'][a['_L'] == 2].to_numpy()))

    def movement(self,values=False,plot=True,figsize=(20,10),cmap='seismic'):
        """Show number of crossings."""
        array = np.zeros((max(self.y_coordinates),max(self.x_coordinates)))
        for i in range(len(self.x_coordinates)):
            array[ self.y_coordinates[i] -1,
                   self.x_coordinates[i] - 1] += 1
        fig,ax = plt.subplots(1,figsize=figsize)
        ax.imshow(array,cmap=cmap,vmin=0,vmax=np.max(array))
        ax.invert_yaxis()

    def what_data(self,plot=True,figsize=(20,40)):
        """Return dataframe summarizing the imetronic dat file."""
        d = {}
        for k in self.SYSTEM['IMETRONIC'].keys():
            d.update(self.SYSTEM['IMETRONIC'][k])
        elements = {}
        detected_tuples = list(set(zip(self.df.F,self.df.ID)))
        unnamed_tuples = sorted([i for i in detected_tuples if i not in [tuple(i) for i in d.values()]])
        unnamed_dict = {k:list(k) for k in unnamed_tuples}
        for k,v in {**d,**unnamed_dict}.items():
            df = self.get(v)
            if len(df):
                elements[k] = [len(df)] + [round(np.mean(df[i]),3) if np.mean(df[i]) != 0 else '' for i in df.columns]
        data = pd.DataFrame(elements,index=['count']+list(self.df.columns)).T.sort_values('count',ascending=False)
        if plot:
            names = data.index.values
            columns = data.columns.values[4:]
            fig,axes = plt.subplots(len(names),figsize=figsize)
            for n,ax in enumerate(axes):
                name = names[n]
                df = self.get(name)
                for c in columns:
                    ax.scatter(df['TIME']/60_000,df[c],label=c.split('_')[1])
                ax.legend()
                ax.title.set_text(name)
                ax.set_xlim((0, max([i for i in data['TIME'] if type(i) != str])/60_000))
        return data

    def get(self,name):
        """Extract dataframe section corresponding to selected counter name ('name') or tuple ('idtuple')."""
        if type(name) == str:
            nomenclature = {}
            for i in [self.SYSTEM['IMETRONIC'][d] for d in self.SYSTEM['IMETRONIC'].keys()]: nomenclature.update(i)
            if name:
                name = name.upper()
                if name in nomenclature.keys():
                    return self.df[(self.df['F']  == int(nomenclature[name][0])) & (self.df['ID'] == int(nomenclature[name][1]))]
        elif type(name) in [list,tuple]: 
            return self.df[(self.df['F']  == int(name[0])) & (self.df['ID'] == int(name[1]))]
        else: 
            return self.df

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
                   length        = False,
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
        self._print(f'Event timestamps: {events_data}')
        if interval == 'all':
            interval_data = [(self.start,self.end)]
            self._print(f'Choosen interval: {interval_data} (all)')
        else:
            interval_data = self._union(*self._internal_selection(interval))
        selected_interval = interval_data
        if intersection != []:
            intersection_data = self._intersection(*self._internal_selection(intersection))
            selected_interval = self._intersection(selected_interval,intersection_data)
            self._print(f'Intersection of {intersection}: {intersection_data}')
        else:
            intersection_data = []
        if exclude  != []:
            exclude_data = self._union(*self._internal_selection(exclude))
            selected_interval = self._intersection(selected_interval,self._non(exclude_data,end=self.end))
            self._print(f"Excluded intervals: {exclude}: selected {exclude_data}")
        else:
            exclude_data = []
        if length:
            selected_interval = [(a,a+length) for a,b in selected_interval]
            self._print(f"Intervals restricted to {length} seconds.")
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
            data = {f"Event(s):     {','.join([self.elements[i][1] for i in self._list(events)])}"      : (events_data,               'r'),          #'k'),#
                    f"Interval(s):  {','.join([self.elements[i][1] for i in self._list(interval)])}"    : (interval_data,             'g'),          #'g'),#
                    f"Intersection: {','.join([self.elements[i][1] for i in self._list(intersection)])}": (intersection_data,   '#069AF3'),    #'darkgrey'),#
                    f"Excluded:     {','.join([self.elements[i][1] for i in self._list(exclude)])}"     : (exclude_data,              '#069AF3'),          #'r'),#
                    "Selected interval(s):"                                                             : (selected_interval,         'orange'),     #'y'),#
                    "Selected timestamp(s):"                                                            : (selected_timestamps,       'darkorange')} #'g')}#
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

    def events(self,recorded=False,window=(0,0)):
        """Retrieve list of events. Optionally only those which can be used in a perievent analysis (ie during the recording period and taking into account a perievent window)."""
        events = {k:v for k,v in self.__dict__.items() if type(v)==np.ndarray}
        #if window_unit == 'default': window_unit = 's'
        if not recorded:
            return events
        else:
            recorded_and_window = [(a+window[0],b-window[1]) for a,b in self.TTL1_ON]
            return {k: self._set_element(v,recorded_and_window,is_element=True) for k,v in events.items()}

    def intervals(self,recorded=False,window=(0,0)):
        """Retrieve list of intervals."""
        intervals = {k:v for k,v in self.__dict__.items() if type(v)==list and k[0]!='_'}
        if not recorded:
            return intervals
        else:
            recorded_and_window = [(a+window[0],b-window[1]) for a,b in self.TTL1_ON]
            return {k: self._intersection(v,recorded_and_window) for k,v in intervals.items()}


class MultiBehavior(FiberPhotopy):

    _savgol = FiberPhotopy._savgol

    def __init__(self,folder,**kwargs):
        super().__init__()
        self.sessions = {}
        self.foldername = folder
        self.paths = []
        for currentpath, folders, files in os.walk(folder):
            for file in files:
                path = os.path.join(currentpath, file)
                if path[-3:] == 'dat':
                    self.paths.append(path)
                    self.sessions[path] = BehavioralData(path,**kwargs)
        self.names = list(self.sessions.keys())
        self.number = len(self.sessions.items())
        event_names = list(list(self.sessions.items())[0][1].events().keys())
        for name,obj in self.sessions.items():
            for attr,val in obj.__dict__.items():
                if attr in self.__dict__.keys():
                    try:
                        self.__dict__[attr].append(val)
                    except AttributeError:
                        if attr+'_all' in self.__dict__.keys():
                            self.__dict__[attr+'_all'].append(val)
                        else:
                            self.__dict__[attr+'_all'] = [val]
                else:
                    self.__dict__[attr] = [val]
        for attr,val in self.__dict__.items():
            if attr in event_names:
                self.__dict__[attr] = pd.DataFrame(val,index=self.names)
    
    def __repr__(self): return f"<MultiBehavior object> // {self.folder}"
    
    def _cnt(self,attribute):
        return {k: np.histogram(self.__dict__[attribute].loc[k,:].dropna().to_numpy(), bins=round(self.sessions[k].end)+1, range=(0, round(self.sessions[k].end)+1))[0] for k in self.names}

    def count(self,attribute):
        return pd.DataFrame({ k:pd.Series(v) for k,v in self._cnt(attribute).items()}).T

    def cumul(self,attribute,plot=True,figsize=(20,15),**kwargs):
        cumul = pd.DataFrame({ k:pd.Series(np.cumsum(v)) for k,v in self._cnt(attribute).items()})
        if plot: cumul.plot(figsize=figsize,**kwargs)
        return cumul.T

    def show_rate(self,attribute,interval='HLED_ON',binsize=120,percentiles=[15,50,85],figsize=(20,10),interval_alpha=0.3):
        plt.figure(figsize=figsize)
        dic = {}
        for name in self.count(attribute).index:
            count = self.count(attribute).loc[name,:].copy()
            count.dropna(inplace=True)
            count = count.values
            dic[name] = np.array([np.sum(count[n-binsize:n]) for n in range(binsize,len(count))])
            plt.plot(dic[name],linewidth=1,label=name)
            plt.legend()
        if interval:
            if type(interval) == str: interval = list(self.sessions.items())[0][-1].__dict__[interval]
            if len(interval):
                for a,b in interval:
                    plt.axvspan(a-binsize,b-binsize,alpha=interval_alpha)
        self.__dict__[attribute+'_rate'] = pd.DataFrame({k:pd.Series(v) for k,v in dic.items()}).T
        if percentiles:
            idx = ['p'+str(i) for i in percentiles]
            self.__dict__[attribute+'_percentiles'] = pd.DataFrame({k:[np.nan]*len(idx) for k in self.__dict__[attribute+'_rate'].columns},index=idx)
            for c in self.__dict__[attribute+'_rate'].columns:
                data = self.__dict__[attribute+'_rate'].loc[:,c].copy().values
                self.__dict__[attribute+'_percentiles'].loc[:,c] = np.nanpercentile(data,percentiles)
            self.__dict__[attribute+'_percentiles'].T.plot(figsize=figsize)
        if interval:
            if type(interval) == str: interval = list(self.sessions.items())[0][-1].__dict__[interval]
            if len(interval):
                for a,b in interval:
                    plt.axvspan(a-binsize,b-binsize,alpha=interval_alpha)

    def summary(self):
        for r in self.sessions.keys():
            self.sessions[r].summary()
            plt.title(r)
