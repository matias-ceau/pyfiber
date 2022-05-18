"""The pyfiber.behavior module define two classes: Behavior and MultiBehavior,
that extracts behavioral events from raw files. They possess a number of methods that are useful when dealing
with complex experimental paradigm. Intervals corresponding to periods between specified timestamps can be 
automatically calculated by defining them in the configuration file.
"""

import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from ._utils import PyFiber as PyFiber
import typing
from typing import List, Tuple, Union, Any

__all__ = [
        'Behavior',
        'MultiBehavior',
        'Intervals',
        'Events',
        'select_interval_by_duration',
        'generate_interval',
        'interval_is_close_to',
        'element_of',
        'set_operation',
        'set_intersection',
        'set_union',
        'set_non'
        ]

Intervals = List[Tuple[float,float]]
Events = np.ndarray
# FUNCTIONS

def select_interval_by_duration(interval: Intervals, condition: list) -> Intervals:
    """Select interval based on their duration.

    :param interval: interval as a list of tuples
    :type interval: ``Intervals``
    :param condition: user inputed condition (*e.g.* ``['<',25]``)
    :param type: ``list``
    :return: Filtered intervals
    :rtype: ``Intervals``

    This function selects intervals by their duration, the conditional argument is a list
    containing the operator and a float (it mainly reads from the configuration file)."""
    if condition[0] == '<':
        return [(a, b) for a, b in interval if (b-a) < condition[1]]
    elif condition[0] == '>':
        [(a, b) for a, b in interval if (b-a) > condition[1]]
    elif condition[0] == '=':
        [(a, b) for a, b in interval if (b-a) == condition[1]]
    else:
        print('Invalid input for select by interval duration.')

def generate_interval(on : Events, off: Events, end: float) -> Intervals:
    """Compute intervals based on a two lists of right and left limits. Discards redundant values (i.e. if two consecutive right limits are provided).

    :param on: left limits of desired intervals.
    :type on: ``Events``
    :param off: right limits of desired intervals.
    :type off: ``Events``
    :return: computed intervals
    :rtype: ``Intervals``

    .. code:: python
        
        # This function is primarly used internally by the Behavior class, but can be directly invoked
        >> on  = [0., 5., 20., 40]   
        >> off = [10., 11., 30., 50]
        # 5 and 11 are redundant and should thus be eliminated
        >> end = 60
        # The end should be inputed, if the last command is off, the interval can thus end at the end of the experiment
        >> pyfiber.behavior.generate_interval(on,off,end)
        [(0.0, 10.0), (20.0, 30.0), (40.0, 50.0)]

    .. note::
       This function is based on the metaphor of a light switch, hence the names 'on' and 'off'. However, a number of
       intervals can be computed with the same principle. Importantly, it assumes that everything is 'off' at the beginning
       of the experiment. If the interval of interest start at the beginning of the experiment, the complementary
       interval should be computed and its complement (the interval of interest) itself obtained with the ``behavior.set_non``
       fonction.
    
       """
    on = list(set([i for i in on if i not in off]))
    off = list(set([i for i in off if i not in on]))

    on_series = pd.Series([1]*len(set(on)), index=on, dtype='float64')
    off_series = pd.Series([0]*len(set(off)), index=off, dtype='float64')
   
    s = pd.concat((on_series, off_series)).sort_index()
    status, intervals, current = 0, [], [None, None]
  
    for n in s.index:
        if status == 0 and s[n] == 1:
            status = 1
            current[0] = n
        if status == 1 and s[n] == 0:
            status = 0
            current[1] = n
            intervals.append(current)
            current = [None, None]
    if current != [None, None]:
        current[1] = end
        intervals.append(current)

    return [tuple(i) for i in intervals if i[0]-i[1] != 0]

def interval_is_close_to(intervals : Intervals,
                         events : Events,
                         nearness : float,
                         closed : str='right') -> Intervals:
    """Return a list of intervals each near of at least one event from the specified event array.

    :param intervals: evaluated intervals
    :type intervals: ``Intervals``
    :param events: events of which intervals must be near
    :type events: ``Events``
    :param nearness: maximum distance from interval to events (in seconds)
    :type nearness: ``float``
    :param closed: see ``pandas.IntervalArray`` for details
    :type closed: ``str``
    :return: selected intervals
    :rtype: ``Intervals``

    This function is used to calculate some of the custom events. For example, if the user wishes to isolate
    timestamps of particular events only if they happen near a period where a certain light is on."""
    result = []
    intervals = pd.arrays.IntervalArray.from_tuples(
        intervals, closed=closed)
    for n in range(len(intervals)):
        if (abs(events - intervals[n].left) < nearness).any():
            result.append((intervals[n].left, intervals[n].right))
    return result

def element_of(event_array : Events,
               intervals : Intervals,
               is_element : bool =True,
               boolean : bool =False) -> Union[bool,Events]:
    """Return events that are elements (or are not) of specified intervals.

    :param event_array: input list of event timestamps
    :type event_array: ``Events``
    :param intervals: intervals that should (or shouldn't) include events
    :type intervals: ``Intervals``
    :param is_element: ``True`` if intervals should be in intervals, ``False`` if they should't
    :type is_element: ``bool``
    :param boolean: if ``True``, the function return type is a boolean, else it is the list of timestamps that pass the condition
    :type boolean: ``bool``
    :return: Either the timestamps or a boolean
    :rtype: ``Events`` or ``bool``"""
    if type(intervals) != pd.core.arrays.interval.IntervalArray:
        intervals = pd.arrays.IntervalArray.from_tuples(
            intervals, closed='both')  # 'left')
    if is_element:
        res = np.array(
            [event for event in event_array if intervals.contains(event).any()])
    else:
        res = np.array(
            [event for event in event_array if not intervals.contains(event).any()])
    if not boolean:
        return res
    if boolean and is_element:
        return len(res) > 0
    if boolean and not is_element:
        return len(res) == 0

def set_operations(A : Intervals, B : Intervals, operation : str) -> Intervals:
    """Compute the union or the intersection of list of sets.

    :param A: Intervals A
    :param B: Intervals B
    :type A: ``Intervals``
    :type B: ``Intervals``

    .. note::
        - **union**               A U B
        - **intersection**        A n B
    """
    if (A == []) or (B == []):
        if operation == 'intersection':
            return []
        if operation == 'union':
            if A == []:
                return B
            if B == []:
                return A
    if A == B:
        return A
    left = np.sort(list(set([i[0] for i in A] + [i[0] for i in B])))
    right = np.sort(list(set([i[1] for i in A] + [i[1] for i in B])))
    pA = pd.arrays.IntervalArray.from_tuples(A, closed="both")
    pB = pd.arrays.IntervalArray.from_tuples(B, closed="both")
    if operation == 'intersection':
        result = list(zip([l for l in left if (pA.contains(l).any() and pB.contains(l).any())],
                          [r for r in right if (pA.contains(r).any() and pB.contains(r).any())]))
    if operation == 'union':
        result = list(zip([l for l in left if not(pA.contains(l).any() and pB.contains(l).any())],
                          [r for r in right if not(pA.contains(r).any() and pB.contains(r).any())]))
    return [(left, right) for left, right in result if left-right != 0]

def set_union(*sets : Intervals) -> Intervals:
    """Find union of any number of Intervals.

    :param sets: intervals to be united.
    :type sets: ``Intervals``
    :return: Union of all inputed Intervals
    :rtype: ``Intervals``
    """
    if len(sets) == 1:
        return sets[0]  # intersection of an ensemble with itself is itself
    union = set_operations(sets[0], sets[1], 'union')
    if len(sets) == 2:
        return union
    else:
        for i in range(2, len(sets)):
            union = set_operations(union, sets[i], 'union')
    return union

def set_intersection(*sets : Intervals) -> Intervals:
    """Find intersection of any number of Intervals.

    :param sets: intervals to be united.
    :type sets: ``Intervals``
    :return: Intersection of all inputed Intervals
    :rtype: ``Intervals``
    """
    if len(sets) == 1:
        return sets[0]  # intersection of an ensemble with itself is itself
    intersection = set_operations(sets[0], sets[1], 'intersection')
    if len(sets) == 2:
        return intersection
    else:
        for i in range(2, len(sets)):
            intersection = set_operations(
                intersection, sets[i], 'intersection')
    return intersection

def set_non(A : Intervals, end : float, start : float = 0) -> Intervals:
    """Returns the complement of a collection of sets.

    :param A: Intervals of which to compute the complement
    :type A: ``Intervals``
    :param end: End of experiment
    :type end: float
    :param start: start of experiment, default value is 0
    :type start: float
    :return: Complement of A
    :rtype: ``Intervals``
    """
    if len(A) == 0:
        return [(0, end)]
    if A == [(0, end)]:
        return []
    sides = [i for a in A for i in a]
    if sides[-1] < end:
        sides.append(end)
    if sides[0] != 0:
        sides.insert(0, 0)
        return list(zip(sides[::2], sides[1::2]))
    else:
        return list(zip(sides[1::2], sides[2::2]))


class Behavior(PyFiber):
    """Class extracting and analyzing behavioral data.

    :param filepath: filepath (by default must be a .dat file)
    :type filepath: str
    :param filetype: must be set to user-defined format (by default takes the value of 'BEHAVIOR_FILE_TYPE' in the configuration file
    :param kwargs: arguments passed to _utils.PyFiber, the parent class (ex: verbosity=False)

    :ivar filetype: input filetype (takes the default value defined in the config file)
    :ivar filepath: data file filepath
    :ivar df: ``pandas.DataFrame`` containing the extracted data
    :ivar start: first timestamp of the data file, if the input is a csv and no start is provided, it takes the value 0.0
    :ivar end: last timestamp of the data file, if the input is a csv file, there should be an 'end' column with a float value
    :ivar <other>: all generated timestamps and intervals get added as instance attribute (and can be refered to their name as strings in most functions)
    """
    vars().update(PyFiber.BEHAVIOR)

    def __init__(self,
                 filepath,
                 filetype=None,
                 **kwargs):
        self.__dict__.update(
            {k: v for k, v in locals().items() if k not in ('self', '__class__', 'kwargs')})
        super().__init__(**kwargs)

        if filetype:
            self.filetype = filetype # :ivar input filetype
        else:
            self.filetype = self.BEHAVIOR_FILE_TYPE

        self._print(f'IMPORTING {filepath}...')
        start = time.time()

        if self.filetype == 'IMETRONIC':
            self.df = pd.read_csv(filepath, skiprows=12, delimiter='\t', header=None, names=[
                                  'TIME', 'F', 'ID', '_P', '_V', '_L', '_R', '_T', '_W', '_X', '_Y', '_Z'])
            self.df['TIME'] /= self.behavior_time_ratio
            self.start = self.df.iloc[0, 0]
            # last timestamp (in ms) automatically changed to user_unit (see _utils.py)
            self.end = self.df.iloc[-1, 0]
            self.rec_start = None
            self._create_imetronic_attributes()

        else:
            self.df = pd.read_csv(filepath)
            for c in self.df.columns:
                c_ = ''.join(''.join(''.join(('_'.join(('_'.join(c.split('-'))).split(' '))).split('/')).split('(')).split(')'))
                self.__dict__[c_] = self.df[c].to_numpy()

        if 'start' not in self.__dict__:
            self.start = 0.

        self.all = [(self.start, self.end)]
        self._compute_attributes()
        self._print(
            f'Importing finished in {np.round(time.time() - start,3)} seconds\n'
                     )

    @property
    def raw(self):
        """Output the unprocessed data file."""
        with open(self.filepath) as f:
            print(''.join(f.readlines()))

    @property
    def total(self):
        """Return the count for all extracted event (*i.e.* if events a,b and c are added,
        return the total count of a,b and c.)"""
        return pd.DataFrame({k: v.shape[0] for k, v in self.events().items()}, index=['count']).T

    @property
    def data(self):
        """Return the count for all extracted event for all calculated intervals. (*i.e.* if events a,b and c, and
        intervals A, B, C are added, return the count of a,b and c for each of the A, B, C intervals.)"""
        user_value = self._verbosity
        self._verbosity = False
        events = self.events().keys()
        intervals = self.intervals().keys()
        values = np.vstack([[self.timestamps(events=e, interval=i).shape[0]
                           for e in self.events().keys()] for i in self.intervals()])
        df = pd.DataFrame(values, index=intervals, columns=events).T
        self._verbosity = user_value
        return pd.concat((self.total, df), axis=1)

    def __repr__(self):
        """Return __repr__."""
        return f"""\
GENERAL INFORMATION:
******************************************************************************************************************************************************************
    Filename            : {self.filepath}
    Animal ID           : {self.ID}
    Experiment duration : {self.end} (s)
    Time unit           : converted to seconds (ratio: {self.behavior_time_ratio})

(Need help?: <obj>.help"""

######################### HELPER FUNCTIONS ###########################

######################################################################################################
    def _extract(self, family : int, subtype: int, column: str, value: int) -> pd.DataFrame:
        """Extract timestamps for counters from Imetronic file.

        :param family: data family (see Imetronic nomenclature)
        :type family: int
        :param subtype: data subtype
        :type family: int
        :param column: additional column precising event of interest
        :type column: str
        :param value: value needed in column to extract it
        :return: data corresponding to input parameters
        :rtype: ``pd.DataFrame``
        """
        return self.df[(self.df['F'] == family) &
                       (self.df['ID'] == subtype) &
                       (self.df[column] == value)]['TIME'].to_numpy()

        # RAW EVENT TIMESTAMP
    def _create_imetronic_attributes(self):
        """Extract event specified in the configuration file, in the **imetronic_events** section.

        This function gets called during instance initialization if the filetype is 'IMETRONIC' (see configuration file)."""
        for e, param in self.BEHAVIOR['imetronic_events'].items():
            self._print(f"Detecting {e+'...':<30}  {param})")
            if param[0] == 'conditional':
                (f, i), (c, v) = param[1:]
                self.__dict__[e] = self._extract(f, i, c, v)
            if param[0] == 'simple':
                (f, i), c = param[1:]
                self.__dict__[e] = self.get((f, i))[c].to_numpy()
        # local function, translates from string to data, including special nomenclature

    
    def _compute_attributes(self):
        """Create rule-based intervals and events as specified in the configuration file (in sections **basic_intervals** and **custom**)"""
        def t(inp):
            if type(inp) == str:
                return self._translate(inp)
            if type(inp) in [float, int]:
                return inp
            elif type(inp) == list:
                return [t(i) for i in inp]
            else:
                print('inp', inp)

        # simple intervals, only need events
        if bool(self.BEHAVIOR['basic_intervals']):
            for e, param in self.BEHAVIOR['basic_intervals'].items():
                self._print(f"Detecting {e+'...':<30}  {param})")
                if param[0] == 'ON_OFF':
                    w, (a, b) = param[1:]
                    ON = generate_interval(t(a), t(b), self.end)
                    if w == 'on' or w == 'both':
                        self.__dict__[e+'_ON'] = ON
                    if w == 'off' or w == 'both':
                        self.__dict__[e+'_OFF'] = set_non(ON, self.end)

        # custom events an intervals, function defined first as generation is sequential and depend on a specific order
        if bool(self.BEHAVIOR['custom']):
            for e, param in self.BEHAVIOR['custom'].items():
                p, a, *b = param
                self._print(f"Detecting {e+'...':<30}  {param})")
                if p == 'INTERSECTION':
                    self.__dict__[e] = set_intersection(*t(a))
                if p == 'NEAR_EVENT':
                    self.__dict__[e] = interval_is_close_to(
                        **{k: v for k, v in zip(['intervals', 'events', 'nearness'], t([a]+b))})
                if p == 'DURATION':
                    self.__dict__[e] = select_interval_by_duration(t(a), b)
                if p == 'UNION':
                    self.__dict__[e] = set_union(*t(a))
                if p == 'boundary':
                    self.__dict__[e] = np.array(
                        [i if a == 'start' else j for i, j in t(*b)])
                if p == 'combination':
                    self.__dict__[e] = np.unique(np.sort(np.concatenate(t(a))))
                if p == 'indexed':
                    self.__dict__[e] = t(a)[b[0]-1: b[0]]
                if p == 'iselement':
                    self.__dict__[e] = element_of(t(a), t(*b))
                if p == 'timerestricted':
                    self.__dict__[e] = t(a)[(t(a) > b[0][0]) & (t(a) < b[0][1])]
                if p == 'generative':
                    self.__dict__.update(
                        {e.replace('_n', f'_{str(i+1)}'): t(a)[i::b[0]] for i in range(b[0])})
                if p == 'GENERATIVE':
                    self.__dict__.update(
                        {e.replace('_n', f'_{str(i+1)}'): [n] for i, n in enumerate(t(a))})
            

 ######################################################################################################

    def _translate(self, obj : str):
        """Translate strings into corresponding arrays.
        
        :param obj: name of the interval/event
        :type obj: str

        Equivalent to a lookup in ``self.__dict__`` with the addition of a call to the ``behavior.set_non``
        function if the ``~`` sign is prepended to the string."""
        if type(obj) == str:
            if '~' in obj:
                obj = obj[1:]
                non = True
            else:
                non = False
            try:
                if non:
                    return set_non(self.__dict__[obj], self.end)
                else:
                    return self.__dict__[obj]
            except KeyError:
                (f'Acceptable keys: {self.elements.keys()}')
        else:
            return obj

    def _internal_selection(self, obj : Any) -> list:
        """Transform string into corresponding data.
        
        :param obj: either the event/interval name or its data

        :return: object data as a list

        """
        if type(obj) == str:
            # events arrays
            obj = [self._translate(obj)]
        if type(obj) == np.ndarray:
            return [obj]                              # events arrays
        if type(obj) in [list, tuple]:
            return [self._translate(i) for i in obj]
        else:
            return []

    def _graph(self,
               ax,
               obj,
               label=None,
               color=None,
               demo=True,
               unit='min',
               x_lim='default',
               alpha=1):
        """Internal function for plotting on a single axis."""
        factor = {'ms': 0.001, 's': 1, 'min': 60, 'h': 3.6*10**3}[unit]
        if x_lim == 'default':
            x_lim = (self.start/factor, self.end/factor)
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
                label_list = [i for i in [k for k, v in self.__dict__.items() if type(
                    v) == list] if data == self.__dict__[i]]
                if len(label_list) == 1:
                    label = label_list[0]
            if label in self.elements.keys():
                if not color:
                    color = self.elements[label][-1]
                if demo:
                    label = self.elements[label][1]
            if len(data) == 0:
                ax.axvspan(0, 0, label=label)
            for n, interval in enumerate(data):
                if color:
                    color = self._list(color)[n % len(self._list(color))]
                if not label:
                    label = ''
                ax.axvspan(interval[0]/factor, interval[1]/factor,
                           label='_'*n+label, color=color, alpha=alpha)
        # Plotting events
        elif type(data) == np.ndarray:
            if not label:
                label = [k for k, v in self.events().items()
                         if np.array_equal(data, v)][0]
            if label in self.elements.keys():
                if not color:
                    color = self.elements[label][-1]
                if demo:
                    label = self.elements[label][1]
            ax.eventplot(data/factor, linelengths=1, lineoffsets=0.5,
                         colors=color, linewidth=0.6, label=label)
        else:
            pass
        ax.legend()
        ax.set_xlim(x_lim)
        ax.set_ylim((0, 1))
        ax.axes.yaxis.set_visible(False)

 ###################### USER FUNCTIONS ##########################

    def _debug_interinj(self):
        """Return interinfusion time (between 2 consecutive pump activations).

        Used to eliminate recording containing short interinfusion time."""
        a = self.get((6, 1))
        return np.mean(abs(a['TIME'][a['_L'] == 1].to_numpy() - a['TIME'][a['_L'] == 2].to_numpy()))


    def figure(self, obj,
               figsize='default', h=0.8, hspace=0, label_list=None, color_list=None, **kwargs):
        """Plot data.

        :param obj: data to plot
        :type obj: ``Events`` or ``Intervals``, list of such elements, str corresponding to such data in ``self.__dict__`` or dictionnary
        
        :return: ``matplotlib.pyplot.figure`` containing a **eventplots** or **axvspan** or a combination of both.

        Main plotting function, calling ``self._plot`` for each ``axis``, called itself by ``self.summary``"""
        if type(obj) == dict:
            label_list = list(obj.keys())
            obj_list = [b[0] for a, b in obj.items()]
            color_list = [b[1] for a, b in obj.items()]
        else:
            obj_list = self._list(obj)
            #
        if not label_list:
            label_list = [None]*len(obj_list)
        if not color_list:
            color_list = [None]*len(obj_list)
        # plotting
        if figsize == 'default':
            figsize = 20, h*len(obj_list)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(len(obj_list), hspace=hspace)
        axes = gs.subplots(sharex=True, sharey=True)
        if len(obj_list) == 1:
            self._graph(
                axes, obj_list[0], label=label_list[0], color=color_list[0], **kwargs)
        else:
            for n, ax in enumerate(axes):
                self._graph(
                    ax, obj_list[n], label=label_list[n], color=color_list[n], **kwargs)
        plt.show()

    def summary(self, demo : bool=True, **kwargs):
        """Return a predifined graphical summary of main events and intervals.
        
        :param demo: Demo mode (``True``). Provides a user defined summary of intervals and events. If ``False`` plots everything.
        :type demo: bool
        :param kwargs: Plotting arguments, passed to ``self.figure`` and ``self._plot``.

        The data and the formating provided by the **demo** option can be configured in pyfiber.yaml."""
        if demo:
            full_list = [i for i in self.elements.keys()
                         if i in self.__dict__.keys()]
            sel_list = [i for i in full_list if self.elements[i][0]]
        else:
            sel_list = [i for i in self.events().keys()] + [i for i in self.intervals().keys()]
        self.figure(sel_list, **kwargs)

    def timestamps(self,
                   events : Union[str, Events],
                   interval : Union[str, List[str], Intervals] ='all',
                   length : Union[float, int, bool] =False,
                   intersection : List[Intervals] = [],
                   exclude : List[Intervals] =[],
                   user_output: bool =False) -> Union[Events,tuple]:
        """Selects timestamps based on conditions.

        :param events: name or array of timestamps
        :type events: str or ``Events``
        
        :param interval: interval or intervals in which the timestamps must be contained, can be ``'all'``
        :type interval: ``str``, ``List[str]``, ``List[Intervals]``, ``Intervals`` 
        
        :param intersection: intersection of intervals in which the timestamps must be contained
        :type intersection: ``List[str]``, ``List[Intervals]``
        
        :param exclude: interval or intervals in which the timestamps must not be contained
        :type exclude: ``List[str]``, ``List[Intervals]`` 
        
        :param user_output: if ``True``, returns a tuple with all the user input data (for plotting purposes)
        :param length: distance from the start of the intervals defined in ``interval`` to be considered.
        
        :return: Timestamps or tuple with timestamps and considered intervals (depending on ``user_output``)
        :rtype: ``Events`` or tuple
        """
        events_data = np.sort(np.concatenate(self._internal_selection(events)))
        self._print(f'Event timestamps: {events_data}')
        if interval == 'all':
            interval_data = [(self.start, self.end)]
            self._print(f'Choosen interval: {interval_data} (any)')
        else:
            interval_data = set_union(*self._internal_selection(interval))
        selected_interval = interval_data
        if intersection != []:
            intersection_data = set_intersection(
                *self._internal_selection(intersection))
            selected_interval = set_intersection(
                selected_interval, intersection_data)
            self._print(f'Intersection of {intersection}: {intersection_data}')
        else:
            intersection_data = []
        if exclude != []:
            exclude_data = set_union(*self._internal_selection(exclude))
            selected_interval = set_intersection(
                selected_interval, set_non(exclude_data, end=self.end))
            self._print(
                f"Excluded intervals: {exclude}: selected {exclude_data}")
        else:
            exclude_data = []
        if length:
            selected_interval = [(a, a+length) for a, b in selected_interval]
            self._print(f"Intervals restricted to {length} seconds.")
        selected_timestamps = element_of(
            events_data, selected_interval, is_element=True)
        if user_output:
            return events_data, interval_data, intersection_data, exclude_data, selected_interval, selected_timestamps
        else:
            return selected_timestamps

    def export_timestamps(self,
                          events : Union[str, Events],
                          interval : Union[str, List[str], Intervals] ='all',
                          length : Union[float, int, bool] =False,
                          intersection : List[Intervals] = [],
                          exclude : List[Intervals] =[],
                          to_csv=True,
                          graph=True,
                          filename='default',
                          start_TTL1=False,
                          **kwargs):
        """User API for visualizing timestamp selection.

        :param events: name or array of timestamps
        :type events: str or ``Events``
        
        :param interval: interval or intervals in which the timestamps must be contained, can be ``'all'``
        :type interval: ``str``, ``List[str]``, ``List[Intervals]``, ``Intervals`` 
        
        :param intersection: intersection of intervals in which the timestamps must be contained
        :type intersection: ``List[str]``, ``List[Intervals]``
        
        :param exclude: interval or intervals in which the timestamps must not be contained
        :type exclude: ``List[str]``, ``List[Intervals]``

        Wrapper for the ``self.timestamps`` function, with added graphical representation and csv output.
        """
        (
        events_data,
        interval_data,
        intersection_data,
        exclude_data,
        selected_interval,
        selected_timestamps
        ) = self.timestamps(
                            events,
                            interval,
                            length,
                            intersection,
                            exclude,
                            user_output=True)
        if graph:
            try:
                events_key = f"Event(s):   {','.join([self.elements[i][1] for i in self._list(events)])}"
            except (KeyError,TypeError):
                events_key = "Event(s)"
            try:
                interval_key = f"Interval(s):  {','.join([self.elements[i][1] for i in self._list(interval)])}"
            except (KeyError,TypeError):
                interval_key = "Interval(s)"
            try:
                intersection_key = f"Intersection: {','.join([self.elements[i][1] for i in self._list(intersection)])}"
            except (KeyError,TypeError):
                intersection_key = "Intersection"
            try:
                exclude_key = f"Excluded:     {','.join([self.elements[i][1] for i in self._list(exclude)])}"
            except (KeyError,TypeError):
                exclude_key = "Excluded"

            data = {
                events_key: (events_data,'r'),
                interval_key: (interval_data, 'g'),
                intersection_key: (intersection_data, '#069AF3'),
                exclude_key: (exclude_data, '#069AF3'),
                "Selected interval(s):": (selected_interval, 'orange'),
                "Selected timestamp(s):": (selected_timestamps, 'darkorange')
                    }
            data_dict = {k: v for k, v in data.items() if v[0] != []}
            self.figure(data_dict, **kwargs)
        if start_TTL1:
            start = self.rec_start
        else:
            start = 0
        result = (selected_timestamps)-start
        if to_csv:
            if filename == 'default':
                filename = f"{self.filepath.split('/')[-1].split('.dat')[0]}.csv"
            else:
                if filename[-3:] != 'csv':
                    filename += f'{self.rat_ID}.csv'
            pd.DataFrame({'timestamps': result}).to_csv(
                os.path.join(Behavior.FOLDER, filename), index=False)
        return result

    def events(self,
        recorded : bool =False,
        recorded_name : str ='TTL1_ON',
        window : tuple =(0, 0)):
        """Retrieve list of events.
        
        :param recorded: only output events that are recorded(default value = ``False``)
        :type recorded: bool
        :param recorded_name: name of the attribute storing recorded intervals (default ``'TTL1_ON'``)
        :type recorded_name: bool
        :param window: perievent analysis window
        :return: plot

        .. note::
            Optionally only those which can be used in a perievent analysis (ie during the recording period
            and taking into account a perievent window)."""
        events = {k: v for k, v in self.__dict__.items() if type(v)
                  == np.ndarray}
        #if window_unit == 'default': window_unit = 's'
        if not recorded:
            return events
        else:
            recorded_and_window = [(a+window[0], b-window[1])
                                   for a, b in self.__dict__[recorded_name] ]
            return {k: element_of(v, recorded_and_window, is_element=True) for k, v in events.items()}

    def intervals(self, recorded=False, window=(0, 0)):
        """Retrieve list of intervals."""
        intervals = {k: v for k, v in self.__dict__.items() if type(v)
                     == list and k[0] != '_'}
        if not recorded:
            return intervals
        else:
            recorded_and_window = [(a+window[0], b-window[1])
                                   for a, b in self.TTL1_ON]
            return {k: set_intersection(v, recorded_and_window) for k, v in intervals.items()}

    def what_data(self, plot=True, figsize=(20, 40)):
        """Return dataframe summarizing the imetronic dat file."""
        d = {}
        for k in self.SYSTEM['IMETRONIC'].keys():
            d.update(self.SYSTEM['IMETRONIC'][k])
        elements = {}
        detected_tuples = list(set(zip(self.df.F, self.df.ID)))
        unnamed_tuples = sorted([i for i in detected_tuples if i not in [
                                tuple(i) for i in d.values()]])
        unnamed_dict = {k: list(k) for k in unnamed_tuples}
        for k, v in {**d, **unnamed_dict}.items():
            df = self.get(v)
            if len(df):
                elements[k] = [len(df)] + [round(np.mean(df[i]), 3)
                                           if np.mean(df[i]) != 0 else '' for i in df.columns]
        data = pd.DataFrame(elements, index=[
                            'count']+list(self.df.columns)).T.sort_values('count', ascending=False)
        if plot:
            names = data.index.values
            columns = data.columns.values[4:]
            fig, axes = plt.subplots(len(names), figsize=figsize)
            for n, ax in enumerate(axes):
                name = names[n]
                df = self.get(name)
                for c in columns:
                    ax.scatter(df['TIME']/60_000, df[c], label=c.split('_')[1])
                ax.legend()
                ax.title.set_text(name)
                ax.set_xlim(
                    (0, max([i for i in data['TIME'] if type(i) != str])/60_000))
        return data

    def get(self, name):
        """Extract dataframe section corresponding to selected counter name ('name') or tuple ('idtuple')."""
        if type(name) == str:
            nomenclature = {}
            for i in [self.SYSTEM['IMETRONIC'][d] for d in self.SYSTEM['IMETRONIC'].keys()]:
                nomenclature.update(i)
            if name:
                name = name.upper()
                if name in nomenclature.keys():
                    return self.df[(self.df['F'] == int(nomenclature[name][0])) & (self.df['ID'] == int(nomenclature[name][1]))]
        elif type(name) in [list, tuple]:
            return self.df[(self.df['F'] == int(name[0])) & (self.df['ID'] == int(name[1]))]
        else:
            return self.df

    def movement(self, values=False, plot=True, figsize=(20, 10), cmap='seismic'):
        """Show number of crossings."""
        array = np.zeros((max(self.y_coordinates), max(self.x_coordinates)))
        for i in range(len(self.x_coordinates)):
            array[self.y_coordinates[i] - 1,
                  self.x_coordinates[i] - 1] += 1
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(array, cmap=cmap, vmin=0, vmax=np.max(array))
        ax.invert_yaxis()
        plt.show()

class MultiBehavior(PyFiber):

    _savgol = PyFiber._savgol

    def __init__(self, folder, **kwargs):
        super().__init__()
        self.sessions = {}
        self.foldername = folder
        self.paths = []
        for currentpath, folders, files in os.walk(folder):
            for file in files:
                path = os.path.join(currentpath, file)
                if path[-3:] == 'dat':
                    self.paths.append(path)
                    self.sessions[path] = Behavior(path, **kwargs)
        self.names = list(self.sessions.keys())
        self.number = len(self.sessions.items())
        event_names = list(list(self.sessions.items())[0][1].events().keys())
        for name, obj in self.sessions.items():
            for attr, val in obj.__dict__.items():
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
        for attr, val in self.__dict__.items():
            if attr in event_names:
                self.__dict__[attr] = pd.DataFrame(val, index=self.names)

    def __repr__(self): return f"<MultiBehavior object> // {self.foldername}"

    def _cnt(self, attribute):
        return {k: np.histogram(self.__dict__[attribute].loc[k, :].dropna().to_numpy(), bins=round(self.sessions[k].end)+1, range=(0, round(self.sessions[k].end)+1))[0] for k in self.names}

    def count(self, attribute):
        return pd.DataFrame({k: pd.Series(v) for k, v in self._cnt(attribute).items()}).T

    def cumul(self, attribute, plot=True, figsize=(20, 15), **kwargs):
        cumul = pd.DataFrame({k: pd.Series(np.cumsum(v))
                             for k, v in self._cnt(attribute).items()})
        if plot:
            cumul.plot(figsize=figsize, **kwargs)
            plt.show()
        return cumul.T

    def show_rate(self, attribute, interval='HLED_ON', binsize=120, percentiles=[15, 50, 85], figsize=(20, 10), interval_alpha=0.3):
        plt.figure(figsize=figsize)
        dic = {}
        for name in self.count(attribute).index:
            count = self.count(attribute).loc[name, :].copy()
            count.dropna(inplace=True)
            count = count.values
            dic[name] = np.array([np.sum(count[n-binsize:n])
                                 for n in range(binsize, len(count))])
            plt.plot(dic[name], linewidth=1, label=name)
            plt.legend()
        if interval:
            if type(interval) == str:
                interval = list(self.sessions.items())[
                    0][-1].__dict__[interval]
            if len(interval):
                for a, b in interval:
                    plt.axvspan(a-binsize, b-binsize, alpha=interval_alpha)
        self.__dict__[
            attribute+'_rate'] = pd.DataFrame({k: pd.Series(v) for k, v in dic.items()}).T
        if percentiles:
            idx = ['p'+str(i) for i in percentiles]
            self.__dict__[attribute+'_percentiles'] = pd.DataFrame({k: [np.nan]*len(
                idx) for k in self.__dict__[attribute+'_rate'].columns}, index=idx)
            for c in self.__dict__[attribute+'_rate'].columns:
                data = self.__dict__[attribute+'_rate'].loc[:, c].copy().values
                self.__dict__[attribute+'_percentiles'].loc[:,
                                                            c] = np.nanpercentile(data, percentiles)
            self.__dict__[attribute+'_percentiles'].T.plot(figsize=figsize)
        if interval:
            if type(interval) == str:
                interval = list(self.sessions.items())[
                    0][-1].__dict__[interval]
            if len(interval):
                for a, b in interval:
                    plt.axvspan(a-binsize, b-binsize, alpha=interval_alpha)
        plt.show()

    def summary(self):
        for r in self.sessions.keys():
            self.sessions[r].summary()
            plt.title(r)
