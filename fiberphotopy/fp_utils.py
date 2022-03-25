import numpy as np
from scipy import signal
import pandas as pd
import yaml


class FiberPhotopy:
    """Parent object for Behavioral, Fiber and Analysis objects."""

    def __init__(self,object_type='general',configfile='../config.yaml',**kwargs):
                # GENERAL
        self.hello          = 'Hi'
        self.configfile     = configfile
        try:
            with open(configfile) as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            print('File not found')
            self.config = None
        self.__dict__.update(self.config['USER']['GENERAL'])
        if object_type == 'fiber':
            self.__dict__.update(self.config['USER']['FIBERPHOTOMETRY'])
        if object_type == 'behavior':
            self.__dict__.update(self.config['USER']['BEHAVIOR'])
        if object_type == 'all':
            for category in self.config['USER'].keys():
                self.__dict__.update(self.config['USER'][category])
        self.__dict__.update(**kwargs)

    def _list(self,anything):
        """Convert user input into list if not already."""
        if not anything: return []
        elif type(anything) in [str,int,float]: return [anything]
        elif type(anything) == np.ndarray: return anything.tolist()
        elif type(anything) == list: return anything
        else: print(f"{type(anything)} not taken care of by _list")
        
    def _savgol(self,data,window=10,polyorder=3,nosmoothing=False):
            if nosmoothing: return data
            window = round(window)
            if window%2 ==0:
                window += 1
            return signal.savgol_filter(data,window,polyorder)
