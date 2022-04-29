import numpy as np
from scipy import signal
import pandas as pd
import yaml
import datetime
import info


class FiberPhotopy:
    """Parent object for Behavioral, Fiber and Analysis objects."""
    try:
        with open('../config.yaml') as f:
            CFG = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print('File not found')
        CFG = None
    vars().update(CFG)
    vars().update(CFG['GENERAL'])
    vars().update(CFG['SYSTEM'])
            
    def __init__(self,verbosity=True,**kwargs):
                # GENERAL
        self.verbosity      = verbosity
        self.info           = []
        self._log           = []
        self.__dict__.update(**kwargs)      
        
    @property
    def log(self):
        print('\n'.join([' '.join(i.split('\n')) for i in self._log])+'\n')
    @log.setter
    def log(self,value):
        log = f"{datetime.datetime.now().strftime('%H:%M:%S')} --- {value}"
        self._log.append(log)
        
    @property
    def help(self):
        if self.type == 'BehavioralData':
            print(info.behavior_help)
                           
        
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
        
    def _print(self,thing):
        self.log = thing
        if self.verbosity: print(self._log[-1])
