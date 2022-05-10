import numpy as np
from scipy import signal
import yaml
import datetime
import info
import inspect
import time


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
        self._verbosity      = verbosity
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
        function_doc = [(k,v.__doc__) for k,v in self.__class__.__dict__.items() if (callable(v)) & (k[0] != '_')]
        # args     = [inspect.getfullargspec(i).args for n,i in self.__class__.__dict__.items() if callable(i) & (n[0]!='_')]
        # defaults = [inspect.getfullargspec(i).defaults for n,i in self.__class__.__dict__.items() if callable(i)]
        # number   = [len(i) for i in args]
        # deflen   = [len(i) if i else 0 for i in defaults]
        # diff     = [['']*(b-a) for a,b in zip(deflen,number)]
        # tuples   = [list(zip(k,a+list(b))) if b else list(zip(k,a)) for k,a,b in zip(args,diff,defaults)]
        # args     = [', '.join([f"{k}={v}" if v != '' else f"{k}" for k,v in a]) for a in tuples]
        for (a,b) in function_doc:
            print(f"<obj>.\033[1m{a}\033[0m()\n {b}\n") 
        if type(self).__name__  == 'BehavioralData':
            print(info.behavior_help)                  
    
    @property
    def _help(self):
        if self.type == 'BehavioralData':
            print(info.behavior_help)
        function_doc = [(k,v) for k,v in self.__class__.__dict__.items() if (callable(v))]
        for (a,b) in function_doc:
            print(f"{a:<20} --> {b.__doc__}")  
            print(f"{inspect.getfullargspec(b)}")
            
    @property
    def info(self):
         print('\n'.join(['<obj>.'+f'\033[1m{i}\033[0m' for i in sorted(self.__dict__) if i[0] != '_']))
    
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
        if self._verbosity: print(self._log[-1])

def timer(ref,name):
    now = time.time()
    delta = now - ref
    print(f'''\
============================================================================
    {name} took {delta} seconds
============================================================================''')
    return now