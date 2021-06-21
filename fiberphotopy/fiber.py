import pandas as pd
import numpy as np
import os
from scipy import stats
import extractor
import yaml

# CSV NOMENCLATURE
path = extractor.path_f()
with open(path['config'],'r') as y:
    nom = yaml.safe_load(y)
for n in nom.keys():
    if nom[n] == 'Time': time_nom = n
    if nom[n] == 'Calcium-dependant': sign_nom = n
    if nom[n] == 'Isosbestic': control_nom = n
possible_ttl = list(nom['TTL'].keys()) 
channel_definition = nom['TTL']


class Recording:
    '''self.name
        self.raw     #removes first 2000 lines
        self.ttl_channels # list of names of the TTL channels
        self.time    #time of the whole recording
        self.cad     # calcium dependant signal
        self.iso     # isosbestic signal
        self.Z       # normalized channel (original deltaZ)
        self.F       # normalized channel ()
        self.L       # normalized channel (alt method)
        '''
    def __init__(self, name):
        self.name = name
        self.raw = _call_raw(name,2000) #removes first 2000 lines
        if len(self.raw)>0:
            self.ttl_channels = _find_ttl_channel(self.raw) # list of names of the TTL channels
            self.Z = _normZ(self.raw)
            self.F = _normF(self.raw)
            self.L = _normL(self.raw)
            self.time = _get_arrays(self.raw)[0] #time of the whole recording
            self.cad = _get_arrays(self.raw)[1] # calcium dependant signal
            self.iso = _get_arrays(self.raw)[2] # isosbestic signal
            self.ttl_info = {k:self.ttl_descriptor(k) for k in self.ttl_channels} #model {'DI/O-1' : {'start time':(1,2)}}
    
    def __repr__(self):
        '''Gives general info on the recording object'''
        if len(self.raw)>0:
            global nom
            rpr = '* Recording name:\n' + self.name+'\n* Number of TTL: '+str(len(self.ttl_channels))+'\n'+'TTL channel(s):'+','.join(self.ttl_channels)+'\n'
            for i in self.ttl_channels: # list of TTL channels
                l = self.ttl_descriptor(i) # info about the specific channel 
                rpr += '_'*20+'\n   '+i+': ' +nom['TTL'][i] +'\n'
                rpr += '\n'.join([f"({n}): {str(round(a))} -> {str(round(b))} ({_round_ut(a,b)})" for n,a,b in zip([str(j+1) for j in range(len(l['start lines']))], l['start times'], l['end times']) ]) +'\n'
            rpr += 20*'_'+'\n'
            rpr += f'* Recording length: {round(_recording_length(self.raw))} s'+'\n'
            info = self.recording_info()
            rpr += f"* Import Date: {info['import date'][:-16]}"+'\n'
            rpr += f"* Experiment Date: {info['exp date']}"+'\n'
            rpr += f"* Comment: {info['comment']}"+'\n'
            return rpr
        else: return f'This recording is not valid, checkout the csv (file: {self.name})'

    def recording_info(self):
        df = extractor.History().recordings
        data = df[df['Recording'] == self.name]
        return {'import date':list(data['Import Date'])[0],
                'exp date':list(data['Experiment Date'])[0],
                'comment':list(data['Comment'])[0]}
    
    def ttl_descriptor(self,ttl_name):
        if len(self.raw)>0:
            '''Finds start,end lines and times and duration of events for a particular TTL channel [0]/[2] = start/end lines & [1]/[3] are start/end times [4] is duration of event'''
            start, end = _ttl_linefinder(self.raw,ttl_name)
            start_times = [self.time[i] for i in start] # list of all even starting times for channel
            end_times = [self.time[i] for i in end]
            return {'start lines': start,
                    'start times':start_times,
                    'end lines': end,
                    'end times': end_times}
        
    def analyze_perievent(self,base,pre,post,std='F',ttl_num=2,event_number=1):
        '''Output = (sample_time,robust_Zscores,TTLstart_time,Z_scores)
        Can also be used to analyse calcium-dependant and isosbestic (std=cad,std=iso)'''
        global possible_ttl
        if len(self.raw)>0:
            if std=='Z': data = self.Z
            if std=='F': data = self.F
            if std=='L': data = self.L
            if std=='cad': data = self.cad
            if std=='iso': data = self.iso
            # retrieve time and line number of ttl of interest
            ttl_ch = possible_ttl[ttl_num-1]
            ttl_time = self.ttl_descriptor(ttl_ch)['start times'][event_number-1] #select ttl channel and event
            ttl_index = self.ttl_descriptor(ttl_ch)['start lines'][event_number-1] 
            # cut data into sample, with the help of the _sampler function to get the indices
            i = _sampler(self.time,base,post,ttl_time,pre)
            sample_time = self.time[i['pre']:i['post']]
            sample_signal = data[i['pre']:i['post']]
            sample_cad = self.cad[i['pre']:i['post']]
            sample_iso = self.iso[i['pre']:i['post']]
            # get baseline
            baseline = data[i['base'] : ttl_index]
            #calculate the z-scores
            zscores = (sample_signal-baseline.mean())/baseline.std()
            # robust zscores ((signal - median(baseline))/mad(baseline))
            robustzscores = (sample_signal - np.median(baseline))/stats.median_abs_deviation(baseline)
            return {'sample time':sample_time,
                    'robust Z-scores':robustzscores,
                    'event time':ttl_time,
                    'Z-scores':zscores,
                    'sample signal':sample_signal,
                    'sample cad':sample_cad,
                    'sample iso':sample_iso}


## FUNCTIONS USED BY OBJECTS

def _find_ttl_channel(data):
    '''Returns the list of TTL channels for events that actually happen during the recording'''
    global possible_ttl
    ttl_channel = []
    for i in [ttl for ttl in data.columns if ttl in possible_ttl]:
        c = 0
        for j in data[i]:
            if j != 0: c+=1
        if c != 0: ttl_channel.append(i)
    return ttl_channel 

def _call_raw(name,n):
    global path
    '''Retrieves raw data of selected recording and remove first n lines'''
    n = int(n)
    df = pd.read_csv(os.path.join(path['OUT'],name+'.csv'))
    if len(df) <= n:
        return [] 
    else: 
        data = df[n:] #cuts of the first n lines (about 2 secondes where diodes initialize)
        data.reset_index(drop=True,inplace=True)
        return data

def _get_arrays(data):
    '''Extract time, signal, control (for normalization) from raw data'''
    global time_nom,sign_nom,control_nom
    time = np.array(data[time_nom])
    signal = np.array(data[sign_nom])
    control = np.array(data[control_nom])
    return (time,signal,control)

def _normZ(data):
    '''Normalizes data '''
    t,signal,control = _get_arrays(data)
    S = (signal - signal.mean())/signal.std()
    I = (control - control.mean())/control.std()
    return S-I

def _normF(data):
    '''Normalizes data'''
    time, signal,control = _get_arrays(data)
    fit = np.polyfit(control,signal,1)
    fitted_control = control*fit[0] + fit[1]
    return (signal - fitted_control)/fitted_control

def _normL(data):
    '''Normalizes data'''
    time, signal,control = _get_arrays(data)
    fit = np.polyfit(control,signal,1)
    fitted_control = control*fit[0] + fit[1]
    return (signal - fitted_control)/fitted_control.std()

def _ttl_linefinder(data,ttl_name):
    '''Finds the starting and ending lines of a particular ttl channel'''
    global possible_ttl
    time = _get_arrays(data)[0]
    ttl = np.array(data[ttl_name])
    start = [i-1 for i in range(1,len(ttl)) if ttl[i-1]-ttl[i] < 0]
    end = [i-1 for i in range(1,len(ttl)) if ttl[i-1]-ttl[i] > 0]
    if ttl[0] == 1: start.insert(0,1)
    if ttl[-1] == 1: end.append(len(ttl)-1)
    return (start,end)

def _recording_length(data):
    '''Calculates length of a recording (for class info)'''
    start = data[time_nom][0]
    end = data[time_nom][len(data)-1]
    return end-start

def _sampler(time,base,post,ttl_time,pre):
    '''Takes a sample of the recording, based on one TTL event and desired baseline and postevent duration'''
    start = ttl_time - base
    end = ttl_time + post
    before = ttl_time - pre
    start_idx = np.where(abs(time-start) == min(abs(time-start)))[0][0]
    end_idx = np.where(abs(time-end) == min(abs(time-end)))[0][0]
    pre_idx = np.where(abs(time-before) == min(abs(time-before)))[0][0]
    return {'base':start_idx,
            'post':end_idx,
            'pre':pre_idx}

def _round_ut(a,b):
    diff = abs(a-b)
    if diff >= 1:
        return str(round(diff))+' s'
    else:
        return str(round(1000*diff))+' ms'

