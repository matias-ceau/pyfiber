import pandas as pd
import numpy as np
import os
from scipy import stats
import extractor
import yaml

# CSV NOMENCLATURE
path = extractor.path()
with open(os.path.join( path['data'] , 'config.yaml'),'r') as y:
    nom = yaml.safe_load(y)
for n in nom.keys():
    if nom[n] == 'Time': time_nom = n
    if nom[n] == 'Calcium-dependant': sign_nom = n
    if nom[n] == 'Isosbestic': control_nom = n
possible_ttl = [k for k,v in nom['TTL']] #
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
        self.raw = call_raw(name,2000) #removes first 2000 lines
        self.ttl_channels = find_ttl_channel(self.raw) # list of names of the TTL channels
        self.Z = normZ(self.raw)
        self.F = normF(self.raw)
        self.L = normL(self.raw)
        self.time = get_arrays(self.raw)[0] #time of the whole recording
        self.cad = get_arrays(self.raw)[1] # calcium dependant signal
        self.iso = get_arrays(self.raw)[2] # isosbestic signal
    
    def __repr__(self):
        '''Gives general info on the recording object'''
        print('* Recording name: ',self.name)
        print(f'* Number of TTL: {len(self.ttl_channels)}\nTTL channel(s):', ','.join(self.ttl_channels))
        for i in self.ttl_channels: # list of TTL channels
            l = self.ttl_descriptor(i) # info about the specific channel 
            print('_'*20,'\n   ',i)
            print('Event number(s):',[j+1 for j in range(len(l['start lines'])) ])
            print('Start time(s):',str(l['start times']))
            print('End time(s):',str(l['end times']))
            print('Start line(s):',str(l['start lines']))
            print('End line(s):',(str(l['end lines'])))
        print(20*'_'+'\n')
        print(f'* Recording length: {recording_length(self.raw)} seconds')
        info = recording_info(self.name)
        print(f"* Import Date: {info['import date']}")
        print(f"* Experiment Date: {info['exp date']}")
        print(f"* Comment: {info['comment']}")

    def recording_info(self):
        df = extractor.History.recordings
        data = df[df['Recording'] == self.name]
        return {'import date':list(data['Import Date'])[0],
                'exp date':list(data['Date'])[0],
                'comment':list(data['Comment'])[0]}
    
    def ttl_descriptor(self,ttl_name):
        '''Finds start,end lines and times and duration of events for a particular TTL channel [0]/[2] = start/end lines & [1]/[3] are start/end times [4] is duration of event'''
        start, end = ttl_linefinder(self.raw,ttl_name)
        start_times = [self.time[i] for i in start] # list of all even starting times for channel
        end_times = [self.time[i] for i in end]
        return {'start lines' start,
                'start times':start_times,
                'end lines': end,
                'end times': end_times)
        
    def analyze_perievent(self,std='F',base=3,post=30,ttl_num=2,event_number=1):
        '''Output = (sample_time,robust_Zscores,TTLstart_time,Z_scores)
        Can also be used to analyse calcium-dependant and isosbestic (std=cad,std=iso)'''
        global possible_ttl
        if std=='Z': data = self.Z
        if std=='F': data = self.F
        if std=='L': data = self.L
        if std=='cad': data = self.cad
        if std=='iso': data = self.iso
        # retrieve time and line number of ttl of interest
        ttl_ch = possible_ttl[ttl_num-1]
        ttl_time = self.ttl_descriptor(ttl_ch)['start times'][event_number-1] #select ttl channel and event
        ttl_index = self.ttl_descriptor(ttl_ch)['start lines'][event_number-1] 
        # cut data into sample, with the help of the sampler function to get the indices
        idx = sampler(self.time,base,post,ttl_time)
        sample_time = self.time[idx[0]:idx[1]]
        sample_signal = data[idx[0]:idx[1]]
        # find the new ttl index (since the start has been sliced off)
        sample_ttl_index = ttl_index - idx[0]
        # get baseline
        baseline = sample_signal[:sample_ttl_index]
        #calculate the Z-scores
        Zscores = (sample_signal-baseline.mean())/baseline.std()
        # Robust Zscores ((signal - median(baseline))/MAD(baseline))
        RobustZscores = (sample_signal - np.median(baseline))/stats.median_abs_deviation(baseline)
        return (sample_time,RobustZscores,ttl_time,Zscores)

class TTL:
    def __init__(self,recording,channel):
        self.recording = recording
        self.channel = channel
        self.type = #

## FUNCTIONS USED BY OBJECTS

def find_ttl_channel(data):
    '''Returns the list of TTL channels for events that actually happen during the recording'''
    global possible_ttl
    ttl_channel = []
    for i in [ttl for ttl in data.columns if ttl in possible_ttl]:
        c = 0
        for j in data[i]:
            if j != 0: c+=1
        if c != 0: ttl_channel.append(i)
    return ttl_channel 

def call_raw(name,n):
    global path
    '''Retrieves raw data of selected recording and remove first n lines'''
    data = pd.read_csv(os.path.join(path['OUT'],name+'.csv'))[n:] #cuts of the first n lines (about 2 secondes where diodes initialize)
    data.reset_index(drop=True,inplace=True)
    return data

def get_arrays(data):
    '''Extract time, signal, control (for normalization) from raw data'''
    global time_nom,sign_nom,control_nom
    time = np.array(data[time_nom])
    signal = np.array(data[sign_nom])
    control = np.array(data[control_nom])
    return (time,signal,control)

def normZ(data):
    '''Normalizes data '''
    t,signal,control = get_arrays(data)
    S = (signal - signal.mean())/signal.std()
    I = (control - control.mean())/control.std()
    return S-I

def normF(data):
    '''Normalizes data'''
    time, signal,control = get_arrays(data)
    fit = np.polyfit(control,signal,1)
    fitted_control = control*fit[0] + fit[1]
    return (signal - fitted_control)/fitted_control

def normL(data):
    '''Normalizes data'''
    time, signal,control = get_arrays(data)
    fit = np.polyfit(control,signal,1)
    fitted_control = control*fit[0] + fit[1]
    return (signal - fitted_control)/fitted_control.std()

def ttl_linefinder(data,ttl_name):
    '''Finds the starting and ending lines of a particular ttl channel'''
    global possible_ttl
    time = get_arrays(data)[0]
    ttl = np.array(data[ttl_name])
    start = [i-1 for i in range(1,len(ttl)) if ttl[i-1]-ttl[i] < 0]
    end = [i-1 for i in range(1,len(ttl)) if ttl[i-1]-ttl[i] > 0]
    if ttl[0] == 1: start.insert(0,1)
    if ttl[-1] == 1: end.append(len(ttl)-1)
    return (start,end)

def recording_length(data):
    '''Calculates length of a recording (for class info)'''
    start = data[time_nom][0]
    end = data[time_nom][len(data)-1]
    return end-start

def recording_info(name):
    '''Asks INFO.csv about the selected recording (for Recording class objects)'''
    df = pd.read_csv('INFO.csv')
    df = df.loc[df['Recording'] == name]
    imp_date = df['Import Date'].iloc[0]
    exp_date = df['Experiment Date'].iloc[0]
    comment = df['Comment'].iloc[0]
    return (imp_date,exp_date,comment)

def sampler(time,base,post,ttl_time):
    '''Takes a sample of the recording, based on one TTL event and desired baseline and postevent duration'''
    start = ttl_time - base
    end = ttl_time + post
    start_idx = np.where(abs(time-start) == min(abs(time-start)))[0][0]
    end_idx = np.where(abs(time-end) == min(abs(time-end)))[0][0]
    return (start_idx,end_idx)
