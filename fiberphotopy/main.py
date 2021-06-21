import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import os
import PySimpleGUI as sg    
import datetime
import extractor
import fiber
import gui

######### VARIABLES ###############################
history = extractor.History() 
recordings = history.recordings
files = history.files

all_recordings = list(recordings['Recording'])
TTL_nice_list=[f'{v} ({k})' for k, v in fiber.channel_definition.items()]

class RecSel:
    def __init__(self):
        self.recs = {}
    def get(self,rec_name):
        if rec_name in self.recs.keys():
            return self.recs[rec_name]
        else:
            try:
                rec = fiber.Recording(rec_name)
                self.recs[rec_name] = rec
                return rec
            except: return None

#--------------------------------------------------------------------------------------------------------
def plotter():
    plt.plot([1, 2, 3, 4, 5, 6])
    plt.show()

#--------------------------------------------------------------------------------------------------------
sg.theme('Dark Green 5')
tab1_importer =  [[sg.Input(key='-filename-'), sg.FileBrowse()], 
                   [sg.Text('Session Name',s=(15,1)), sg.Input(key='-name-')], 
                   [sg.Text('Experiment Date',s=(15,1)), sg.Input(key='-exp_date-')], 
                   [sg.Text('Optional Comment',s=(15,1)), sg.Input(key='-comment-')], 
                   [sg.Text(key='-file_info-')], 
                   [sg.Button('Import')]] 

#--------------------------------------------------------------------------------------------------------
col_browser = [[sg.Text('Recording informations')],
               [sg.Listbox(values=all_recordings, size=(40, 6), key='-selected_recording-')],
               [sg.Button('Show Details', key='-select_recording-')]] 

col_details = [[sg.Multiline('',size=(40,10),key='-details-')]]

tab2_browser = [[sg.Column(col_browser),sg.Column(col_details)],
                [sg.Listbox(values=TTL_nice_list, s=(20, 4)),sg.Text('Event Number'),sg.Input('',s=(1,2))]
                   ]

#--------------------------------------------------------------------------------------------------------
col_norm = [[sg.Text('Normalisation')],
            [sg.Radio('Robust Z-scores', 'STD', default=True, enable_events=True, key='-sR-')], 
            [sg.Radio('Z-scores', 'STD', enable_events=True, key='-sZ-')]]

col_std = [[sg.Text('Standardization')],
           [sg.Radio('Standard delta F/F0', 'NORM', default=True, enable_events=True, key='-nF-')],
           [sg.Radio('Z-score Substraction', 'NORM', enable_events=True, key='-nZ-')],
           [sg.Radio('Alternative Linear', 'NORM', enable_events=True, key='-nL-')]]

col_options = [[sg.Text('Time selection')],
               [sg.Text('Baseline',size=(15,1)),sg.Input('5',key='-base-',size=(4,1))],
               [sg.Text('Post-event',size=(15,1)),sg.Input('10',key='-post-',size=(4,1))],
               [sg.Text('Graph limits',size=(15,1)),sg.Input('5',key='-limit1-',size=(2,1)),
                                                          sg.Input('5',key='-limit2-',size=(2,1))]]

tab3_analyser =  [[sg.Column(col_std),sg.Column(col_options),sg.Column(col_norm)],
                  [sg.Text('Recording',size=(15,1)),
                      sg.Input('Selected recording',k='-plotted_rec-',readonly=True)],
                  [sg.Text('Channel',size=(15,1)),
                      sg.Input('Selected TTL channel',k='-plotted_ttl-',readonly=True)],
                  [sg.Text('Event',size=(15,1)),
                      sg.Input('Selected event',k='-plotted_event-',readonly=True)],
                    [sg.Button('Plot')]]

#--------------------------------------------------------------------------------------------------------
layout = [[sg.TabGroup([[sg.Tab('Importer', tab1_importer), 
                         sg.Tab('Recording Browser', tab2_browser), 
                         sg.Tab('Analyser', tab3_analyser)]])
                         ]]    
window = sg.Window('Fiberphotopy', layout)    
#--------------------------------------------------------------------------------------------------------

while True:    
    event, values = window.read()
    print(values['-exp_date-'])
    if event == '-select_recording-':
        print(values['-selected_recording-'])
    if event == 'Plot': plotter()
    if event == sg.WIN_CLOSED:
        break 
