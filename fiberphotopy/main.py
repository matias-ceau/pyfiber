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
        global all_recordings
        self.rec_list = all_recordings
        self.rec_data = {}
        for i in self.rec_list:
            self.obtain(i)           

    def obtain(self,rec_name):
        if rec_name in self.rec_data.keys():
            return self.rec_data[rec_name]
        else:
            try:
                rec = fiber.Recording(rec_name)
                self.rec_data[rec_name] = rec
                return rec
            except: return None

recsel = RecSel()
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
                   [sg.Button('Import')]
                  ] 

#--------------------------------------------------------------------------------------------------------
col_info = [[sg.Text('Recording Infos')],
            [sg.Listbox(values=all_recordings, size=(40, 15), key='-selected_recording-',enable_events=True)],
            [sg.Multiline('',size=(40,15),key='-details-')]] 

col_selection = [[sg.Image('foo.png')],
                 [sg.Text('Selection for analysis')], 
                 [sg.Listbox(values=TTL_nice_list, s=(20, 4)),
                     sg.Column([[sg.Text('Event Number')],
                                [sg.Spin([i for i in range(1,15)],initial_value=1,k='-event_num-',size=(4,4))]]),
                     sg.Column([[sg.Button('Select')],
                                [sg.Input('Ready',size=(10,1),readonly=True,key='-message-')]])]
                 ]

tab2_browser = [[sg.Column(col_info),sg.Column(col_selection)]]

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
               [sg.Text('Pre-event (graph)',size=(15,1)),sg.Input('5',key='-pre-',size=(4,1))]]

col_analysis = [[sg.Input('Selected recording',k='-plotted_rec-',readonly=True,s=(10,1))],
                [sg.Input('Selected TTL channel',k='-plotted_ttl-',readonly=True,s=(10,1))],
                [sg.Input('Selected event',k='-plotted_event-',readonly=True,s=(10,1))],
                [sg.Button('Plot'),sg.Button('Open Graph Editor',key='-pltplot-')]]

tab3_analyser =  [[sg.Column(col_std),sg.Column(col_options),sg.Column(col_norm),sg.Column(col_analysis)],
                   [sg.Image('foo.png')]]

#--------------------------------------------------------------------------------------------------------
layout = [[sg.TabGroup([[sg.Tab('Importer', tab1_importer), 
                         sg.Tab('Recording Browser', tab2_browser), 
                         sg.Tab('Analyser', tab3_analyser)]])
                         ]]    
window = sg.Window('Fiberphotopy', layout)    
#--------------------------------------------------------------------------------------------------------

def main():
    while True:    
        event, values = window.read()
        # import

        # click on recordings

        # select (for analysis)

        # plot

        # open editor
        print(values['-exp_date-'])
        if event == '-select_recording-':
            print(values['-selected_recording-'])
        if event == '-pltplot-': plotter()
        if event == sg.WIN_CLOSED:
            break 

if __name__ == '__main__':
    main()
