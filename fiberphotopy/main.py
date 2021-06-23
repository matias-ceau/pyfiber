import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import os
import PySimpleGUI as sg    
import datetime
import extractor
import fiber

######### VARIABLES ###############################
class Var:

    def __init__(self):
        self.TTL_nice_list=[f'{v} ({k})' for k, v in fiber.channel_definition.items()]
        # updates with the self.ipdate() function (based on files)
        self.files = extractor.History().files
        self.recordings = extractor.History().recordings
        self.rec_list = list(self.recordings['Recording'])
        # different update methods
        self.rec_data = {}
        for i in self.rec_list:
            self.obtain(i)
        self.analyses = {}
        self.descriptions = {}
        self.csel = {}
        self.pars = {}
    
    def describe(self,rec):
        if rec in self.descriptions.keys():
            return self.descriptions[rec]
        else:
            descr = str(self.rec_data[rec])
            self.descriptions[rec] = descr
            return descr

    def obtain(self,rec_name):
        if rec_name in self.rec_data.keys():
            return self.rec_data[rec_name]
        else:
            try:
                rec = fiber.Recording(rec_name)
                self.rec_data[rec_name] = rec
                return rec
            except: return None

    def analyze(self,name,base,pre,post,norm='F',ttl_num=2,event_number=1,z='robust'):
        saved = f'{name}_{norm}-{ttl_num}-{event_number}'
        if saved in self.analyses.keys():
            dw = self.analyses[saved]
        else: 
            dw =  self.rec_data[name].analyze_perievent(base,pre,post,norm,ttl_num,event_number)
            self.analyses[saved] = dw
        x = dw['sample time']
        if z == 'robust': y = dw['robust Z-scores']
        if z == 'standard': y = dw['Z-scores']
        vert = dw['event time']
        plt.plot(x,y)
        plt.axvline(vert,color='r')
        plt.title(saved)
        plt.show()

    def plotter(self,name):
        pass

    def to_csv(self,rec):
        pass 
    
    def update(self):
        self.recordings = extractor.History().recordings
        self.files = extractor.History().files
        self.rec_list = list(self.recordings['Recording'])
        for i in self.rec_list:
            self.obtain(i)
            self.describe(i)

var = Var()
#--------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------
sg.theme('Dark Green 5')
tab1_importer =  [[sg.Input(key='-filename-'), sg.FileBrowse()], 
                   [sg.Text('Session Name',s=(17,1)), sg.Input(key='-name-')], 
                   [sg.Text('Experiment Date',s=(17,1)), sg.Input(key='-exp_date-')], 
                   [sg.Text('Optional Comment',s=(17,1)), sg.Input(key='-comment-')], 
                   [sg.Text(key='-file_info-')], 
                   [sg.Button('Import')]
                  ] 

#--------------------------------------------------------------------------------------------------------
col_info = [[sg.Text('Recording Infos')],
            [sg.Listbox(values=var.rec_list, size=(40, 15), key='-recordings-',enable_events=True)],
            [sg.Multiline('',size=(40,15),key='-details-')]] 

col_selection = [[sg.Text('Selection for analysis')], 
                 [sg.Listbox(values=var.TTL_nice_list, s=(20, 4),key='-ttl_descr-')],
                 [sg.Text('Event Number')],
                 [sg.Spin([i for i in range(1,15)],initial_value=1,k='-event_num-',size=(4,4))],
                 [sg.Button('Select')],
                 [sg.Input('',size=(50,1),readonly=True,key='-message-')]
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

col_analysis = [[sg.Input('Selected recording',k='-plotted_rec-',readonly=True,s=(30,1))],
                [sg.Input('Selected TTL channel',k='-plotted_ttl-',readonly=True,s=(30,1))],
                [sg.Input('Selected event',k='-plotted_event-',readonly=True,s=(30,1))],
                [sg.Button('Plot'),sg.Button('Open Graph Editor',key='-pltplot-')]]

tab3_analyser =  [[sg.Column(col_std),sg.Column(col_options),sg.Column(col_norm),sg.Column(col_analysis)]]
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
        # import------------------------------------------------------
        if event == 'Import':
            try: extractor.NewFile(values['-filename-'],
                        values['-name-'],
                        values['-exp_date-'],values['-comment-'])
            except: pass
            var.update()
            window.Element('-recordings-').Update(values=var.rec_list)
        # click on recordings------------------------------------------
        if event == '-recordings-':
            #print info
            rc = values['-recordings-'][0]
            description = var.describe(rc)
            window.Element('-details-').Update(description)
        if event == 'Select':
            try:
                var.csel['recording'] = values['-recordings-'][0]
                var.csel['ttl'] = [int(i) for i in [str(n) for n in range(1,5)] if i in values['-ttl_descr-'][0]][0] 
                var.csel['event'] = int(values['-event_num-'])
                window['-message-'].update(str(var.csel))
                window['-plotted_rec-'].update(var.csel['recording'])
                window['-plotted_ttl-'].update('TTL: '+str(var.csel['ttl']))
                window['-plotted_event-'].update('Event: '+str(var.csel['event']))
            except: pass
            # to add (plot ttl)
        # select (for analysis)----------------------------------------
    #analyze(self,name,base,pre,post,norm='F',ttl_num=2,event_number=1,z='robust'):
        # open editor--------------------------------------------------
        if event == '-pltplot-': 
            try:
                var.pars['baseline'] = float(values['-base-'])
                var.pars['pre'] = float(values['-pre-'])
                var.pars['post'] = float(values['-post-'])
                if values['-sR-'] == True: var.pars['std'] = 'robust'
                if values['-sZ-'] == True: var.pars['std'] = 'standard'
                if values['-nF-'] == True: var.pars['norm'] = 'F'
                if values['-nZ-'] == True: var.pars['norm'] = 'Z'
                if values['-nL-'] == True: var.pars['norm'] = 'L'
                d = var.analyze(name=var.csel['recording'],
                            ttl_num=var.csel['ttl'],
                            event_number=var.csel['event'],
                            norm=var.pars['norm'],
                            z=var.pars['std'],
                            base=var.pars['baseline'],
                            pre=var.pars['pre'],
                            post=var.pars['post'])
            except: pass
        # quit -------------------------------------------------------
        if event == sg.WIN_CLOSED:
            break 

if __name__ == '__main__':
    main()
