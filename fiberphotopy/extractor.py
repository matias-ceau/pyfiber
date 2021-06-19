import numpy as np
import pandas as pd
import os
import datetime
import yaml

#--FUNCTIONS ------------------------------------- 

def path():
    return {'data': os.path.join(os.pardir,'data'),
            'IN': os.path.join(data_folder,'IN'),
            'OUT': os.path.join(data_folder,'OUT'),
            'history' : os.path.join(data_folder,'history'),
            'imported' : os.path.join(history,'imported_files.csv'),
            'recordings' : os.path.join(history,'recordings.csv')}
            

def file_and_folder_creator():
    # create folder infrastructure if non existent
    if 'IN' not in os.listdir(path()['data']): os.mkdir(path()['IN'])
    if 'OUT' not in os.listdir(path()['data']): os.mkdir(path()['OUT'])
    if 'history' not in os.listdir(path()['data']): os.mkdir(path()['history'])
    # create files if non existent
    if 'imported_files.csv' not in os.listdir(path()['history']):
        pd.DataFrame({i:[] for i in ['File','Date']).to_csv(path()['imported'],index = False)
    if 'recordings.csv' not in os.listdir(path()['history']):
        pd.DataFrame({i:[] for i in ['Recording', 'Import Date', 'Experiment Date', 'Comment','Columns']}).to_csv(infopath,index = False)

def file_splitter(path,name,time):
    '''
    This function inputs a session in raw csv containing multiple recordings and outputs a python dictionnary with the following structure {recording : data (panda DataFrame) }
    '''
    df = pd.read_csv(path)
    time = np.array(df[time])
    time_bis = np.array( [0] + list(time.copy())[:-1] )
    loc = np.where(time - time_bis > 1)
    recording_start = [0]+[i for i in loc[0]]
    return {f'{name[-4]}_rec-{list(recording_start).index(k)+1}': #Rec name
            df.iloc[k:v,:]  # value 
            for k,v in zip(recording_start,
                           recording_start[1:]+[len(df)] )}

def check_new(filepath):
'''Imputs a filepath and returns a boolean value (True if not present in imported_files)'''
    imported_df = pd.read_csv(path()['imported'])
    filename = os.path.split(filepath)[-1]
    return filename in list(imported_df)['File']

#--------- CLASSES ---------------------
class History:
''''''
    def __init__(self):
        self.files = pd.read_csv(path()['imported'])
        self.recordings = pd.read_csv(path()['recordings'])
    
    def update(self, newfile):
        self.files = self.files.append(pd.DataFrame({'File':newfile.filename,
                                                     'Date':newfile.importdate)}),ignore_index = True)
        for rec in newfile.recordings:
            self.recordings = self.recordings.append(pd.DataFrame({'Recording':rec,
                                                                   'Import Date':newfile.importdate,
                                                                   'Experiment Date':newfile.exp_date,
                                                                   'Comment':newfile.comment}),ignore_index=True)
        self.files.to_csv(path()['imported'],index=False)
        self.recordings.to_csv(path()['recordings'],index=False)
    
class NewFile:
''''''
    def __init__(self,path,name,exp_date,columns='std',comment=None):
        self.path = path
        self.filename = os.path.split(self.path)[-1]
        self.importdate = datetime.datetime.now()
        self.exp_date = exp_date
        self.session_name = name
        self.recordings = [] #name of recordings
        self.data = pd.read_csv(path)
        self.columns = list(self.data.columns)
        self.new = check_new(path) #True or False
        with open(os.path.join( path()['data'] , 'config.yaml'),'r') as ymlfile:
            self.definition = yaml.safe_load(ymlfile)

    def __repr__(self):
        return f"Path: {self.path}\nImport Date: {str(self)}\n\nSession: {self.session_name}\nRecordings:\n{'\n* '.join(self.recordings)}\n"
    
    def split_and_save(self,history):
        with open(os.path.join( path()['data'] , 'config.yaml' ),'r') as ymlfile:
            nom = yaml.safe_load(ymlfile)
        for n in nom.keys(): 
            if nom[n] == 'Time': time = n
        if self.new:
            for k,v in file_splitter(self.path,self.name,time):
                v.to_csv(os.path.join(path()['OUT'],k+'.csv'), index=False)
                self.recordings.append(k)

