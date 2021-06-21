import numpy as np
import pandas as pd
import os
import datetime
import yaml

#--FUNCTIONS ------------------------------------- 

def path_f():
    data =  os.path.join(os.pardir,'data')
    history = os.path.join(data,'history')
    return {'data': data,
            'IN': os.path.join(data,'IN'),
            'OUT': os.path.join(data,'OUT'),
            'history' : history,
            'imported' : os.path.join(history,'imported_files.csv'),
            'recordings' : os.path.join(history,'recordings.csv'),
            'config': os.path.join(os.pardir,'config.yaml')}
            

def file_and_folder_creator():
    # create folder infrastructure if non existent
    if 'IN' not in os.listdir(path_f()['data']): os.mkdir(path_f()['IN'])
    if 'OUT' not in os.listdir(path_f()['data']): os.mkdir(path_f()['OUT'])
    if 'history' not in os.listdir(path_f()['data']): os.mkdir(path_f()['history'])
    # create files if non existent
    if 'imported_files.csv' not in os.listdir(path_f()['history']):
        pd.DataFrame({i:[] for i in ['File','Date']}).to_csv(path_f()['imported'],index = False)
    if 'recordings.csv' not in os.listdir(path_f()['history']):
        pd.DataFrame({i:[] for i in ['Recording', 'Import Date', 'Experiment Date', 'Comment','Columns']}).to_csv(path_f()['recordings'],index = False)

def file_splitter(path,name,time):
    '''
    This function inputs a session in raw csv containing multiple recordings and outputs a python dictionnary with the following structure {recording : data (panda DataFrame) }
    '''
    df = pd.read_csv(path)
    time = np.array(df[time])
    time_bis = np.array( [0] + list(time.copy())[:-1] )
    loc = np.where(time - time_bis > 1)
    recording_start = [0]+[i for i in loc[0]]
    return {f'{name[:-4]}_rec-{list(recording_start).index(k)+1}': #Rec name
            df.iloc[k:v,:]  # value 
            for k,v in zip(recording_start,
                           recording_start[1:]+[len(df)] )}

def check_new(filepath):
    '''Imputs a filepath and returns a boolean value (True if not present in imported_files)'''
    imported_df = pd.read_csv(path_f()['imported'])
    filename = os.path.split(filepath)[-1]
    return filename not in list(imported_df['File'])

#--------- CLASSES ---------------------
class History:
    def __init__(self):
        try:
            self.files = pd.read_csv(path_f()['imported'])
            self.recordings = pd.read_csv(path_f()['recordings'])
        except:
            self.files = pd.DataFrame({'File':[''],'Date':['']})
            self.recordings = pd.DataFrame({'Recording':[''],
                                            'Import Date':[''],
                                            'Experiment Date':[''],
                                            'Comment':['']})

    def update(self, newfile):
        self.files = self.files.append(pd.DataFrame({'File':[newfile.filename],
                                                     'Date':[newfile.importdate]}),ignore_index = True)
        for rec in newfile.recordings:
            self.recordings = self.recordings.append(pd.DataFrame({'Recording':[rec],
                                                                   'Import Date':[newfile.importdate],
                                                                   'Experiment Date':[newfile.exp_date],
                                                                   'Comment':[newfile.comment]}),ignore_index=True)
        self.files.to_csv(path_f()['imported'],index=False)
        self.recordings.to_csv(path_f()['recordings'],index=False)
    
class NewFile:
    def __init__(self,path,name,exp_date,comment=None):
        file_and_folder_creator()
        self.path = path
        self.filename = os.path.split(self.path)[-1]
        self.importdate = datetime.datetime.now()
        self.exp_date = exp_date
        self.session_name = name
        self.comment = comment
        self.recordings = [] #name of recordings
        self.data = pd.read_csv(path)
        self.columns = list(self.data.columns)
        self.new = check_new(path) #True or False
        with open(path_f()['config'],'r') as ymlfile:
            self.definition = yaml.safe_load(ymlfile)
        self._split_and_save()

    def __repr__(self):
        return 'Path: {}\nImport Date: {},\nExperiment Date: {}\nSession: {}\nRecordings:\n* {}\n'.format(self.path,self.importdate,self.exp_date,self.session_name,'\n* '.join(self.recordings))
    
    def _split_and_save(self):
        with open(path_f()['config'],'r') as ymlfile:
            nom = yaml.safe_load(ymlfile)
        for n in nom.keys(): 
            if nom[n] == 'Time': time = n
        if self.new:
            for k,v in file_splitter(self.path,self.filename,time).items():
                v.to_csv(os.path.join(path_f()['OUT'],k+'.csv'), index=False)
                self.recordings.append(k)
            History().update(self)

