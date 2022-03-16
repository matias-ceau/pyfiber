import importlib
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import pprint
import fp_utils
import behavioral_data
import fiber_data
import analysis
import info
importlib.reload(behavioral_data)
importlib.reload(fiber_data)
importlib.reload(analysis)
importlib.reload(fp_utils)


f04 = fiber_data.FiberData('poster_data/AS21RSA2Rat1204032022_0.csv')
b04 = behavioral_data.BehavioralData('poster_data/bsa04032022c20_01.dat')
p04 = analysis.RatSession(b04,f04)
res04 = p04.analyze_perievent((b04.switch_d_nd[0] + b04.rec_start[0]))

b04 = behavioral_data.BehavioralData('35/bsa09032022c20_02.dat')

 # def graph(self,
 #           obj,
 #           graph_title = None,
 #           figsize     = (30,0.5),
 #           color       = None,
 #           unit        = 'min',
 #           x_lim       = 'default'):
 #     ''''Show graphical representation of events and/or periods'''
 #     data = self._translate(obj)
 #     factor = {k:v*({'s':1,'ms':1000}[self.user_unit]) for k,v in {'ms' : 0.001, 's': 1, 'min': 60, 'h': 3.6*10**3}[unit]
 #     fig = plt.figure(figsize=figsize)
 #     if x_lim == 'default':
 #         x_lim = (0,self.end/factor)
 #     plt.xlim(x_lim)
 #     plt.ylim((0,1))
 #     if graph_title:
 #         plt.title(graph_title)
 #     try:
 #         if type(data[0]) in (np.int64,np.float64):
 #             plt.eventplot(data/factor,linelengths=1,lineoffsets=0.5,colors=color,linewidth=0.6)
 #         if type(data[0]) == tuple:
 #             for n,i in enumerate(data):
 #                 if color:
 #                     if type(color) == list:
 #                         color = color[n%len(color)]
 #                 else: color = 'grbcmk'[n%5]
 #                 plt.axvspan(i[0]/factor,i[1]/factor,color=color)
 #     except IndexError:
 #         plt.eventplot([])
