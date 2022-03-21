import behavioral_data
import sys
import os
import pandas as pd

def calc_interINJ1_2(path):
    folders = [f for f in os.listdir(path)]
    di = {}
    for f in folders:
        files = [i for i in os.listdir(f"{path}/{f}") if i[-3:] == 'dat']
        for g in files:
            if 'c20_' in g:
                rec_num = int(g.split('c20_')[-1].split('.')[0])
                obj = behavioral_data.BehavioralData(f"{path}/{f}/{g}")
                if  len(obj.inj1) != 0:
                    i1 = obj.inj1
                    i2 = obj._extract(6,1,'_L',2)
                    if len(i1) == len(i2):
                        diff = (i2-i1).mean()
                        date_str = g[3:11]
                        date = '-'.join([date_str[4:],date_str[2:4],date_str[:2]])
                        di[g] = [path[6:],
                                 date,
                                 f,
                                 rec_num,
                                 len(obj.inj1),
                                 diff]
                    else:
                        print(f"for file {g}, {len(i2)}, {len(i1)}")
    return pd.DataFrame(di,index=['exp','date','day','rec','injections','interpulse']).T

def debug_interpulse():
    folders = [f for f in os.listdir('.') if f[:6] == '_debug']
    return pd.concat([calc_interINJ1_2(i) for i in folders])


# ANIMATION
t = b999.xytime[::10]
x = b999.x[::10]
y = b999.y[::10]


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


# Setting up Data Set for Animation
dataSet = np.array([x, y])  # Combining our position coordinates
numDataPoints = len(t)

def animate_func(num):
    ax.clear()  # Clears the figure to update the line, point,   
    # Updating Trajectory Line (num+1 due to Python indexing)
    ax.plot(dataSet[0, :num+1], dataSet[1, :num+1], c='blue')
    # Updating Point Location 
    ax.scatter(dataSet[0, num], dataSet[1, num], c='blue', marker='o')
    # Adding Constant Origin
    #ax.scatter(dataSet[0, 0], dataSet[1, 0], c='black', marker='o')
    # Setting Axes Limits
#    ax.set_xlim([1, np.max(x)])
#    ax.set_ylim([1, np.max(y)])

#    # Adding Figure Labels
#    ax.set_title('Trajectory \nTime = ' + str(np.round(t[num], decimals=2)) + ' sec')
#    ax.set_xlabel('x')
#    ax.set_ylabel('y')
    
# Plotting the Animation
fig = plt.figure()
ax = plt.axes()
line_ani = animation.FuncAnimation(fig, animate_func, interval=100,  frames=numDataPoints)
line_ani
# Saving the Animation
f = '../../DATA/vid.gif'
writergif = animation.PillowWriter(fps=numDataPoints/6)
line_ani.save(f, writer=writergif)