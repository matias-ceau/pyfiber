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