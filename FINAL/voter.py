
# coding: utf-8

# In[5]:

import pandas as pd
import numpy as np
from scipy import stats
import os
import re
import sys


# In[6]:

output_fname = sys.argv[1]


# In[7]:

fnames = [fname for fname in os.listdir('./') if fname.endswith('.txt') and fname.startswith('__')]


# In[8]:

all_ans = []
for fname in fnames:
    with open(fname, 'r') as f:
        all_ans.append(np.array([int(v) for v in f.readline().split(',')]))
all_ans = np.array(all_ans)
mode_ans = stats.mode(all_ans).mode[0]


# In[9]:

with open(output_fname, 'w') as f:
    f.write('id,ans\n')
    f.write('\n'.join(['%d,%d' % (i+1, a) for i, a in enumerate(mode_ans)]))
    f.write('\n')


# In[ ]:



