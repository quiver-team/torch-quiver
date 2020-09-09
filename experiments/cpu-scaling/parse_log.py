import re

import numpy as np

ds = []
for line in open('out.log'):
    line = line.strip()
    if 'np=' in line:
        if ds:
            print('%.2f' % (np.mean(ds)))
            print('')
        ds = []
        print(line)
    if 'train one epoch took' in line:
        m = re.search(r'train one epoch took (.*)s', line)
        t = float(m.group(1))
        ds.append(t)

print('%.2f' % (np.mean(ds)))
