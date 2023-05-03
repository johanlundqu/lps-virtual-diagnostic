# -*- coding: utf-8 -*-
"""
Calibrating the TDC X-axis, i.e. X coordinate to time. Uses a phase scan of the TDC.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

f=h5py.File('C:\\Users\\johlun\\Downloads\\testall01(1).hdf','r')

kesy=list(f.keys())
kesy2=[i[4:] for i in kesy]
kesy2.sort(key=int)
kesy=['page'+i for i in kesy2[1:]]

xl=np.linspace(-9,9,1920)
lines=[np.sum(f[i][:],axis=1) for i in kesy]
temp=1/2.9985e9/360
temp2=np.linspace(-0.1,0.1,21)*temp

sus=[]

for j in range(21):
    sus.append(sum([xl[i]*lines[j][i]/sum(lines[j]) for i in range(len(xl))]))
    
sus=np.asarray(sus)

lin,a=np.polyfit(temp2*1e12,sus*1e-3,1)

linf=[i*1e12*lin for i in temp2]

plt.plot(temp2*1e12,sus*1e-3,marker='+')
plt.plot(temp2*1e12,linf,label='-0.0195 m/ps')
plt.xlabel('Time [ps]')
plt.ylabel('Mean X pos. [m]')
plt.legend()
#plt.savefig('figs/calibcurvetinyphase',dpi=200,bbox_inches='tight')