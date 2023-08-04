# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:03:36 2023

@author: johlun
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

wd='C:\\Users\\johlun\\Documents\\Python Scripts'

if os.getcwd()!=wd:
    os.chdir(wd)

f=h5py.File('data/0619randfilltime.hdf', 'r')
backf=h5py.File('data/cam09backg.hdf','r')

cam07disp=0.5135391469451931 #m/%
cam07time=1.23 #1.64 #mm/ps

backim=backf['page1']['img'][:]
backf.close()

kesy=list(f.keys())
kesy2=[i[4:] for i in kesy]
kesy2.sort(key=int)
kesy=['page'+i for i in kesy2[:-1]]

imsave=[]
Xsave=[]

for j in kesy:
    bpml=np.array([float(k) for i in f[j]['bpm'][:,:] for k in i]) #4 to include only disp, -14 to include all bpms
    bpml=bpml/max(abs(bpml))
    ###CTs###
    bc1max=17862.196078431378 #17221.725490196077
    bc2max=16769.529411764706 #16015.176470588234
    bc1=np.sum(f[j]['bc1'][:][0])/bc1max
    bc2=np.sum(f[j]['bc2'][:][0])/bc2max
    im=f[j]['img'][:]-backim
    
    
    centx=np.argmax(np.sum(im,axis=1)) #[905:1235,430:1200]+905
    centy=np.argmax(np.sum(im[:,300:1000],axis=0))+300 #[905:1235,430:1200]+430
    
    Xsave.append([(f[j]['phs'][-2][0]-395.2)/6.0,(f[j]['phs'][-1][0]-177.7)/5]+list(bpml)+[bc1,bc2])
    sliceim=im[centx-250:centx+250,centy-50:centy+50]
    x0,y0=np.where(sliceim<0)
    for i in range(len(x0)):
        sliceim[x0[i],y0[i]]=0
    imsave.append(sliceim/np.amax(sliceim))
    #plt.figure()
    #plt.imshow(np.rot90(sliceim))
'''
with h5py.File('C:\\Users\\johlun\\Documents\\Python Scripts\\data\\randfilltimeclean.hdf','w') as hf:
    for i in range(len(kesy)):
        grp=hf.create_group('page'+str(i+1))
        grp.create_dataset('Y',data=imsave[i])
        grp.create_dataset('X',data=Xsave[i])
#hf.close()
'''
f.close()