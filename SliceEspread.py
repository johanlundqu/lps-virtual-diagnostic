# -*- coding: utf-8 -*-
"""
Extracting slice and full energy spread and bunch length and chirp from images.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

wd='C:\\Users\\johlun\\Documents\\Python Scripts'

if os.getcwd()!=wd:
    os.chdir(wd)
f=h5py.File('data/cam07k02cleanallin.hdf','r')#20kScanLPSim
fe=h5py.File('data/200SXScanEdges.hdf','r')#20kScanLPSedges


kesy=list(f.keys())
kesy2=[i[4:] for i in kesy]
kesy2.sort(key=int)
kesy=['page'+i for i in kesy2]

def Gauss(x, A, B,C):
    y = A/(B*np.sqrt(2*np.pi))*np.exp(-1*((x-C)/B)**2)
    return y

listl=[]
binl=[]
maxel=[]
order=2
sp='Chirp'

def spreader(sp, cam=False):
    if cam:
        if sp=='Slice':
            for i in kesy:
                testsli=np.rot90(f[i]['Y'][:])[:,75:125] #2 fs per bckt, 20 fs window [:,40:50] looks gud
                edge=np.linspace(-0.304,0.304,500)
                try:
                    parameters, covariance = curve_fit(Gauss, edge, np.sum(testsli,axis=1))
                except:
                    print(i)
                listl.append(parameters)
                binl.append([np.sum(testsli,axis=1),edge])
        if sp=='E':
            for i in kesy:
                testsli=np.rot90(f[i]['Y'][:])[:] #2 fs per bckt, 20 fs window [:,40:50] looks gud
                edge=np.linspace(-0.304,0.304,500)
                parameters, covariance = curve_fit(Gauss, edge, np.sum(testsli,axis=1))
                listl.append(parameters)
                binl.append([np.sum(testsli,axis=1),edge])
        if sp=='Bunch':
            for i in kesy:
                testsli=np.rot90(f[i]['Y'][:])[:] #2 fs per bckt, 20 fs window [:,40:50] looks gud
                edge=np.linspace(-0.381,0.381,200)
                parameters, covariance = curve_fit(Gauss, edge, np.sum(testsli,axis=0))
                listl.append(parameters)
                binl.append([np.sum(testsli,axis=0),edge])
        if sp=='Chirp':
            for i in kesy:
                im=np.rot90(f[i]['Y'][:])[:]
                #plt.imshow(im,extent=[fe[i][0],fe[i][1],fe[i][2],fe[i][3]],aspect='auto')
                maxr=np.array([np.argmax(im[:,j]) for j in range(len(im[0]))])
                edgex=np.linspace(-0.381,0.381,200)
                edgey=np.linspace(-0.304,0.304,500)
                maxe=np.array([edgey[i] for i in maxr])
                if order==2:
                    fit=np.polyfit(edgex[60:150],maxe[60:150],2,full=True)
                else:
                    fit=np.polyfit(edgex[50:150],maxe[50:150],1,full=True)
                fit=fit[:2]
                listl.append(fit[0])
                binl.append(fit[1][0])
                maxel.append(maxe[60:150])
                #plt.figure()
                #plt.plot(edgex[40:60],maxe[40:60],color='r')
                #plt.plot(edgex[40:60],fit[0][0]*edgex[40:60]+fit[0][1],color='r')
                
    else:
        if sp=='Slice':
            for i in kesy:
                testsli=np.rot90(f[i])[:,40:50] #2 fs per bckt, 20 fs window [:,40:50] looks gud
                edge=np.linspace(fe[i][3],fe[i][2],100)
                try:
                    parameters, covariance = curve_fit(Gauss, edge, np.sum(testsli,axis=1))
                except:
                    print(i)
                listl.append(parameters)
                binl.append([np.sum(testsli,axis=1),edge])
        if sp=='E':
            for i in kesy:
                testsli=np.rot90(f[i])[:] #2 fs per bckt, 20 fs window [:,40:50] looks gud
                edge=np.linspace(fe[i][3],fe[i][2],100)
                parameters, covariance = curve_fit(Gauss, edge, np.sum(testsli,axis=1))
                listl.append(parameters)
                binl.append([np.sum(testsli,axis=1),edge])
        if sp=='Bunch':
            for i in kesy:
                testsli=np.rot90(f[i])[40:55,:] #2 fs per bckt, 20 fs window [:,40:50] looks gud
                edge=np.linspace(fe[i][0],fe[i][1],100)
                parameters, covariance = curve_fit(Gauss, edge, np.sum(testsli,axis=0))
                listl.append(parameters)
                binl.append([np.sum(testsli,axis=0),edge])
        if sp=='Chirp':
            for i in kesy:
                im=np.rot90(f[i])[:]
                #plt.imshow(im,extent=[fe[i][0],fe[i][1],fe[i][2],fe[i][3]],aspect='auto')
                maxr=np.array([np.argmax(im[:,j]) for j in range(len(im))])
                edgex=np.linspace(fe[i][0],fe[i][1],100)
                edgey=np.linspace(fe[i][3],fe[i][2],100)
                maxe=np.array([edgey[i] for i in maxr])
                if order==2:
                    fit=np.polyfit(edgex[40:60],maxe[40:60],2,full=True)
                else:
                    fit=np.polyfit(edgex[40:60],maxe[40:60],1,full=True)
                fit=fit[:2]
                listl.append(fit[0])
                binl.append(fit[1][0])
                maxel.append(maxe[40:60])
                #plt.figure()
                #plt.plot(edgex[40:60],maxe[40:60],color='r')
                #plt.plot(edgex[40:60],fit[0][0]*edgex[40:60]+fit[0][1],color='r')
                
spreader(sp,True)
listl=np.asarray(listl)

def plotfit(indx,cam=False):
    if sp=='Chirp':
        i=kesy[indx]
        if cam:
            im=np.rot90(f[i]['Y'])[:]
            edgex=np.linspace(-0.381,0.381,200)
            edgex=edgex[60:150]
            plt.imshow(im,extent=[-0.381,0.381,-0.304,0.304],aspect='equal')
            plt.plot(edgex,-maxel[indx],color='r',marker='+')
            if order==2:
                plt.plot(edgex,-listl[indx][0]*edgex**2-listl[indx][1]*edgex-listl[indx][2],label='Chirp='+str(round(listl[indx][0],3))+' ps$^{-2}$ + '+str(round(listl[indx][1],3))+'ps$^{-1}$',color='y',linestyle='--')
            else:
                plt.plot(edgex,-listl[indx][0]*edgex-listl[indx][1],label='Chirp='+str(round(listl[indx][0],3))+'ps$^{-1}$',color='y',linestyle='--')
            plt.xlabel('t [ps]')
            plt.ylabel('$\Delta$E/E [%]')
            #plt.title('TDC')
            plt.legend()
            #plt.savefig('figs/TDCexamplechirp.png',dpi=200,bbox_inches='tight')
        else:
            im=np.rot90(f[i])[:]
            edgex=np.linspace(fe[i][0],fe[i][1],100)
            edgex=edgex[40:60]
            plt.imshow(im,extent=[fe[i][0],fe[i][1],fe[i][2],fe[i][3]],aspect='auto')
            plt.plot(edgex,maxel[indx],color='r',marker='+')
            if order==2:
                plt.plot(edgex,listl[indx][0]*edgex**2+listl[indx][1]*edgex+listl[indx][2],label='Chirp='+str(round(listl[indx][0],3))+' ps$^{-2}$ + '+str(round(listl[indx][1],3))+'ps$^{-1}$',color='y',linestyle='--')
            else:
                plt.plot(edgex,listl[indx][0]*edgex+listl[indx][1],label='Chirp='+str(round(listl[indx][0],3))+'ps$^{-1}$',color='y',linestyle='--')
            plt.xlabel('t [ps]')
            plt.ylabel('$\Delta$E/E [%]')
            #plt.title('TDC')
            plt.legend()
            #plt.savefig('figs/TDCexamplechirp.png',dpi=200,bbox_inches='tight')
    else:
        fit_y = Gauss(binl[indx][1], listl[indx,0], listl[indx,1],listl[indx,2])
        plt.plot(binl[indx][1], binl[indx][0], '+', label='data')
        plt.plot(binl[indx][1], fit_y, '-', label='fit $\sigma$='+str(round(listl[indx,1],3)))
        if sp=='Bunch':
            plt.xlabel('t [ps]')
        else:
            plt.xlabel('$\Delta$E/E [%]')
        plt.legend()
plotfit(1,True)
#%%
hl=[[] for i in kesy]
call=['Slice','E','Bunch','Chirp'] #Saving done in this order
for i in range(4):
    listl=[]
    binl=[]
    maxel=[]
    spreader(call[i],True)
    if call[i]=='Chirp':
        if order==2:
            for j in range(len(kesy)):
                hl[j].append((listl[j][1]))
        else:
            for j in range(len(kesy)):
                hl[j].append((listl[j][0]))
    else:
        for j in range(len(kesy)):
            hl[j].append(abs(listl[j][1]))
'''
hf=h5py.File('data/spreadschirpCam2.hdf','w')
hl=np.asarray(hl)
for i in range(len(hl)):
    hf.create_dataset('page'+str(i+1),data=hl[i])
    
hf.close()
'''
#%%
f.close()
fe.close()