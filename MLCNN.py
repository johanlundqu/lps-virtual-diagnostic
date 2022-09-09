#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 14:25:32 2022

@author: johlun
"""

import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

wd='C:\\Users\\johlun\\Documents\\Python Scripts'

if os.getcwd()!=wd:
    os.chdir(wd)


def plot_loss(history,title):
  plt.figure()  
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  #plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.title(title)
  plt.legend()
  plt.grid(True)

f=h5py.File('data/1000heatmapsdipON.hdf','r')
scanf=h5py.File('data\scanbothrand.hdf','r')
bpmf=h5py.File('data\BPMs100021dip.hdf','r')

scanl=scanf['page1']['columns']

kesy=list(f.keys())
kesy2=[i[4:] for i in kesy]
kesy2.sort(key=int)
kesy=['page'+i for i in kesy2]

X=[[scanl['phs01'][i]/max(scanl['phs01'][:]),
    scanl['phs02'][i]/max(scanl['phs02'][:]),
    scanl['volt01'][i]/max(scanl['volt01'][:]),
    scanl['volt02'][i]/max(scanl['volt02'][:])] 
   for i in range(len(scanl['volt02'][:]))]

for j in range(len(scanl['volt02'][:])):
    for r in range(21):
        X[j].append(bpmf[kesy[j]]['Cx'][r]/max(abs(bpmf[kesy[j]]['Cx'][:])))
        X[j].append(bpmf[kesy[j]]['Cy'][r]/max(abs(bpmf[kesy[j]]['Cy'][:])))

X=np.asarray(X)

Y=np.asarray([f[i]/np.amax(f[i]) for i in kesy])

X,Xtest,Y,Ytest=train_test_split(X,Y,test_size=0.10,random_state=11)

model= tf.keras.models.Sequential()

def buildModel(model,modelname,inputs,labels,trainb=True):
    model.add(tf.keras.layers.Dense(200, activation='linear'))
    model.add(tf.keras.layers.Dense(200, activation='linear'))
    model.add(tf.keras.layers.Dense(2500, activation='linear'))
    model.add(tf.keras.layers.Reshape((5,5,100)))
    model.add(tf.keras.layers.Conv2D(100, (2,2), padding='same'))
    model.add(tf.keras.layers.Reshape((25,25,4)))
    model.add(tf.keras.layers.Conv2D(4, (4,4), padding='same'))
    model.add(tf.keras.layers.Reshape((50,50,1)))
    model.add(tf.keras.layers.Conv2D(1, (5,5), padding='same'))

    mcp_save = tf.keras.callbacks.ModelCheckpoint('Nets/mdl'+modelname+'_wts.hdf5', 
                                                  save_best_only=True, 
                                                  monitor='val_loss', 
                                                  mode='min')
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss='mean_absolute_error')
    if trainb:
        history=model.fit(inputs,labels,batch_size=50,epochs=2000, validation_split=0.01, callbacks=[mcp_save], verbose=10)
        
        plot_loss(history,modelname)
    
        model.load_weights('Nets/mdl'+modelname+'_wts.hdf5')
    else:
        history=model.fit(inputs,labels,batch_size=5000,epochs=1, validation_split=0.01)

        model.load_weights('Nets/mdl'+modelname+'_wts_bestdip.hdf5')

buildModel(model, 'heatCNN',X, Y,False)

model.evaluate(Xtest,Ytest)

Ypred=np.reshape(model.predict(Xtest),np.shape(Ytest))

diff=abs(Ytest-Ypred)
fftYt=np.fft.fft(Ytest)
fftYp=np.fft.fft(Ypred)
ffdiff=np.fft.fft(Ytest)-np.fft.fft(Ypred) #Investigate further... Maybe individual ffts of vertical and hor slices. Also attempt reconstruction of real space with smaller part of the fourier image.

def plotim(i,fft=False,save=''):
    if fft:
        fftYpim=np.rot90(fftYp[i])
        fftYtim=np.rot90(fftYt[i])
        ffdiffim=np.rot90(ffdiff[i])
        fig,[ax1,ax2,ax3]=plt.subplots(1,3,sharey=True)
        fig.suptitle('TDC FFT',y=0.8)
        ax1.imshow(np.real(fftYpim),vmin=0,vmax=max(np.matrix.flatten(np.real(fftYt[i]))))
        ax1.set_title('Prediction')
        ax1.set_ylabel('Y []')
        #ax1.set_xticks([0,12.5,25,37.5,49])
        #ax1.set_xticklabels([-2,-1,0,1,2])
        #ax1.set_yticks([0,12.5,25,37.5,49])
        #ax1.set_yticklabels([-0.1,-0.05,0,0.05,0.1])
        ax2.imshow(np.real(fftYtim),vmin=0,vmax=max(np.matrix.flatten(np.real(fftYt[i]))))#,vmin=0,vmax=max(np.matrix.flatten(Ytest[i])))
        ax2.set_title('True')
        ax2.set_xlabel('X []')
        ax3.imshow(np.real(ffdiffim),vmin=0,vmax=max(np.matrix.flatten(np.real(fftYt[i]))))#,vmin=0,vmax=max(np.matrix.flatten(Ytest[i])))
        ax3.set_title('Abs. Diff.')
        #plt.savefig('BC2SCRN01Cent2',dpi=200)
    else:
        Ypim=np.rot90(Ypred[i])
        Yim=np.rot90(Ytest[i])
        diffim=np.rot90(diff[i])
        
        fig,[ax1,ax2,ax3]=plt.subplots(1,3,sharey=True)
        fig.suptitle('TDC Results '+str(i),y=0.8)
        ax1.imshow(Ypim,vmin=0,vmax=max(np.matrix.flatten(Ytest[i])))
        ax1.set_title('Prediction')
        ax1.set_ylabel('Y [mm]')
        ax1.set_xticks([0,12.5,25,37.5,49])
        ax1.set_xticklabels([-2,-1,0,1,2])
        ax1.set_yticks([0,12.5,25,37.5,49])
        ax1.set_yticklabels([-0.1,-0.05,0,0.05,0.1])
        ax2.imshow(Yim,vmin=0,vmax=max(np.matrix.flatten(Ytest[i])))
        ax2.set_title('Simulated')
        ax2.set_xlabel('X [mm]')
        ax2.set_xticks([0,12.5,25,37.5,49])
        ax2.set_xticklabels([-2,-1,0,1,2])
        ax3.imshow(diffim,vmin=0,vmax=max(np.matrix.flatten(Ytest[i])))
        ax3.set_title('Abs. Diff.')
        ax3.set_xticks([0,12.5,25,37.5,49])
        ax3.set_xticklabels([-2,-1,0,1,2])
        if len(save)>0:
            plt.savefig('Figs/'+str(save),dpi=200,bbox_inches='tight')
        
for i in range(0,100,20):
    plotim(i)
#    plotim(i,True)

Ytest2=[np.matrix.flatten(i) for i in Ytest]
Ypred2=[np.matrix.flatten(i) for i in Ypred]
diff2=[np.matrix.flatten(i) for i in diff]


rmsl=[(np.mean((i/np.amax(i))**2))**(1/2) for i in diff2]
print('Normalized RMS: '+str(np.mean(rmsl)))

meantrue=[i-np.mean(i) for i in Ytest2]
Rl=[1-sum(diff2[i]**2)/sum(meantrue[i]**2) for i in range(len(diff2))]
print('SLAC Score [R²]: '+str(np.mean(Rl)))

#rfile=h5py.File('Data/cleansave2206NoSet','r')

#rX=np.array([rfile[i]['X'][:] for i in kesy[:len(rfile.keys())]])
#rY=[rfile[i]['Y'][:] for i in kesy[:len(rfile.keys())]]

f.close()
scanf.close()