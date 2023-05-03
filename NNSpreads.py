#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NN predicting energy, slice energy spreads and bunch length.
"""

import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from typing import List, Optional, Union, Tuple

wd='C:\\Users\\johlun\\Documents\\Python Scripts'

if os.getcwd()!=wd:
    os.chdir(wd)

def scheduler(epoch, lr):
   if epoch < 5:
     return lr
   else:
     return lr * 0.1

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

##### DATA #####
#fhet=h5py.File('data/20kScanCSRONHMSlice.hdf','r')
f=h5py.File('data/spreadschirp.hdf','r')

scanfphs01=h5py.File('data\scanphs01csr.hdf','r')
scanfphs02=h5py.File('data\scanphs02csr.hdf','r')
scanfvolt01=h5py.File('data\scanvolt01csr.hdf','r')
scanfvolt02=h5py.File('data\scanvolt02csr.hdf','r')

bpmf=h5py.File('data\BPMs2500CSROn.hdf','r')

scanlphs01=scanfphs01['page1']['columns']['phs01']
scanlphs02=scanfphs02['page1']['columns']['phs02']
scanlvolt01=scanfvolt01['page1']['columns']['volt01']
scanlvolt02=scanfvolt02['page1']['columns']['volt02']
#ds
kesy=list(f.keys())
kesy2=[i[4:] for i in kesy]
kesy2.sort(key=int)
kesy=['page'+i for i in kesy2]
#X=[[] for i in range(len(scanlvolt02[:]))]

X=[[scanlphs01[i]/max(scanlphs01[:]),
    scanlphs02[i]/max(scanlphs02[:]),
    scanlvolt01[i]/max(scanlvolt01[:]),
    scanlvolt02[i]/max(scanlvolt02[:])] 
   for i in range(len(scanlvolt02[:]))]

for j in range(len(scanlvolt02[:])):
    for r in range(21):
        X[j].append(bpmf[kesy[j]]['Cx'][r]/max(abs(bpmf[kesy[j]]['Cx'][:])))
        X[j].append(bpmf[kesy[j]]['Cy'][r]/max(abs(bpmf[kesy[j]]['Cy'][:])))

X=np.asarray(X)

Y=[list(f[i][:3]) for i in kesy]#=np.asarray([list(f[i])+list(np.reshape(fhet[i]/np.amax(fhet[i]),10000)) for i in kesy])
chir=[f[i][3] for i in kesy]
normch=abs(np.amin(chir))
chir=chir/normch
for i in range(len(kesy)):
    Y[i].append(chir[i])
Y=np.asarray(Y)
X,Xtest,Y,Ytest=train_test_split(X,Y,test_size=0.10,shuffle=False)

f.close()
#fhet.close()
scanfphs01.close()
scanfphs02.close()
scanfvolt01.close()
scanfvolt02.close()
bpmf.close()

#### MODEL ####

model= tf.keras.models.Sequential()

def buildModel(model,modelname,inputs,labels,trainb=True): #For 200x200 images, multiply stuff by 4
    model.add(tf.keras.layers.Dense(100, activation='tanh'))
    model.add(tf.keras.layers.Dense(100, activation='tanh'))
    #model.add(tf.keras.layers.Dense(200, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(len(labels[0]), activation='tanh'))

    mcp_save = tf.keras.callbacks.ModelCheckpoint('Nets/mdl'+modelname+'_wts.hdf5', 
                                                  save_best_only=True, 
                                                  monitor='val_loss', 
                                                  mode='min')
    
    schedul= tf.keras.callbacks.LearningRateScheduler(scheduler)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss='mean_absolute_error')
    if trainb:
        history=model.fit(inputs,labels,batch_size=250,epochs=100, validation_split=0.01, callbacks=[mcp_save], verbose=0)
        
        plot_loss(history,modelname)
    
        model.load_weights('Nets/mdl'+modelname+'_wts.hdf5')
    else:
        history=model.fit(inputs,labels,batch_size=5000,epochs=1, validation_split=0.01)

        model.load_weights('Nets/mdl'+modelname+'_wts.hdf5')

buildModel(model, 'spread',X, Y,False)

#%%### EVAL ####
YpredEdge=model.predict(Xtest)


model.evaluate(Xtest,Ytest)

fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2)#,width_ratios=[5,5,5])

fig.tight_layout()

line=np.linspace(np.amin(Ytest[:,0]),np.amax(Ytest[:,0]),100)
ax1.plot(line,line,linestyle='--',color='black')
x1=ax1.scatter(Ytest[:,0],YpredEdge[:,0],marker='.',label='Slice E Spread')#,c=Xtest[:,2]) #'X Edge 1 Predictions'
ax1.legend(prop={'size':8})
ax1.set_xlabel('Truth [%]') #True X Edge 1 [ps]
ax1.set_ylabel('Prediction [%]') #'Predicted X Edge 1 [ps]'

line=np.linspace(np.amin(Ytest[:,1]),np.amax(Ytest[:,1]),100)
ax2.plot(line,line,linestyle='--',color='black')
ax2.scatter(Ytest[:,1],YpredEdge[:,1],marker='.',label='Full E spread')#,c=Xtest[:,2])
ax2.legend(prop={'size':8})
ax2.set_xlabel('Truth [%]')
ax2.set_ylabel('Prediction [%]')

line=np.linspace(np.amin(Ytest[:,2]),np.amax(Ytest[:,2]),100)
ax3.plot(line,line,linestyle='--',color='black')
ax3.scatter(Ytest[:,2],YpredEdge[:,2],marker='.',label='Bunch length')#,c=Xtest[:,2])
ax3.legend(prop={'size':8})
ax3.set_ylabel('Prediction [ps]')
ax3.set_xlabel('Truth [ps]')
#plt.colorbar(x1,ax=ax3,label='L01 Voltage Offset [%]')


line=np.linspace(np.amin(Ytest[:,3])*normch,np.amax(Ytest[:,3])*normch,100)
ax4.plot(line,line,linestyle='--',color='black')
ax4.scatter(Ytest[:,3]*normch,YpredEdge[:,3]*normch,marker='.',label='Chirp')#,c=Xtest[:,2])
ax4.legend()
ax4.set_xlabel('Truth [/ps]')
#ax4.xlabel('True Y Edge 2 [%]')
ax4.set_ylabel('Prediction [/ps]')

#fig.supxlabel('Prediction')
#fig.supylabel('Ground Truth')

#plt.savefig('Figs/IPACFOMsPredictionSim.png',dpi=200,bbox_inches='tight')

diffsE=Ytest[:,0]-YpredEdge[:,0]
diffE=Ytest[:,1]-YpredEdge[:,1]
diffB=Ytest[:,2]-YpredEdge[:,2]
diffC=Ytest[:,3]-YpredEdge[:,3]

RMSX=np.sum(diffsE**2)**(1/2)/len(diffsE)
RMSY=np.sum(diffE**2)**(1/2)/len(diffE)

RMSB=np.sum(diffB**2)**(1/2)/len(diffB)
RMSC=np.sum(diffC**2)**(1/2)/len(diffC)

print(f'RMS slice E: {RMSX:.5f}%, RMS full E: {RMSY:.5f}%, RMS Bunch: {RMSB*1000:.5f} fs, RMS Chirp: {RMSC:.5f} /ps')
