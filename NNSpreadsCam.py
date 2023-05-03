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
f=h5py.File('data/cam07k02cleanallin.hdf','r')
fy=h5py.File('data/spreadschirpCam2.hdf','r')

kesy=list(f.keys())
kesy2=[i[4:] for i in kesy]
kesy2.sort(key=int)
kesy=['page'+i for i in kesy2]


X=np.asarray([f[i]['X'] for i in kesy])
Y=np.asarray([fy[i][:] for i in kesy])

                
X,Xtest,Y,Ytest=train_test_split(X,Y,test_size=0.25,shuffle=True)

f.close()

#### MODEL ####

model= tf.keras.models.Sequential()

def buildModel(model,modelname,inputs,labels,trainb=True): #For 200x200 images, multiply stuff by 4
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    #model.add(tf.keras.layers.Dense(200, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(len(labels[0]), activation='relu'))

    mcp_save = tf.keras.callbacks.ModelCheckpoint('Nets/mdl'+modelname+'_wts.hdf5', 
                                                  save_best_only=True, 
                                                  monitor='val_loss', 
                                                  mode='min')
    
    schedul= tf.keras.callbacks.LearningRateScheduler(scheduler)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss='mean_absolute_error')
    if trainb:
        history=model.fit(inputs,labels,batch_size=250,epochs=500, validation_split=0.05, callbacks=[mcp_save], verbose=0)
        
        plot_loss(history,modelname)
    
        model.load_weights('Nets/mdl'+modelname+'_wts.hdf5')
    else:
        history=model.fit(inputs,labels,batch_size=5000,epochs=1, validation_split=0.01)

        model.load_weights('Nets/mdl'+modelname+'_wts_best.hdf5')

buildModel(model, 'spreadcam',X, Y,True)

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


line=np.linspace(np.amin(Ytest[:,3]),np.amax(Ytest[:,3]),100)
ax4.plot(line,line,linestyle='--',color='black')
ax4.scatter(Ytest[:,3],YpredEdge[:,3],marker='.',label='Chirp')#,c=Xtest[:,2])
ax4.legend()
ax4.set_xlabel('Truth [/ps]')
#ax4.xlabel('True Y Edge 2 [%]')
ax4.set_ylabel('Prediction [/ps]')

#fig.supxlabel('Prediction')
#fig.supylabel('Ground Truth')

#plt.savefig('Figs/IPACFOMsPredictionReal.png',dpi=200,bbox_inches='tight')

diffsE=Ytest[:,0]-YpredEdge[:,0]
diffE=Ytest[:,1]-YpredEdge[:,1]
diffB=Ytest[:,2]-YpredEdge[:,2]
diffC=Ytest[:,3]-YpredEdge[:,3]

RMSX=np.sum(diffsE**2)**(1/2)/len(diffsE)
RMSY=np.sum(diffE**2)**(1/2)/len(diffE)

RMSB=np.sum(diffB**2)**(1/2)/len(diffB)
RMSC=np.sum(diffC**2)**(1/2)/len(diffC)

print(f'RMS slice E: {RMSX:.5f}%, RMS full E: {RMSY:.5f}%, RMS Bunch: {RMSB*1000:.5f} fs, RMS Chirp: {RMSC:.5f} /ps')
