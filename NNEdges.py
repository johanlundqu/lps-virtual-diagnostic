"""
Prediction of axis edges to reconstruct axes on sliced images.
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

f=h5py.File('data/20kScanCSRONSliceEdges.hdf','r')

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

X=[[scanlphs01[i]/max(scanlphs01[:]),
    scanlphs02[i]/max(scanlphs02[:]),
    scanlvolt01[i]/max(scanlvolt01[:]),
    scanlvolt02[i]/max(scanlvolt02[:])] 
   for i in range(len(scanlvolt02[:]))]
'''
for j in range(len(scanlvolt02[:])):
    for r in range(21):
        X[j].append(bpmf[kesy[j]]['Cx'][r]/max(abs(bpmf[kesy[j]]['Cx'][:])))
        X[j].append(bpmf[kesy[j]]['Cy'][r]/max(abs(bpmf[kesy[j]]['Cy'][:])))
'''
X=np.asarray(X)

Y=np.asarray([f[i] for i in kesy])

X,Xtest,Y,Ytest=train_test_split(X,Y,test_size=0.10,shuffle=False)

#### MODEL ####

model= tf.keras.models.Sequential()

def buildModel(model,modelname,inputs,labels,trainb=True): #For 200x200 images, multiply stuff by 4
    model.add(tf.keras.layers.Dense(200))#, activation='tanh'))
    model.add(tf.keras.layers.Dense(200))#, activation='tanh'))
    #model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(len(labels[0])))#, activation='tanh'))

    mcp_save = tf.keras.callbacks.ModelCheckpoint('Nets/mdl'+modelname+'_wts.hdf5', 
                                                  save_best_only=True, 
                                                  monitor='val_loss', 
                                                  mode='min')
    
    schedul= tf.keras.callbacks.LearningRateScheduler(scheduler)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss='mean_absolute_error')
    if trainb:
        history=model.fit(inputs,labels,batch_size=250,epochs=2000, validation_split=0.01, callbacks=[mcp_save], verbose=10)
        
        plot_loss(history,modelname)
    
        model.load_weights('Nets/mdl'+modelname+'_wts.hdf5')
    else:
        history=model.fit(inputs,labels,batch_size=5000,epochs=1, validation_split=0.01)

        model.load_weights('Nets/mdl'+modelname+'_wts_bestnoBPM.hdf5')

buildModel(model, 'Edges',X, Y,False)

#%%### EVAL ####
YpredEdge=model.predict(Xtest)


model.evaluate(Xtest,Ytest)

fig,(ax1,ax2,ax3,ax4)=plt.subplots(1,4)

line=np.linspace(np.amin(Ytest[:,0]),np.amax(Ytest[:,0]),100)
ax1.plot(line,line,linestyle='--',color='black')
x1=ax1.scatter(Ytest[:,0],YpredEdge[:,0],marker='.',label='X Edge 1 Predictions',c=Xtest[:,2]*0.5) #'X Edge 1 Predictions'
#ax1.legend()
#ax1.set_xlabel('True bunch length [ps]') #True X Edge 1 [ps]
ax1.set_ylabel('Prediction') #'Predicted X Edge 1 [ps]'

line=np.linspace(np.amin(Ytest[:,1]),np.amax(Ytest[:,1]),100)
ax2.plot(line,line,linestyle='--',color='black')
ax2.scatter(Ytest[:,1],YpredEdge[:,1],marker='.',label='X Edge 2 Predictions',c=Xtest[:,2])
#ax2.legend()
ax2.set_xlabel('Truth')
#ax2.set_ylabel('Predicted X Edge 2 [ps]')

line=np.linspace(np.amin(Ytest[:,2]),np.amax(Ytest[:,2]),100)
ax3.plot(line,line,linestyle='--',color='black')
ax3.scatter(Ytest[:,2],YpredEdge[:,2],marker='.',label='Y Edge 1 Predictions',c=Xtest[:,2])
#ax3.legend()
#ax3.set_xlabel('True Y Edge 1 [%]')
#ax3.set_ylabel('Predicted Y Edge 1 [%]')


line=np.linspace(np.amin(Ytest[:,3]),np.amax(Ytest[:,3]),100)
ax4.plot(line,line,linestyle='--',color='black')
ax4.scatter(Ytest[:,3],YpredEdge[:,3],marker='.',label='Y Edge 2 Predictions')
plt.colorbar(x1,ax=ax4,label='L01 Voltage Offset [%]')

#ax4.xlabel('True Y Edge 2 [%]')
#ax4.ylabel('Predicted Y Edge 2 [%]')

#fig.supxlabel('Prediction')
#fig.supylabel('Ground Truth')

#plt.savefig('Figs/BunchLengPredL01.png',dpi=200,bbox_inches='tight')

diffX=Ytest[:,0]-YpredEdge[:,0]
diffY=Ytest[:,2]-YpredEdge[:,2]
RMSX=np.sum(diffX**2)**(1/2)/len(diffX)
RMSY=np.sum(diffY**2)**(1/2)/len(diffY)

print(f'RMS X: {RMSX:.5f} fs, RMS Y: {RMSY:.5f}%')

#%% CLOSE
f.close()
scanfphs01.close()
scanfphs02.close()
scanfvolt01.close()
scanfvolt02.close()
bpmf.close()
