#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Most up to date image predictions on simulations. Most often running 100x100 images.
"""
#%%
import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from typing import List, Optional, Union, Tuple
import wandb
from skimage.metrics import structural_similarity as ssim

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


wandb.init(
    # set the wandb project where this run will be logged
    project="CNNSim",
    
    # track hyperparameters and run metadata
    config={
        "layer1": 200,
        "activation1": "relu",
        "layer2": 200,
        "activation2": "relu",
        "epoch": 750,
        'learning_rate': 1e-4
    }
)

config=wandb.config

def ssim_loss(y_true, y_pred):
    ssiml=np.array([ssim(y_true[i], y_pred[i],win_size=9) for i in tf.range(len(y_true))]) #window?
    return np.mean(ssiml)
##### DATA #####

f=h5py.File('data/20kScanCSRONHMSlice.hdf','r')

#sxscanf=h5py.File('data\sxscan.hdf','r')
scanfphs01=h5py.File('data\scanphs01csr.hdf','r')
scanfphs02=h5py.File('data\scanphs02csr.hdf','r')
scanfvolt01=h5py.File('data\scanvolt01csr.hdf','r')
scanfvolt02=h5py.File('data\scanvolt02csr.hdf','r')

bpmf=h5py.File('data\BPMsSXScan.hdf','r') #BPMs2500CSROn

#sxscanl1=sxscanf['page1']['columns']['k21']
#sxscanl2=sxscanf['page1']['columns']['k22']

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
'''
X=[[sxscanl1[i]/max(abs(sxscanl1[:])),
    sxscanl2[i]/max(sxscanl2[:])] 
   for i in range(len(sxscanl2[:]))]

'''
X=[[scanlphs01[i]/max(scanlphs01[:]),
    scanlphs02[i]/max(scanlphs02[:]),
    scanlvolt01[i]/max(scanlvolt01[:]),
    scanlvolt02[i]/max(scanlvolt02[:])] 
   for i in range(len(scanlvolt02[:]))]
'''
dispmask=[4,5,6,20]
for j in range(len(kesy)):
    tempb=np.array([bpmf[kesy[j]]['Cx'][r] for r in dispmask])
    #range(21)
    X[j]+=list(tempb/max(abs(tempb)))
        #X[j].append(bpmf[kesy[j]]['Cy'][r]/max(abs(bpmf[kesy[j]]['Cy'][:])))
'''
X=np.asarray(X)

Y=np.asarray([f[i]/np.amax(f[i]) for i in kesy])
'''
for k in range(len(Y)): #Set low intensity pixels to 0?
    for i in range(100):
        for j in range(100):
            if Y[k][i][j]<=0.01:
                Y[k][i][j]=0.0
  '''              
X,Xtest,Y,Ytest=train_test_split(X,Y,test_size=0.10,shuffle=False)

f.close()
#sxscanf.close()
scanfphs01.close()
scanfphs02.close()
scanfvolt01.close()
scanfvolt02.close()
bpmf.close()
#%%### MODEL ####

model= tf.keras.models.Sequential()

def buildModel(model,modelname,inputs,labels,trainb=True): #For 200x200 images, multiply stuff by 4
    model.add(tf.keras.layers.Dense(config.layer1, activation=config.activation1))
    model.add(tf.keras.layers.Dense(config.layer2, activation=config.activation2))
    #model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10000, activation='relu'))
    model.add(tf.keras.layers.Reshape((10,10,100)))
    model.add(tf.keras.layers.Conv2D(100, (3,3), padding='same',activation='relu'))
    model.add(tf.keras.layers.Reshape((25,25,16)))
    model.add(tf.keras.layers.Conv2D(16, (4,4), padding='same',activation='relu'))
    model.add(tf.keras.layers.Reshape((100,100,1)))
    model.add(tf.keras.layers.Conv2D(1, (5,5), padding='same',activation='relu'))

    mcp_save = tf.keras.callbacks.ModelCheckpoint('Nets/mdl'+modelname+'_wts.hdf5', 
                                                  save_best_only=True, 
                                                  monitor='val_loss', 
                                                  mode='min')
    
    schedul= tf.keras.callbacks.LearningRateScheduler(scheduler)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),loss='mean_absolute_error')
    if trainb:
        history=model.fit(inputs,labels,batch_size=250,epochs=config.epoch, validation_split=0.01, callbacks=[mcp_save], verbose=10)
        
        plot_loss(history,modelname)
    
        model.load_weights('Nets/mdl'+modelname+'_wts.hdf5')
    else:
        history=model.fit(inputs,labels,batch_size=5000,epochs=1, validation_split=0.01)

        model.load_weights('Nets/mdl'+modelname+'_wts_bestnoBPM.hdf5')

buildModel(model, 'heatCNNv2',X, Y,False)

#%%### EVAL ####

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
        fig.tight_layout()
        #fig.suptitle('TDC Results '+str(i),y=0.8)
        ax1.imshow(Ypim,vmin=0,vmax=max(np.matrix.flatten(Ytest[i])))
        ax1.set_title('Prediction')
        ax1.set_ylabel('$\Delta$E/E')
       # ax1.set_xticks([0,50,99])
       # ax1.set_xticklabels(Xedgep.round(2))
       # ax1.set_yticks([0,25,50,75,99])
       # ax1.set_yticklabels(Yedgep.round(2))
        ax2.imshow(Yim,vmin=0,vmax=max(np.matrix.flatten(Ytest[i])))
        ax2.set_title('Simulated')
        ax2.set_xlabel('t')
        #ax2.set_xticks([0,50,99])
        #ax2.set_xticklabels(Xedgep.round(2))
        ax3.imshow(diffim,vmin=0,vmax=max(np.matrix.flatten(Ytest[i])))
        ax3.set_title('Abs. Diff.')
        #ax3.set_xticks([0,50,99])
        #ax3.set_xticklabels(Xedgep.round(2))
        if len(save)>0:
            plt.savefig('Figs/'+str(save),dpi=200,bbox_inches='tight')
        
plotim(1)

Ytest2=[np.matrix.flatten(i) for i in Ytest]
Ypred2=[np.matrix.flatten(i) for i in Ypred]
diff2=[np.matrix.flatten(i) for i in diff]


rmsl=[(np.mean((i/np.amax(i))**2))**(1/2) for i in diff2]
print(f'Normalized RMS: {np.mean(rmsl):.3f}')

meantrue=[i-np.mean(i) for i in Ytest2]
Rl=[1-sum(diff2[i]**2)/sum(meantrue[i]**2) for i in range(len(diff2))]
print(f'SLAC Score [R²]: {np.mean(Rl):.3f}')

print(f'Mean SSIM: {ssim_loss(Ytest,Ypred):.3f}')

#rfile=h5py.File('Data/cleansave2206NoSet','r')

#rX=np.array([rfile[i]['X'][:] for i in kesy[:len(rfile.keys())]])
#rY=[rfile[i]['Y'][:] for i in kesy[:len(rfile.keys())]]
wandb.log({"R2": np.mean(Rl), 'SSIM': ssim_loss(Ytest,Ypred)})
wandb.finish()