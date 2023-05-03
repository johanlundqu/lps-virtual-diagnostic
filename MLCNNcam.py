
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

kesy=list(f.keys())
kesy2=[i[4:] for i in kesy]
kesy2.sort(key=int)
kesy=['page'+i for i in kesy2]


X=np.asarray([f[i]['X'] for i in kesy])

Y=np.asarray([f[i]['Y'] for i in kesy])

                
X,Xtest,Y,Ytest=train_test_split(X,Y,test_size=0.05,shuffle=True)

f.close()

#%%### MODEL ####

model= tf.keras.models.Sequential()

def buildModel(model,modelname,inputs,labels,trainb=True): #For 200x200 images, multiply stuff by 4
    model.add(tf.keras.layers.Dense(200))#, activation='relu'))
    model.add(tf.keras.layers.Dense(200))#, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(100000))#, activation='relu'))
    model.add(tf.keras.layers.Reshape((10,10,1000)))
    model.add(tf.keras.layers.Conv2D(1000, (3,3), padding='same',activation='relu'))#,activation='relu'))
    model.add(tf.keras.layers.Reshape((25,25,160)))
    model.add(tf.keras.layers.Conv2D(160, (4,4), padding='same',activation='relu'))#,activation='relu'))
    model.add(tf.keras.layers.Reshape((100,100,10)))
    model.add(tf.keras.layers.Conv2D(10, (5,5), padding='same',activation='relu'))#,activation='relu'))
    model.add(tf.keras.layers.Reshape((200,500,1)))
    model.add(tf.keras.layers.Conv2D(1, (10,10), padding='same',activation='relu'))#,activation='relu'))


    mcp_save = tf.keras.callbacks.ModelCheckpoint('Nets/mdl'+modelname+'_wts.hdf5', 
                                                  save_best_only=True, 
                                                  monitor='val_loss', 
                                                  mode='min')
    
    schedul= tf.keras.callbacks.LearningRateScheduler(scheduler)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss='mean_absolute_error')
    if trainb:
        history=model.fit(inputs,labels,batch_size=25,epochs=100, validation_split=0.01, callbacks=[mcp_save], verbose=1)
        
        plot_loss(history,modelname)
    
        model.load_weights('Nets/mdl'+modelname+'_wts.hdf5')
    else:
        history=model.fit(inputs,labels,batch_size=5000,epochs=1, validation_split=0.01)

        model.load_weights('Nets/mdl'+modelname+'_wts_bestallin.hdf5')

buildModel(model, 'CNNcam07',X, Y,False)

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
        ax1.set_ylabel('$\Delta$E/E [%]')
        ax1.plot(500-np.sum(Ypim,axis=0),color='r')
        ax1.plot(np.sum(Ypim,axis=1)*3, range(len(Ypim)),color='r')
        ax1.set_yticks([0,100,200,300,400])
        ax1.set_yticklabels([0.304 , 0.182, 0.061,  -0.061,  -0.182])
        ax1.set_xticks([0,100,199])
        ax1.set_xticklabels([-0.381,  0.    ,  0.381])
        ax2.imshow(Yim,vmin=0,vmax=max(np.matrix.flatten(Ytest[i])))
        ax2.plot(500-(np.sum(Yim,axis=0)),color='r')
        ax2.plot(np.sum(Yim,axis=1)*3, range(len(Ypim)),color='r')
        ax2.set_title('Measured')
        ax2.set_xlabel('t [ps]')
        ax2.set_xticks([0,100,199])
        ax2.set_xticklabels([-0.381,  0.    ,  0.381])
        ax3.imshow(diffim,vmin=0,vmax=max(np.matrix.flatten(Ytest[i])))
        ax3.set_title('Abs. Diff.')
        ax3.set_xticks([0,100,199])
        ax3.set_xticklabels([-0.381,  0.    ,  0.381])
        if len(save)>0:
            plt.savefig('Figs/'+str(save),dpi=200,bbox_inches='tight')
        
#for i in range(len(Ytest)): #Real im span in mm = 3.125 mm, in Delta E % = 0.6085222555260256 %, x span = 1.25 mm = 0.7621951219512195 ps
plotim(3)

Ytest2=[np.matrix.flatten(i) for i in Ytest]
Ypred2=[np.matrix.flatten(i) for i in Ypred]
diff2=[np.matrix.flatten(i) for i in diff]


rmsl=[(np.mean((i/np.amax(i))**2))**(1/2) for i in diff2]
print(f'Normalized RMS: {np.mean(rmsl):.3f}')

meantrue=[i-np.mean(i) for i in Ytest2]
Rl=[1-sum(diff2[i]**2)/sum(meantrue[i]**2) for i in range(len(diff2))]
print(f'SLAC Score [R²]: {np.mean(Rl):.3f}')

#rfile=h5py.File('Data/cleansave2206NoSet','r')

#rX=np.array([rfile[i]['X'][:] for i in kesy[:len(rfile.keys())]])
#rY=[rfile[i]['Y'][:] for i in kesy[:len(rfile.keys())]]
