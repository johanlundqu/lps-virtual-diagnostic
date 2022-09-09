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


f=h5py.File('Data/cleansave0728ALLimgs','r')

kesy=list(f.keys())
kesy2=[i[4:] for i in kesy]
kesy2.sort(key=int)
kesy=['page'+i for i in kesy2]

rX=np.array([np.append(f[k]['X'][:3],f[k]['X'][3:]/np.amax(abs(f[k]['X'][3:]))) for k in kesy]) #7,9 and 23 bad, perhaps it don't matter

rY=np.asarray([f[i]['Y'][:] for i in kesy])

Y=[rY[k]/np.amax(rY[k]) for k in range(len(rY))]
X,Xtest,Y,Ytest=train_test_split(rX,rY,test_size=0.10,random_state=20)

#X = np.asarray(X).astype('float32')
#Y = np.asarray(Y).astype('float32')

model= tf.keras.models.Sequential()

def buildModel(model,modelname,inputs,labels,trainb):
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    model.add(tf.keras.layers.Dense(10000, activation='relu'))
    #model.add(tf.keras.layers.Reshape((5,5,100)))
    #model.add(tf.keras.layers.Conv2D(100, (2,2), padding='same'))
    model.add(tf.keras.layers.Reshape((25,25,16)))
    model.add(tf.keras.layers.Conv2D(16, (4,4), padding='same'))
    model.add(tf.keras.layers.Reshape((200,50,1)))
    model.add(tf.keras.layers.Conv2D(1, (5,5), padding='same'))
    
    mcp_save = tf.keras.callbacks.ModelCheckpoint('Nets/mdl'+modelname+'_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss='mean_absolute_error')

    if trainb:
        history=model.fit(inputs,labels,batch_size=10,epochs=300, validation_split=0.01, callbacks=[mcp_save])
          
        plot_loss(history,modelname)
      
        model.load_weights('Nets/mdl'+modelname+'_wts.hdf5')
    else:
        history=model.fit(inputs,labels,batch_size=5000,epochs=1, validation_split=0.01)

        model.load_weights('Nets/mdl'+modelname+'_wts_bestAll.hdf5')
buildModel(model, 'imgreal',X, Y,False)


model.evaluate(Xtest,Ytest)

Ypred=np.reshape(model.predict(Xtest),np.shape(Ytest))

diff=abs(Ytest-Ypred)
fftYt=np.fft.fft(Ytest)
fftYp=np.fft.fft(Ypred)
ffdiff=np.fft.fft(Ytest)-np.fft.fft(Ypred) #Investigate further... Maybe individual ffts of vertical and hor slices. Also attempt reconstruction of real space with smaller part of the fourier image.

def plotim(i,fft=False):
    if fft:
        fftYpim=np.rot90(fftYp[i])
        fftYtim=np.rot90(fftYt[i])
        ffdiffim=np.rot90(ffdiff[i])
        fig,[ax1,ax2,ax3]=plt.subplots(1,3,sharey=True)
        fig.suptitle('BC2 SCRN-01 FFT',y=0.8)
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
        #fig.suptitle('BC2 SCRN-01',y=0.8)
        ax1.imshow(Ypim,vmin=0,vmax=max(np.matrix.flatten(Ytest[i])))
        ax1.set_title('Prediction')
        ax1.set_ylabel('Y [mm]')
        ax2.imshow(Yim,vmin=0,vmax=max(np.matrix.flatten(Ytest[i])))
        ax2.set_title('True')
        ax2.set_xlabel('X [mm]')
        ax3.imshow(diffim,vmin=0,vmax=max(np.matrix.flatten(Ytest[i])))
        ax3.set_title('Abs. Diff.')
        #plt.savefig('BC2SCRN01Cent2',dpi=200)
        
for i in range(1):
    plotim(i,False)
#    plotim(i,True)

diff2=[np.matrix.flatten(i) for i in diff]

rmsl=[(np.mean((i/np.amax(i))**2))**(1/2) for i in diff2]
print('Normalized RMS: '+str(np.mean(rmsl)))

f.close()