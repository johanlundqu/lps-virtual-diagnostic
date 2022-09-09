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

X=[[scanl['phs01'][i]/max(scanl['phs01'][:]),scanl['phs02'][i]/max(scanl['phs02'][:]),scanl['volt01'][i]/max(scanl['volt01'][:]),scanl['volt02'][i]/max(scanl['volt02'][:])] for i in range(len(scanl['volt02'][:]))]

for j in range(len(scanl['volt02'][:])):
    for r in range(21):
        X[j].append(bpmf[kesy[j]]['Cx'][r]/max(abs(bpmf[kesy[j]]['Cx'][:])))
        X[j].append(bpmf[kesy[j]]['Cy'][r]/max(abs(bpmf[kesy[j]]['Cy'][:])))

X=np.asarray(X)

Y=np.asarray([np.matrix.flatten(f[i][:]/np.amax(f[i])) for i in kesy])

X,Xtest,Y,Ytest=train_test_split(X,Y,test_size=0.10)

model= tf.keras.models.Sequential()

def buildModel(model,modelname,inputs,labels,trainb=True):
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    #model.add(tf.keras.layers.Dense(, activation='relu'))
    model.add(tf.keras.layers.Dense(len(labels[0])))

    
    
    mcp_save = tf.keras.callbacks.ModelCheckpoint('Nets/mdl'+modelname+'_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=100e-4),loss='mean_absolute_error')
    if trainb:
        history=model.fit(inputs,labels,batch_size=50,epochs=800, validation_split=0.01,verbose=1, callbacks=[mcp_save])
        
        plot_loss(history,modelname)
    
        model.load_weights('Nets/mdl'+modelname+'_wts.hdf5')
    else:
        history=model.fit(inputs,labels,batch_size=5000,epochs=1, validation_split=0.01)
        #print('No')
        model.load_weights('Nets/mdl'+modelname+'_wts_best.hdf5')

buildModel(model, 'heat',X, Y,False)


model.evaluate(Xtest,Ytest)

Ypred=model.predict(Xtest)

diff=abs(Ytest-Ypred)

fftYt=np.fft.fft(Ytest)
fftYp=np.fft.fft(Ypred)
ffdiff=np.fft.fft(Ytest)-np.fft.fft(Ypred)


def plotim(i,fft=False):
    if fft:
        fftYpim=np.flip(np.reshape(fftYp[i],(50,50)).T)
        fftYtim=np.flip(np.reshape(fftYt[i],(50,50)).T)
        ffdiffim=np.flip(np.reshape(ffdiff[i],(50,50)).T)
        
        fig,[ax1,ax2,ax3]=plt.subplots(1,3,sharey=True)
        #fig.suptitle('BC2 SCRN-01',y=0.8)
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
        Ypim=np.flip(np.reshape(Ypred[i],(50,50)).T)
        Yim=np.flip(np.reshape(Ytest[i],(50,50)).T)
        diffim=np.flip(np.reshape(diff[i],(50,50)).T)
        
        fig,[ax1,ax2,ax3]=plt.subplots(1,3,sharey=True)
        ax1.imshow(Ypim,vmin=0,vmax=max(Ytest[i]))
        ax1.set_title('Prediction')
        ax1.set_ylabel('Y []')
        ax2.imshow(Yim,vmin=0,vmax=max(Ytest[i]))
        ax2.set_title('True')
        ax2.set_xlabel('X []')
        ax3.imshow(diffim,vmin=0,vmax=max(Ytest[i]))
        ax3.set_title('Abs. Diff.')
        #plt.savefig('/home/johlun/figs/threeheat',dpi=100)

for i in range(0,100,10):
    plotim(i,False)
    plotim(i,True)
    
rmsl=[(np.mean(i**2))**(1/2) for i in diff]
maxl=[max(i) for i in diff]
normrmsl=[rmsl[i]/maxl[i] for i in range(len(rmsl))]
print('Normalized RMS: '+str(np.mean(normrmsl)))

meantrue=[i-np.mean(i) for i in Ytest]
Rl=[1-sum(diff[i]**2)/sum(meantrue[i]**2) for i in range(len(diff))]
print('SLAC Score [R²]: '+str(np.mean(Rl)))

f.close()
scanf.close()