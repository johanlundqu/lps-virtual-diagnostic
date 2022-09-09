import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

rX=np.array([np.append(f[k]['X'][:3],f[k]['X'][3:]/np.amax(abs(f[k]['X'][3:]))) for k in kesy]) #

rY=np.asarray([f[i]['mu'][:] for i in kesy])

normx=np.amax(abs(rY[:,0]))
normy=np.amax(abs(rY[:,1]))
Y=np.array([[rY[k,0]/normx,rY[k,1]/normy] for k in range(len(rY))])
X,Xtest,Y,Ytest=train_test_split(rX,Y,test_size=0.10, random_state=20)

model= tf.keras.models.Sequential()

def buildModel(model,modelname,inputs,labels,trainb):
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    #model.add(tf.keras.layers.Dense(, activation='relu'))
    model.add(tf.keras.layers.Dense(len(labels[0])))
    
    mcp_save = tf.keras.callbacks.ModelCheckpoint('Nets/mdl'+modelname+'_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),loss='mean_absolute_error')

    if trainb:
        history=model.fit(inputs,labels,batch_size=10,epochs=12000, validation_split=0.01, callbacks=[mcp_save], verbose=0)
         
        plot_loss(history,modelname)
     
        model.load_weights('Nets/mdl'+modelname+'_wts.hdf5')
    else:
        history=model.fit(inputs,labels,batch_size=5000,epochs=1, validation_split=0.01)

        model.load_weights('Nets/mdl'+modelname+'_wts_bestallfilev2.hdf5')

buildModel(model, 'MUSIG',X, Y,False)


model.evaluate(Xtest,Ytest)

Ypred=model.predict(Xtest)

Ynew=np.array([[Ytest[k,0]*normx,Ytest[k,1]*normy] for k in range(len(Ypred))])

Ypnew=np.array([[Ypred[k,0]*normx,Ypred[k,1]*normy] for k in range(len(Ypred))])

diff=Ynew-Ypnew

calibx=63/1280
caliby=52/1000
print(round(np.mean(diff[:,0])*calibx,3),round(np.mean(diff[:,1])*caliby,3))

line=np.linspace(np.amin(Ynew[:,0]),np.amax(Ynew[:,0]),100)
liney=np.linspace(np.amin(Ynew[:,1]),np.amax(Ynew[:,1]),100)

plt.figure()
plt.plot(line*calibx,line*calibx,linestyle='--',color='black')
plt.scatter(Ynew[:,0]*calibx,Ypnew[:,0]*calibx,marker='.')
plt.xlabel('True X Centroid [mm]')
plt.ylabel('Predicted X Centroid [mm]')

plt.figure()
plt.scatter(Ypnew[:,0],Ypnew[:,1],marker='.',label='Prediction')
plt.scatter(Ynew[:,0],Ynew[:,1],marker='.',label='Real')
plt.xlabel('X [pixels]')
plt.ylabel('Y [pixels]')
plt.ylim(450,550)
plt.legend()

#plt.savefig('/home/johlun/figs/bestcentroid',dpi=150,bbox_inches='tight')

f.close()