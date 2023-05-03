'''
Poor attempt to put both centroid and image predictions in one script, ended up just using the plotting down at the bottom.
'''
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


f=h5py.File('Data/cleansave220622K01imgs','r')

kesy=list(f.keys())
kesy2=[i[4:] for i in kesy]
kesy2.sort(key=int)
kesy=['page'+i for i in kesy2]

rX=np.array([np.delete(f[k]['X'][:],22)/np.amax(abs(np.delete(f[k]['X'][:],22))) for k in kesy]) #

rY=np.asarray([f[i]['mu'][:] for i in kesy])

normx=np.amax(abs(rY[:,0]))
normy=np.amax(abs(rY[:,1]))
Y=np.array([[rY[k,0]/normx,rY[k,1]/normy] for k in range(len(rY))])
X,Xtest,Y,Ytest=train_test_split(rX,Y,test_size=0.10)

model= tf.keras.models.Sequential()
modelcnn = tf.keras.models.Sequential()

def buildModel(model,modelname,inputs,labels,trainb):
    mcp_save = tf.keras.callbacks.ModelCheckpoint('Nets/mdl'+modelname+'_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    if modelname=='MUSIG':
        model.add(tf.keras.layers.Dense(100, activation='relu'))
        model.add(tf.keras.layers.Dense(100, activation='relu'))
        model.add(tf.keras.layers.Dense(100, activation='relu'))
        #model.add(tf.keras.layers.Dense(, activation='relu'))
        model.add(tf.keras.layers.Dense(len(labels[0])))
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),loss='mean_absolute_error')
    
        if trainb:
            history=model.fit(inputs,labels,batch_size=10,epochs=1200, validation_split=0.01, callbacks=[mcp_save], verbose=10)
             
            plot_loss(history,modelname)
         
            model.load_weights('Nets/mdl'+modelname+'_wts.hdf5')
        else:
            history=model.fit(inputs,labels,batch_size=5000,epochs=1, validation_split=0.01)
    
            model.load_weights('Nets/mdl'+modelname+'_wts_bestk01.hdf5')
    else:
        model.add(tf.keras.layers.Dense(200, activation='relu'))
        model.add(tf.keras.layers.Dense(200, activation='relu'))
        model.add(tf.keras.layers.Dense(10000, activation='relu'))
        # model.add(tf.keras.layers.Reshape((5,5,100)))
        # model.add(tf.keras.layers.Conv2D(100, (2,2), padding='same'))
        model.add(tf.keras.layers.Reshape((25,25,16)))
        model.add(tf.keras.layers.Conv2D(16, (4,4), padding='same'))
        model.add(tf.keras.layers.Reshape((200,50,1)))
        model.add(tf.keras.layers.Conv2D(1, (5,5), padding='same'))
        
        mcp_save = tf.keras.callbacks.ModelCheckpoint('Nets/mdl'+modelname+'_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss='mean_absolute_error')

        if trainb:
            history=model.fit(inputs,labels,batch_size=10,epochs=1200, validation_split=0.01, callbacks=[mcp_save], verbose=10)
            
            plot_loss(history,modelname)
        
            model.load_weights('Nets/mdl'+modelname+'_wts.hdf5')
        else:
            history=model.fit(inputs,labels,batch_size=5000,epochs=1, validation_split=0.01)
   
            model.load_weights('Nets/mdl'+modelname+'_wts_bestk01.hdf5')

buildModel(model, 'MUSIG',X, Y,False)
buildModel(modelcnn, 'imgreal',X,Yimg,False)

model.evaluate(Xtest,Ytest)

Ypred=model.predict(Xtest)

Ynew=np.array([[Ytest[k,0]*normx,Ytest[k,1]*normy] for k in range(len(Ypred))])

Ypnew=np.array([[Ypred[k,0]*normx,Ypred[k,1]*normy] for k in range(len(Ypred))])

diff=Ynew-Ypnew

calibx=63/1280
caliby=52/1000
print(round(np.mean(diff[:,0])*calibx,3),round(np.mean(diff[:,1])*caliby,3))

line=np.linspace(np.amin(Ynew[:,0]),np.amax(Ynew[:,0]),100)

plt.figure()
plt.plot(line,line,linestyle='--',color='black')
plt.scatter(Ynew[:,0],Ypnew[:,0],marker='.')
plt.xlabel('True X Centroid [$\mu$m]')
plt.ylabel('Predicted X Centroid [$\mu$m]')

plt.figure()
plt.scatter(Ypnew[:,0],Ypnew[:,1],marker='.',label='Prediction')
plt.scatter(Ynew[:,0],Ynew[:,1],marker='.',label='Real')
plt.xlabel('X [pixels]')
plt.ylabel('Y [pixels]')
plt.ylim(450,550)
plt.legend()

#plt.savefig('/home/johlun/figs/bestcentroid',dpi=150,bbox_inches='tight')

def plotim(i,name=''):
    Ypim=np.rot90(Ypred[i])
    Yim=np.rot90(Ytest[i])
    diffim=np.rot90(diff[i])
    centX=round(Ypnew[i,0]*calibx,2)
    centY=round(Ypnew[i,1]*caliby,2)
    fig,[ax1,ax2,ax3]=plt.subplots(3,1,sharex=True)
    #fig.suptitle('Centroid X:{} mm Y:{} mm'.format(round(Ypnew[i,0]*calibx,2),round(Ypnew[i,1]*caliby,2)),y=1.0)
    ax1.imshow(Ypim,vmin=0,vmax=max(np.matrix.flatten(Ytest[i])))
    ax1.set_title('Prediction')
    ax1.set_yticks([0,25,49])
    ax1.set_yticklabels([round(centY-25*caliby,2),centY,round(centY+25*caliby,2)])
    ax2.imshow(Yim,vmin=0,vmax=max(np.matrix.flatten(Ytest[i])))
    ax2.set_title('True')
    ax2.set_ylabel('Y [mm]')
    ax2.set_yticks([0,25,49])
    ax2.set_yticklabels([round(centY-25*caliby,2),centY,round(centY+25*caliby,2)])
    ax3.imshow(diffim,vmin=0,vmax=max(np.matrix.flatten(Ytest[i])))
    ax3.set_title('Abs. Diff.')
    ax3.set_xticks([0,50,100,150,199])
    ax3.set_xticklabels([round(centX-100*calibx,2),round(centX-50*calibx,2),centX,round(centX+50*calibx,2),round(centX+100*calibx,2)])
    ax3.set_yticks([0,25,49])
    ax3.set_yticklabels([round(centY-25*caliby,2),centY,round(centY+25*caliby,2)])
    fig.tight_layout(pad=0.1)
    ax3.set_xlabel('X [mm]')
    if len(name)>0:
        plt.savefig('Figs/'+name,dpi=200,bbox_inches='tight')

f.close()