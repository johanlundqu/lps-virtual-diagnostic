# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:22:47 2022

@author: johlun
"""

import h5py
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
#posl=[0.0,0.0,0.0,0.0,-441/822011,-2204/822011,9383/822011,-492923/822011,822011/822011,-215480/822011,20899/822011,-60686/822011,3657/822011,-40515/822011,12602/822011,15082/822011,-66703/822011,2329/357194,-4142/357194,1613/357194,18854/357194,19173/357194,48438/357194,-3532/357194,-177598/357194,-179304/357194,357194/357194,6437/357194,2783/357194,38907/357194]

realf=h5py.File('V:\controlroom\johlun\save220622K02LSFT275','r') #'V:\controlroom\johlun\save220622K02LSFT315'
backf=h5py.File('V:\controlroom\johlun\save220622K02LSFT315','r')
filltime=2.75
kesy=list(realf.keys())
kesy2=[i[4:] for i in kesy]
kesy2.sort(key=int)
kesy=['page'+i for i in kesy2]

sety=list(realf['page1'].keys())
kesy2=[i[3:] for i in sety]
kesy2.sort(key=int)
sety=['set'+i for i in kesy2]

def sorter():
    a=[]
    sl=[float(i[3]) for i in bpl]

    while len(a)<len(bpl) and len(sl)>0:
        for i in range(len(bpl)):
            if float(bpl[i][3])==min(sl,default='EMPTY'):
                a.append(bpl[i])
                sl.remove(min(sl,default='EMPTY'))
    a=np.asarray(a)
    return a

bpl2=[]
img=[]
profX=[]
profY=[]
setl=[]
ct2=[]

backim=backf[kesy[0]][sety[0]]['img'][:]
backx=sig.medfilt(np.sum(backim[:,450:550],axis=1))
backy=sig.medfilt(np.sum(backim,axis=0))
meanX=[]
meanY=[]

for key in kesy:
    for setr in sety:
        ct=realf[key][setr]['CT'][:,-1]
        ct2.append([float(k) for k in ct])
        if sum(ct2[-1])==0:
            ct2.pop(-1)
            continue
        else:
            im=realf[key][setr]['img'][:]-backim
            proX=sig.medfilt(np.sum(im[:,450:550],axis=1))
            proY=sig.medfilt(np.sum(im,axis=0))
            maxx=[j for j in range(len(proX)) if (proX)[j]==np.amax(proX)][0]
            maxy=[j for j in range(len(proY)) if (proY)[j]==np.amax(proY)][0]
            sliceim=realf[key][setr]['img'][maxx-100:maxx+100,maxy-25:maxy+25]
            if len(sliceim)==200 and len(sliceim[0])==50:
                profX.append(proX)
                profY.append(proY)
                meanX.append(maxx)
                meanY.append(maxy)
                bpl=realf[key][setr]['bpm'][:]
                a=sorter()
                bpl2.append(a)
                img.append(sig.medfilt(sliceim))
                setl.append([float(realf[key][setr]['phs'][-2][1]),float(realf[key][setr]['phs'][-1][1]),filltime])
            else:
                continue
    #bpl.append([realf[i][setr]['bpm'][:] for setr in sety])   
    #img.append([realf[i][setr]['img'][:] for setr in sety])

#Some SUM stuff for checking
sumct=[sum(i) for i in ct2]
suml=[sum(profX[i])+sum(profY[i]) for i in range(len(profX))]
ct2=np.asarray(ct2)

'''
plt.figure()
plt.plot(ct2[:,1],suml,'.')
plt.ylim(2.6e7,2.9e7)
plt.figure()
plt.plot(ct2[:,1],suml,'.')
plt.ylim(2.2e7,2.5e7)
'''
#Some MUSIG stuff
'''
backx=profX[0]
backy=profY[0]


for i in profX-backx:
    meanX.append([j for j in range(len(i)) if i[j]==np.amax(i)][0])
for i in profY-backy:
    meanY.append([j for j in range(len(i)) if i[j]==np.amax(i)][0])
 '''
#round(np.average(range(len(i)), weights = i-backx),2)
#meanY=[ for i in profY[1:]] #round(np.average(range(len(i)), weights = i-backy),2)


strl=[b'i-s01b/dia/bpl-01',b'i-bc1/dia/bpl-01',
      b'i-bc1/dia/bpl-02',b'i-bc1/dia/bpl-03',
      b'i-bc2/dia/bpl-01',b'i-ex1/dia/bpl-01',
b'i-s04b/dia/bpl-01',
b'i-s06a/dia/bpl-01',
b'i-s08a/dia/bpl-01',
b'i-ex1/dia/bpl-02',
b'i-ex3/dia/bpl-01',
b'i-ms1/dia/bpl-01',
b'i-ms1/dia/bpl-02',
b'i-ms1/dia/bpl-03',
b'i-ms2/dia/bpl-01',
b'i-s12a/dia/bpl-01',
b'i-s14b/dia/bpl-01',
b'i-s16b/dia/bpl-01',
b'i-s19b/dia/bpl-01',
b'i-ms3/dia/bpl-01',
b'i-ms3/dia/bpl-02',]

positl=[]
#setl=[0,0,0,0]

for i in bpl2:
    positl.append([float(j[1]) for j in i if j[0] in strl]+[float(j[2]) for j in i if j[0] in strl])
positl2=np.array([setl[k]+positl[k] for k in range(len(positl))])
'''
hf=h5py.File('data/cleansave2206K02TEST','w')
for i in range(len(positl2)):
    grp=hf.create_group('page'+str(i+1))
    grp.create_dataset('X',data=positl2[i])
    grp.create_dataset('Y',data=img[i])
hf.close()

#MUSIG saving
hf=h5py.File('data/cleansave220622K01imgs','w')
for i in range(len(positl2)):
    grp=hf.create_group('page'+str(i+1))
    grp.create_dataset('X',data=positl2[i])
    grp.create_dataset('Y',data=img[i])
    grp.create_dataset('mu',data=[meanX[i],meanY[i]])
hf.close()


k=0
for i in img[::10]:
    plt.figure()
    plt.imshow(i)
    plt.title(str(meanX[k])+','+str(meanY[k])+','+str(k))
    k+=10

imgmax=[np.amax(j) for j in img]
plt.figure()
plt.plot(imgmax)
'''
realf.close()
backf.close()