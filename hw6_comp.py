#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 23:19:20 2020

@author: jahnvirc
"""

import numpy as np
import pickle
import skimage
import skimage.measure
import time
from skimage.util import view_as_windows
from cross_entropy import *
from lag import *
from llsr import *
from pixelhop2 import *

" Perform Image shrinking "
" the function to view as windows is becasue we need it to use the channel last input"
"w stands for the win parameter and h for the hop parameter"
def img_srk(img, shr_arg):  
  w = shr_arg['win']
  if shr_arg['hop']==1:
    img = view_as_windows(img, (1,3,3,img.shape[-1]), (1,1,1,img.shape[-1]))
    img = img.reshape(img.shape[0], img.shape[1], img.shape[2], -1)
    img = skimage.measure.block_reduce(img, (1,2,2,1), np.max)
  elif shr_arg['hop']==2:
    img = view_as_windows(img, (1,3,3,img.shape[-1]), (1,1,1,img.shape[-1]))
    img = img.reshape(img.shape[0], img.shape[1], img.shape[2], -1)
    img = skimage.measure.block_reduce(img, (1,2,2,1), np.max)
  elif shr_arg['hop']==3:
    img = view_as_windows(img, (1,3,3,img.shape[-1]), (1,1,1,img.shape[-1]))
    img = img.reshape(img.shape[0], img.shape[1], img.shape[2], -1)
  elif shr_arg['hop']==4:
    img = view_as_windows(img, (1,3,3,img.shape[-1]), (1,1,1,img.shape[-1]))
    img = img.reshape(img.shape[0], img.shape[1], img.shape[2], -1)
  return img

def ce__calc(feat,ytr):
  ce = Cross_Entropy(num_class=10, num_bin=5)
  feat = feat.reshape((feat.shape[0],-1))
  feat_ce = np.zeros(feat.shape[-1])
  for k in range(feat.shape[-1]):
    feat_ce[k] = ce.compute(feat[:,k].reshape(-1,1), ytr)
  return feat_ce
"Image Concatination"
def cct(img, concatArg):
    return img
"Feature Selection"
def feal__Sel(dat,i_arr,ns):
  op = dat.reshape((dat.shape[0],-1))
  if ns<1:
      Ns = int(ns*i_arr.shape[-1])
  else:
    Ns=ns
  min_val = min(Ns,op.shape[-1])
  res = np.zeros((op.shape[0],min_val))
  j=1
  for i in i_arr:
    if j<min_val:
      res[:,j] = op[:,i]
      j+=1
    if j==Ns:
      exit
  return res

"LAG layer"
def lag_val(xtr, ytr,xte, yte,alp=5):
    lag = LAG(encode='distance', num_clusters=[5,5,5,5,5,5,5,5,5,5], alpha=5, learner=LLSR(onehot=False))  
    lag.fit(xtr, ytr)
    xtr_tran = lag.transform(xtr)
    xte_tran = lag.transform(xte)
    print("train accuracy: %s"%str(lag.score(xtr, ytr)))
    print("test accuracy.: %s"%str(lag.score(xte, yte)))
    return xtr_tran,xte_tran


"Deciding on the neighbouring pixel parameters"
SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'useDC':True, 'batch':None, 'cw': True},{'num_AC_kernels':-1, 'needBias':True, 'useDC':True, 'batch':None, 'cw': True},{'num_AC_kernels':-1, 'needBias':True, 'useDC':True, 'batch':None, 'cw': True}]
shrinkArgs = [{'func':img_srk, 'win':5, 'hop':1},{'func': img_srk, 'win':5,'hop':2},{'func': img_srk, 'win':5, 'hop':3}]
concatArg = {'func':cct}

"Declaring the training and testing dataset"
from keras.datasets import cifar10
(xtr, ytr), (xte, yte) = cifar10.load_data()
xtr_50k = xtr.astype('float32')/255
xte = xte.astype('float32')/255
"For changing the number of training sets in the modules 2 and 3"
tr_sz = np.asarray(xtr.shape)
tr_sz[0] = 10000
tr_dat = np.zeros(tr_sz)
func = np.where(ytr==1)
k=1
for i in range(10):
  func = np.where(ytr==i)
  func = func[0][0:1000]
  for i in func:
    if k>=10000:
      break
    tr_dat[k,:,:,:] = xtr_50k[i,:,:,:]
    k=k+1
"For time calculation: Start"
start_time = time.time()
"To perform pixelhop"
pix__hop = Pixelhop2(depth=4, TH1=0.01, TH2=0.0001, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)
pix__hop.fit(tr_dat)
tr__out = pix__hop.transform(tr_dat)
te__trf = pix__hop.transform(xte)

"Loading the pixelhop weights"
with open('pixelhop2.pkl','wb') as f:
    pickle.dump(pix__hop,f)
with open('pixelhop2.pkl', 'rb') as f:
    clf2 = pickle.load(f)    
op = pix__hop.transform(xtr_50k)
"CRoss Entropy Calculation"
ce__h1 = ce__calc(op[0],ytr)
ce__h2 = ce__calc(op[1],ytr)
ce__h3 = ce__calc(op[2],ytr)
ce__h4 = ce__calc(op[3],ytr)

Ns = 0.5
"Sorting the Cross Entropy"
ce_sor1 = ce__h1.argsort()
ce_sor2 = ce__h2.argsort()
ce_sor3 = ce__h3.argsort()
ce_sor4 = ce__h4.argsort()


Ns = 1000
tr_fs1 = feal__Sel(op[0],ce_sor1,Ns)
tr_fs2 = feal__Sel(op[1],ce_sor2,Ns)
tr_fs3 = feal__Sel(op[2],ce_sor3,Ns)
tr_fs4 = feal__Sel(op[3],ce_sor4,Ns)
te_fs1 = feal__Sel(te__trf[0],ce_sor1,Ns)
te_fs2 = feal__Sel(te__trf[1],ce_sor2,Ns)
te_fs3 = feal__Sel(te__trf[2],ce_sor3,Ns)
te_fs4 = feal__Sel(te__trf[3],ce_sor4,Ns)


alp = 5
"LAG layer computation"
lag1 = lag_val(tr_fs1,ytr,te_fs1, yte,alp)
lag2 = lag_val(tr_fs2,ytr,te_fs2, yte,alp)
lag3 = lag_val(tr_fs3,ytr,te_fs3, yte,alp)
lag4 = lag_val(tr_fs4,ytr,te_fs4, yte,alp)

lag_val0 = np.concatenate((lag1[0],lag2[0],lag3[0],lag4[0]),axis = 1)
lag_val1 = np.concatenate((lag1[1],lag2[1],lag3[1],lag4[1]),axis = 1)
"Ending Time"
stop_time = time.time()
"Time Computation"
print('Total time taken for training:',stop_time-start_time)



"Classification"

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score as acc_scr
from sklearn import preprocessing

ssvc=preprocessing.StandardScaler()
fe_tran = ssvc.fit_transform(lag_val0)
fe_tran_te = ssvc.transform(lag_val1)
clf=SVC().fit(fe_tran, ytr) 
print('Train Accuracy:', acc_scr(ytr,clf.predict(fe_tran)))
print('Test Accuracy:', acc_scr(yte,clf.predict(fe_tran_te)))

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=20, random_state=0)
fe_tran = ssvc.fit_transform(lag_val0)
fe_tran_te = ssvc.transform(lag_val1)  
clf = clf.fit(fe_tran, ytr)
print('Train Accuracy:', acc_scr(ytr,clf.predict(fe_tran)))
print('Test Accuracy:', acc_scr(yte,clf.predict(fe_tran_te)))

"COnfusion Matrices"

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plooot
import itertools
plooot.rcParams['figure.figsize'] = [10,7]

def plt_cm(cm, classes,normalize=True, title='Confusion matrix',cmap=plooot.cm.Blues):
  print(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
  plooot.imshow(cm, interpolation='nearest', cmap=cmap)
  plooot.title(title)
  plooot.colorbar()
  tm = np.arange(len(classes))
  plooot.xticks(tm, classes, rotation=45)
  plooot.yticks(tm, classes)

  f_mat = '.2f'
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plooot.text(j, i, format(cm[i, j], f_mat),horizontalalignment="center",color="white" if cm[i, j] > (cm.max()/2) else "black")
  plooot.tight_layout()
  plooot.ylabel('True label')
  plooot.xlabel('Predicted label')
  plooot.show()
p_test = clf.predict(fe_tran_te)
cm = confusion_matrix(yte, p_test)
plt_cm(cm, list(range(10)))



