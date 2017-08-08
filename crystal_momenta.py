#!/usr/bin/python3

import os
import math
import cmath as cm
import random
import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt

from auxi import holeBasis, spinBasis, subDim, dim, numSiteX, numSiteY, numSite, numEval, numSam
from auxi import FindState, Flip

def LoadEigVal(f, paras):
    n = paras[2]
    rawData = np.fromfile(f % paras, dtype=np.float)
    eigVal = np.zeros((n, numEval))
    print(len(rawData))
    for i in range(n): 
        for j in range(numEval):
            eigVal[i][j] = rawData[i*numEval+j]
    return eigVal

def LoadEigVec(f, paras, l):
    rawData = np.fromfile(f % paras, dtype=np.float)
    eigVec = np.zeros((paras[0], dim), dtype=complex)
    for i in range(paras[0]):
        for j in range(dim):
            eigVec[i][j] = complex(rawData[l*(paras[0]*dim*2)+i*dim*2+j*2], rawData[l*(paras[0]*dim*2)+i*dim*2+j*2+1])
    return eigVec

def TranslationX(v):
    w = np.zeros(dim, dtype=complex)
    for l in range(dim):
        h = holeBasis[l // subDim] # h = hy*numSiteX+hx 
        hx = h % numSiteX
        hy = h // numSiteX
        hh = hy*numSiteX+((hx+1) % numSiteX)

        s = spinBasis[l % subDim]
        b = bin(s)[2:]
        b = b.rjust(numSite-1, '0')
        b = b[::-1]
        config = b[:h]+'0'+b[h:] # Reconstruct the half-filled configuration.
        lst = [None]*numSite
        for j in range(numSiteY):
            for i in range(numSiteX):
                k = j*numSiteX+i
                kk = j*numSiteX+((i+1) % numSiteX)
                lst[kk] = config[k]
        newConfig = ''.join(lst)
        sign = np.power(-1., hh-h-(numSiteX-1)*numSiteY)  # fermionic sign arised  
        bb = newConfig[:hh]+newConfig[hh+1:]
        bb = bb[::-1]
        ss = int(bb, 2)
        n = FindState(ss, 0, subDim-1)
        w[hh*subDim+n] = sign*v[l]
    return w

def TranslationY(v):
    w = np.zeros(dim, dtype=complex)
    for l in range(dim):
        h = holeBasis[l // subDim] # h = hy*numSiteX+hx 
        hx = h % numSiteX
        hy = h // numSiteX
        hh = ((hy+1) % numSiteY)*numSiteX+hx

        s = spinBasis[l % subDim]
        b = bin(s)[2:]
        b = b.rjust(numSite-1, '0')
        b = b[::-1]
        config = b[:h]+'0'+b[h:] # Reconstruct the half-filled configuration.
        lst = [None]*numSite
        for j in range(numSiteY):
            for i in range(numSiteX):
                k = j*numSiteX+i
                kk = ((j+1) % numSiteY)*numSiteX+i
                lst[kk] = config[k]
        newConfig = ''.join(lst)
        sign = np.power(-1., hh-h-(numSiteY-1)*numSiteX)
        bb = newConfig[:hh]+newConfig[hh+1:]
        bb = bb[::-1]
        ss = int(bb, 2)
        n = FindState(ss, 0, subDim-1)
        w[hh*subDim+n] = sign*v[l]
    return w

dataDir = '/Users/wayne/Downloads/data/'
eigValsFile = 'eigenvalues%s_size%s_J0.3_10.0_%s_BC_%s_sigma%s.dat'
eigVecsFile = 'eigenvectors%s_size%s_J0.3_10.0_%s_BC_%s_sigma%s.dat'

size = 44

paras = (numEval, size, numSam, 'PP', 'False')
ene = LoadEigVal(os.path.join(dataDir, eigValsFile), paras)

l = 0
print(paras, l, ene[l][:40])
wfArray = LoadEigVec(os.path.join(dataDir, eigVecsFile), paras, l)
print(l, ene[l][:10])

fold = 6
T = np.zeros((fold, fold), dtype=complex)
for i in range(fold):
    for j in range(fold):
        T[i][j] = np.vdot(wfArray[i], TranslationX(wfArray[j]))
        print(i, j)
w, v = la.eig(T) # w[i] is the eigenvalue of the corresponding vector v[:, i]
# print('TranslationX', np.around(np.angle(w, deg=True), decimals=2))
print('Translation X', np.angle(w, deg=False))

# for i in range(6):
    # print(i, ene[l][i])
    # wf = wfArray[i]

l = 4
wfArray = LoadEigVec(os.path.join(dataDir, eigVecsFile), paras, l)
print(ene[l][:10])

for i in range(6):
    print(i, ene[l][i])
    wf = wfArray[i]
