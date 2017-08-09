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
l = 0
J = 0.3

fold = 6 # Ground state degeneracy. 
paras = (numEval, size, numSam, 'PP', 'False')
ene = LoadEigVal(os.path.join(dataDir, eigValsFile), paras)
wfArray = LoadEigVec(os.path.join(dataDir, eigVecsFile), paras, l)

np.set_printoptions(precision=2, suppress=True) # Only valid for printing numpy objects. 

print(l, ene[l][:fold])

# test commutation relation for translational operations.
'''
print('Test commutation relation for translational operations:')
print(TranslationX(TranslationY(wfArray[0]))-TranslationY(TranslationX(wfArray[0])))
'''

Temp = np.zeros((fold, fold), dtype=complex)
for i in range(fold):
    for j in range(fold):
        Temp[i][j] = np.vdot(wfArray[i], wfArray[j])
print('Original matrix')
print(np.absolute(Temp))

for i in range(fold):
    for j in range(fold):
        Temp[i][j] = np.vdot(wfArray[i], TranslationX(wfArray[j]))
        print(i, j)
wx, vx = la.eig(Temp) # w[i] is the eigenvalue of the corresponding vector v[:, i]

# To find the eigenvectors in terms of x-translation.
wfArrayX = np.zeros((fold, dim), dtype=complex)

for i in range(fold):
    for j in range(fold):
        wfArrayX[i] += vx[:, i][j]*wfArray[j]

# To find the eigenvectors in terms of both x- and y-translation.
wfArrayXY = np.zeros((fold, dim), dtype=complex)

print('0 1')
print(np.angle(wx[0]), np.angle(np.vdot(wfArrayX[0], TranslationY(wfArrayX[0]))))
print(np.angle(wx[1]), np.angle(np.vdot(wfArrayX[1], TranslationY(wfArrayX[1]))))
wfArrayXY[0] = wfArrayX[0]
wfArrayXY[1] = wfArrayX[1]

foldAgain = 2 # Degeneracy fold after figureing out quantum numbers for x-translation.
TempAgain = np.zeros((foldAgain, foldAgain), dtype=complex)

start = 2
for i in range(foldAgain):
    for j in range(foldAgain):
        TempAgain[i][j] = np.vdot(wfArrayX[start+i], TranslationY(wfArrayX[start+j]))
wy, vy = la.eig(TempAgain) # w[i] is the eigenvalue of the corresponding vector v[:, i]
print('2 3')
print(np.angle(wx[start+0]), np.angle(wy[0]))
print(np.angle(wx[start+1]), np.angle(wy[1]))
for i in range(foldAgain):
    for j in range(foldAgain):
        wfArrayXY[start+i] += vy[:, i][j]*wfArrayX[start+j]

start = 4
for i in range(foldAgain):
    for j in range(foldAgain):
        TempAgain[i][j] = np.vdot(wfArrayX[start+i], TranslationY(wfArrayX[start+j]))
wyy, vyy = la.eig(TempAgain) # w[i] is the eigenvalue of the corresponding vector v[:, i]
print('4 5')
print(np.angle(wx[start+0]), np.angle(wyy[0]))
print(np.angle(wx[start+1]), np.angle(wyy[1]))
for i in range(foldAgain):
    for j in range(foldAgain):
        wfArrayXY[start+i] += vyy[:, i][j]*wfArrayX[start+j]

vFile = 'translationWaveFunction_fold%s_J%s_%s_sigma%s'
np.save(vFile % (fold, J, 'PP', 'False'), wfArrayXY)
