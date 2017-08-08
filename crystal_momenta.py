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

# Input should be a array consisting of degenerate eigenvectors.
def TransEigenVec(wfArray, fold):
    T = np.zeros((fold, fold), dtype=complex)
    wfArrayX = np.zeros((fold, dim), dtype=complex)
    wfArrayXY = np.zeros((fold, dim), dtype=complex)
    for i in range(fold):
        for j in range(fold):
            T[i][j] = np.vdot(wfArray[i], TranslationX(wfArray[j]))
    wx, vx = la.eig(T) # w[i] is the eigenvalue of the corresponding vector v[:, i]
    for i in range(fold):
        for j in range(fold):
            wfArrayX[i] += vx[:, i][j]*wfArray[j]
    for i in range(fold):
        for j in range(fold):
            T[i][j] = np.vdot(wfArrayX[i], TranslationY(wfArrayX[j]))
    wy, vy = la.eig(T) # w[i] is the eigenvalue of the corresponding vector v[:, i]
    for i in range(fold):
        for j in range(fold):
            wfArrayXY[i] += vy[:, i][j]*wfArrayX[j]
    return np.angle(wx), np.angle(wy), wfArrayXY
 
dataDir = '/Users/wayne/Downloads/data/'
eigValsFile = 'eigenvalues%s_size%s_J0.3_10.0_%s_BC_%s_sigma%s.dat'
eigVecsFile = 'eigenvectors%s_size%s_J0.3_10.0_%s_BC_%s_sigma%s.dat'

size = 44
l = 0
fold = 6

paras = (numEval, size, numSam, 'PP', 'False')
ene = LoadEigVal(os.path.join(dataDir, eigValsFile), paras)
wfArray = LoadEigVec(os.path.join(dataDir, eigVecsFile), paras, l)

print(l, ene[l][:fold])
T = np.zeros((fold, fold), dtype=complex)
for i in range(fold):
    for j in range(fold):
        T[i][j] = np.vdot(wfArray[i], wfArray[j])
print('original matrix')
print(T)

for i in range(fold):
    for j in range(fold):
        T[i][j] = np.vdot(wfArray[i], TranslationX(wfArray[j]))
        # print(i, j)
wx, vx = la.eig(T) # w[i] is the eigenvalue of the corresponding vector v[:, i]
print('x-translation momenta', np.angle(wx, deg=False))

# To find the eigenvectors in terms of x-translation.
wfArrayX = np.zeros((fold, dim), dtype=complex)
for i in range(fold):
    for j in range(fold):
        wfArrayX[i] += vx[:, i][j]*wfArray[j]
    print(np.angle(wx[i]), np.vdot(wfArrayX[i], wfArrayX[i]))

for i in range(fold):
    for j in range(fold):
        T[i][j] = np.vdot(wfArrayX[i], TranslationY(wfArrayX[j]))
        # print(i, j)
wxy, vxy = la.eig(T) # w[i] is the eigenvalue of the corresponding vector v[:, i]
print('y-translation momenta', np.angle(wxy, deg=False))

# To find the eigenvectors in terms of x-translation.
wfArrayXY = np.zeros((fold, dim), dtype=complex)
for i in range(fold):
    for j in range(fold):
        wfArrayXY[i] += vxy[:, i][j]*wfArrayX[j]

# test
for i in range(fold):
    for j in range(fold):
        T[i][j] = np.vdot(wfArrayXY[i], TranslationX(wfArrayXY[j]))
        print(i, j)
print('test x-translated matrix')
print(np.absolute(T))

for i in range(fold):
    for j in range(fold):
        T[i][j] = np.vdot(wfArrayXY[i], TranslationY(wfArrayXY[j]))
        print(i, j)
print('test y-translated matrix')
print(np.absolute(T))
# wxy, vxy = la.eig(T) # w[i] is the eigenvalue of the corresponding vector v[:, i]
# print('test x-translation momenta', np.angle(wxy, deg=False))


