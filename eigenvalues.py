#!/usr/bin/python3

import os
import math
import cmath as cm
import random
import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt

from auxi import holeBasis, spinBasis, subDim, dim, numSiteX, numSiteY, numEval, numSam
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

# To find the sites around the hole h.
def AroundSites(h):
    sList = []
    hx = h % numSiteX
    hy = h // numSiteX
    for x in range(numSiteX):
        for y in range(numSiteY):
            i = y*numSiteX+x
            l = np.sqrt((hx-x)**2+(hy-y)**2)
            if l < 2.0 and i != h:
                sList.append(i)
    # Clockwise.
    sListCW = []
    e0 = sList[0]
    e0x = e0 % numSiteX
    e0y = e0 // numSiteX
    sListCW.append(e0)
    for i in range(len(sList)):
        for e in sList:
            ex = e % numSiteX
            ey = e // numSiteY
            l = np.sqrt((e0x-ex)**2+(e0y-ey)**2)
            if e not in sListCW and 1.0 == l:
                sListCW.append(e)
                e0 = e
                e0x = e0 % numSiteX
                e0y = e0 // numSiteX
    return sListCW

def ChargeCurrent(v, pp):
    w = np.zeros(dim, dtype=complex)
    for i in range(dim):
        s = spinBasis[i % subDim] # i = h*dim_sub+s
        h = holeBasis[i // subDim]
        b = bin(s)[2:]
        b = b.rjust(numSite-1, '0')
        b = b[::-1]
        config = b[:h]+'2'+b[h:] # To restore the t-J configuration. 
        if '1' != config[pp]:
            continue
        else:
            # sites = AroundSites(pp)
            sites = [0, 4, 8, 9, 10, 6, 2, 1]
            # sites = [0, 3, 6, 7, 8, 5, 2, 1]
            ls = len(sites)
            for j in range(ls):
                k = sites[j]
                kk = sites[(j+1) % ls]
                temp = list(config)
                temp[k], temp[kk] = temp[kk], temp[k]
                hh = kk
                bb = ''
                for e in temp:
                    if e != '2':
                        bb = bb+e
                bb = bb[::1]
                ss = int(bb, 2)
                l = FindState(ss, 0, subDim-1)
                if h == k:
                    sign = np.power(-1.0, np.absolute(k-kk)-1)
                elif h == kk:
                    sign = np.power(-1.0, np.absolute(k-kk))
                w[hh*subDim+l] += sign*v[i]
    return w

dataDir = '/Users/wayne/Downloads/data/'
eigValsFile = 'eigenvalues%s_size%s_J0.3_10.0_%s_BC_%s_sigma%s.dat'
eigVecsFile = 'eigenvectors%s_size%s_J0.3_10.0_%s_BC_%s_sigma%s.dat'

numSite = numSiteX*numSiteY

size = 44

pp = 4

paras = (numEval, size, numSam, 'PP', 'True')
ene = LoadEigVal(os.path.join(dataDir, eigValsFile), paras)

l = 0
for i in range(50):
    print(ene[l][i])
    

l = 4

for i in range(50):
    print(ene[l][i])
