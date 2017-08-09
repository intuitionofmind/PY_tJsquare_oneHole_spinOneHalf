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

def SpinCurrent(v, hh):
    w = np.zeros(dim, dtype=complex)
    for i in range(dim):
        # print(dim, i)
        s = spinBasis[i % subDim] #i = h*dim_sub+s
        h = holeBasis[i // subDim]
        b = bin(s)[2:]
        b = b.rjust(numSite-1, '0')
        b = b[::-1]
        if h != hh:
            continue
        else:
            # sites = AroundSites(hh)
            # sites = [0, 3, 6, 7, 8, 5, 2, 1]
            sites = [0, 4, 8, 9, 10, 6, 2, 1]
            ls = len(sites)
            for j in range(ls):
                k = sites[j]
                kk = sites[(j+1) % ls]
                if k < h and kk < h and b[k] != b[kk]:
                    ss = Flip(s, k, kk)
                    l = FindState(ss, 0, subDim-1)
                    sign = 1.0
                    if b[k] == '1':
                        sign = -1.0
                    w[h*subDim+l] += sign*v[i]
                elif k > h and kk < h and b[k-1] != b[kk]:
                    ss = Flip(s, k-1, kk)
                    l = FindState(ss, 0, subDim-1)
                    sign = 1.0
                    if b[k-1] == '1':
                        sign = -1.0
                    w[h*subDim+l] += sign*v[i]
                elif k < h and kk > h and b[k] != b[kk-1]:
                    ss = Flip(s, k, kk-1)
                    l = FindState(ss, 0, subDim-1)
                    sign = 1.0
                    if b[k] == '1':
                        sign = -1.0
                    w[h*subDim+l] += sign*v[i]
                elif k > h and kk > h and b[k-1] != b[kk-1]:
                    ss = Flip(s, k-1, kk-1)
                    l = FindState(ss, 0, subDim-1)
                    sign = 1.0
                    if b[k-1] == '1':
                        sign = -1.0
                    w[h*subDim+l] += sign*v[i]
    return w

def LinkSpinCurrent(v, hh, link):
    w = np.zeros(dim, dtype=complex)
    for i in range(dim):
        # print(dim, i)
        s = spinBasis[i % subDim] #i = h*dim_sub+s
        h = holeBasis[i // subDim]
        b = bin(s)[2:]
        b = b.rjust(numSite-1, '0')
        b = b[::-1]
        if h != hh:
            continue
        else:
            # sites = AroundSites(hh)
            # sites = [0, 3, 6, 7, 8, 5, 2, 1]
            # sites = [0, 4, 8, 9, 10, 6, 2, 1]
            # ls = len(sites)
            # for j in range(ls):
            k = link[0]
            kk = link[1]
            if k < h and kk < h and b[k] != b[kk]:
                ss = Flip(s, k, kk)
                l = FindState(ss, 0, subDim-1)
                sign = 1.0
                if b[k] == '1':
                    sign = -0.0
                w[h*subDim+l] += sign*v[i]
            elif k > h and kk < h and b[k-1] != b[kk]:
                ss = Flip(s, k-1, kk)
                l = FindState(ss, 0, subDim-1)
                sign = 1.0
                if b[k-1] == '1':
                    sign = -0.0
                w[h*subDim+l] += sign*v[i]
            elif k < h and kk > h and b[k] != b[kk-1]:
                ss = Flip(s, k, kk-1)
                l = FindState(ss, 0, subDim-1)
                sign = 1.0
                if b[k] == '1':
                    sign = -0.0
                w[h*subDim+l] += sign*v[i]
            elif k > h and kk > h and b[k-1] != b[kk-1]:
                ss = Flip(s, k-1, kk-1)
                l = FindState(ss, 0, subDim-1)
                sign = 1.0
                if b[k-1] == '1':
                    sign = -0.0
                w[h*subDim+l] += sign*v[i]
    return w

dataDir = '/Users/wayne/Downloads/data/'
eigValsFile = 'eigenvalues%s_size%s_J0.3_10.0_%s_BC_%s_sigma%s.dat'
eigVecsFile = 'eigenvectors%s_size%s_J0.3_10.0_%s_BC_%s_sigma%s.dat'

numSite = numSiteX*numSiteY

size = 44
hp = 5
l = 0
J = 0.3
fold = 6

paras = (numEval, size, numSam, 'PP', 'False')
ene = LoadEigVal(os.path.join(dataDir, eigValsFile), paras)
print(paras, l, ene[l][:fold])

f = './translationWaveFunction_fold%s_J%s_%s_sigma%s.npy'
npyParas = (fold, J, 'PP', 'False')
wfArray = np.load(f % npyParas)

# wf = LoadEigVec(os.path.join(dataDir, eigVecsFile), paras, l)
np.set_printoptions(precision=4, suppress=True) # Only valid for printing numpy objects. 
for i in range(fold):
    wf = wfArray[i]
    # print(i, ene[l][i])
#     for j in range(sl):
        # link = []
        # link.append(sites[j])
        # link.append(sites[(j+1) % sl])
        # lc = np.vdot(w, LinkSpinCurrent(w, hp, link))
        # print(link, lc)
    sc = np.vdot(wf, SpinCurrent(wf, hp))
    print(i, '  ', '%.5f'%(ene[l][i]), '  ', '%.5f'%(np.imag(sc)))
