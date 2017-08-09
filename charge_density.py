#!/usr/bin/python3

import os
import math
import cmath as cm
import random
import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt
import matplotlib as mpl

from auxi import holeBasis, spinBasis, subDim, dim, numSiteX, numSiteY, numEval, numSam, fs
from auxi import FindState, Flip

# paras = (numEval, size, numSam, 'PP', 'False')
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

def ChargeDensity(v, r):
    w = np.zeros(dim, dtype=complex)
    for i in range(dim):
        # print(dim, i)
        s = spinBasis[i % subDim] #i = h*dim_sub+s
        h = holeBasis[i // subDim]
        if h != r:
            continue
        else:
            w[i] = v[i]
    return w

def ChargeDensityDistri(v):
    cd = np.zeros((numSiteX, numSiteY), dtype=complex)
    for i in range(numSiteX):
        for j in range(numSiteY):
            cd [i][j] = np.vdot(v, ChargeDensity(v, j*numSiteX+i))
    return cd

# dataDir = '/Users/wayne/Downloads/data/'
dataDir = './'

l = 0
J = 0.3
fold = 6

f = 'translationWaveFunction_fold%s_J%s_%s_sigma%s.npy'
paras = (fold, J, 'PP', 'False')
wfArray = np.load(os.path.join(dataDir, f) % paras)

CD = []
for i in range(fold):
    wf = wfArray[i]
    CD.append(np.real(ChargeDensityDistri(wf)))

xt = [0, 1, 2, 3]
yt = [0, 1, 2, 3]
fig = plt.figure(figsize=(12, 10))
# fig, axes = plt.subplots(nrows=3, ncols=3)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# plt.suptitle('$\sigma\cdot{t}$-$J$, $J=0.3$, OBC', fontsize=fs)
fig.suptitle('${t}$-$J$ model, $J=0.3$, PBC', fontsize=fs)

import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(3, 2)

i = 0

np.set_printoptions(precision=4, suppress=True)
for s in gs:
    ax = fig.add_subplot(s)
    print(CD[i])
    im = ax.imshow(CD[i], vmin=0.0, vmax=0.16)
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    i += 1

gs.tight_layout(fig, rect=[0.0, 0., 0.8, 0.95]) # [left, bottom, right, top] that the whole subplot area will fit into. Default is [0, 0, 1, 1] 
cax = fig.add_axes([0.9, 0.1, 0.04, 0.8])
fig.colorbar(im, cax=cax, use_gridspec=True)

image = 'chargeDensity%s_J%s_boundary%s_sigma%s.pdf'
plt.savefig(image % paras, format='pdf')
# fig.tight_layout()
plt.show()

