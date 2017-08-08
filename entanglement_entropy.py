#!/usr/bin/python3

import os
import math
import random
import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from auxi import *

inter = 1

def EntanglementSpectrum(wf):
    Rho = np.zeros((numSite, numSite), dtype=complex)
    for i in range(0, numSite):
        for j in range(0, numSite):
            for k in range(0, subDim):
                Rho[i][j] += wf[i*subDim+k]*np.conjugate(wf[j*subDim+k])
    np.set_printoptions(precision=3, suppress=True)
#    print(Rho)
    es = np.linalg.eigvalsh(Rho)
    # print(es)
    return es

def EntanglementEntropy(wf):
    es = EntanglementSpectrum(wf)
    ee = -1.0*np.dot(es, np.log(np.absolute(es)))  # entanglement entropy
    return ee

def EntanglementEntropyArray():
    y = np.zeros(numSam)
    T = np.zeros((2, 2), dtype=complex)
    for i in range(numSam):
#         if np.absolute(ene[i][0]-ene[i][1]) < 1e-10:
            # wf0t = Translation(wf0[i])
            # wf1t = Translation(wf1[i])
            # T[0][0] = np.vdot(wf0[i], wf0t) vdot for complex conjugate...
            # T[0][1] = np.vdot(wf0[i], wf1t)
            # T[1][0] = np.vdot(wf1[i], wf0t)
            # T[1][1] = np.vdot(wf1[i], wf1t)
            # w, v = la.eig(T)
            # if w[0].imag > 0:
                # choice = 0
            # else:
                # choice = 1
            # print(i, np.angle(w, deg=True))
            # wf = v[:, choice][0]*wf0[i]+v[:, choice][1]*wf1[i]  construct eigenstates with specific momentum
        # else:
        wf = wf0[i]
        y[i] = EntanglementEntropy(wf)
        print(i)
    return y

def EntanglementSpectrumArray():
    ES = np.zeros((numSam, numSite))
    T = np.zeros((2, 2), dtype=complex)
    for i in range(numSam):
        if np.absolute(ene[i][0]-ene[i][1]) < 1e-12:
            wf0t = Translation(wf0[i])
            wf1t = Translation(wf1[i])
            T[0][0] = np.vdot(wf0[i], wf0t) # vdot for complex conjugate...
            T[0][1] = np.vdot(wf0[i], wf1t)
            T[1][0] = np.vdot(wf1[i], wf0t)
            T[1][1] = np.vdot(wf1[i], wf1t)
            w, v = la.eig(T)
            if w[0].imag > 0:
                choice = 0
            else:
                choice = 1
            print(i, np.angle(w, deg=True))
            wf = v[:, choice][0]*wf0[i]+v[:, choice][1]*wf1[i]  #construct eigenstates with specific momentum
        else:
            wf = wf0[i]
#        ES[i] = np.exp(EntanglementSpectrum(wf))
        ES[i] = EntanglementSpectrum(wf)
#        ES[i] = -1.0*np.log(np.absolute(EntanglementSpectrum(wf)))
#        print(i, ES[i])
#        print('Spectrum', np.log(np.absolute(ES[i])))
#        print(ES[i])
    return ES

x = np.linspace(step, step*numSam, num=numSam)

fig = plt.figure(figsize=(8, 8))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

ax1 = plt.subplot(111)
f = '/Users/wayne/Downloads/data/ee_size-%s%s_step-%s_sigma-false-OO.npy'
paras = (numSiteX, numSiteY, step)
ee = np.load(f % paras)
plt.plot(x[1::inter], ee[1::inter], '-o', color='red', label='$t$-$J$')

f = '/Users/wayne/Downloads/data/ee_size-%s%s_step-%s_sigma-true-OO.npy'
paras = (numSiteX, numSiteY, step)
ee = np.load(f % paras)
plt.plot(x[1::inter], ee[1::inter], '->', color='blue', label='$\sigma\cdot{t}$-$J$')

plt.xlabel('$J$', fontsize=fs)
plt.ylabel('mutual spin-charge EE', fontsize=fs)
plt.legend(loc='upper left', frameon=False, fontsize=fs)
plt.setp(ax1.get_xticklabels(), fontsize=fs)
plt.setp(ax1.get_yticklabels(), fontsize=fs)

image = 'ee_size-%s%s.png'
imageParas = (numSiteX, numSiteY)
fig.tight_layout()
plt.savefig(image % imageParas, format='PNG')
plt.show()
