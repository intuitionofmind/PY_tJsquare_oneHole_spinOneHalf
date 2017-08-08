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

fig = plt.figure(figsize=(8, 12))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
scale = 1.3

# paras = (numEval, numSite, numSam, 'PBC', 'false', flux)
# ene = LoadEigVal(paras)
# wf0 = LoadEigVec(paras, 0)
# wf1 = LoadEigVec(paras, 1)

ax1 = plt.subplot(211)
# y = EntanglementSpectrumArray()
f = '/Users/wayne/Downloads/data/es_size-%s%s_step-%s_sigma-false-OO.npy'
paras = (numSiteX, numSiteY, step)
es = np.load(f % paras)
plt.plot(x[1::inter], es[1::inter], 'r_', markeredgewidth=1.5, markersize=5)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax1.get_yticklabels(), fontsize=fs*scale)
plt.ylim([0, 0.5])
plt.title('(a)  $t$-$J$', fontsize=fs*scale, x=0.1, y=0.9)
# ax1.text(0.1, 0.91, '', fontsize=fs*1.2, transform=ax1.transAxes)

ax2 = plt.subplot(212, sharex=ax1)
f = '/Users/wayne/Downloads/data/es_size-%s%s_step-%s_sigma-true-OO.npy'
paras = (numSiteX, numSiteY, step)
es = np.load(f % paras)
plt.plot(x[1::inter], es[1::inter], 'r_', markeredgewidth=1.5, markersize=5)
plt.setp(ax2.get_xticklabels(), fontsize=fs*scale)
plt.setp(ax2.get_yticklabels(), fontsize=fs*scale)
plt.xlabel('$J$', fontsize=fs*scale)
plt.ylim([0, 0.5])
plt.title('(b)  $\sigma\cdot{t}$-$J$', fontsize=fs*scale, x=0.12, y=0.9)
# ax2.text(0.1, 0.91, '$\sigma\cdot{t}$-$J$', fontsize=fs*1.2, transform=ax2.transAxes)

'''

fig = plt.figure(figsize=(8, 8))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

ax1 = plt.subplot(111)
paras = (numSiteX, numSiteY, numSam, 'OO', 'false')
# ene = LoadEigVal(paras)
# wf0 = LoadEigVec(paras, 0)
# wf1 = LoadEigVec(paras, 1)
# ee = EntanglementEntropyArray()
ee = np.load('/Users/wayne/Downloads/data/ee_sigma_step-0.02_false-OO.npy')
# print('tj EE', ee)
plt.plot(x[2::inter], ee[2::inter], '-o', color='red', label='$t$-$J$')

paras = (numSiteX, numSiteY, numSam, 'OO', 'true')
# ene = LoadEigVal(paras)
# wf0 = LoadEigVec(paras, 0)
# wf1 = LoadEigVec(paras, 1)
# ee = EntanglementEntropyArray()
ee = np.load('/Users/wayne/Downloads/data/ee_sigma_step-0.02_true-OO.npy')
# print('sigma tj EE', ee)
plt.plot(x[2::inter], ee[2::inter], '->', color='blue', label='$\sigma\cdot{t}$-$J$')

plt.xlabel('$J$', fontsize=fs)
plt.ylabel('mutual spin-charge EE', fontsize=fs)
plt.legend(loc='upper left', frameon=False, fontsize=fs)
plt.setp(ax1.get_xticklabels(), fontsize=fs)
plt.setp(ax1.get_yticklabels(), fontsize=fs)
# plt.ylim([0.0, 2.2])
# plt.axvline(x=16.9, ymin=0.0, ymax=1.0, color='black', linestyle='-.')
# plt.axvline(x=28.5, ymin=0.0, ymax=1.0, color='black', linestyle='-.')

# axins1 = inset_axes(ax1,
        # width='45%',
        # height='35%',
        # loc=1,
        # bbox_to_anchor=(-0.05, -0.05, 1, 1), # (x, y, width, height of the bbox)
        # bbox_transform=ax1.transAxes,
 # )

# EE = np.load('ee-scale_sigma.npy')
# n = 6
# xx = np.linspace(10, 10+(n-1)*2, num=n)
# xxl = np.log(xx)

# p1 = np.polyfit(xxl, EE[1][1:], 1)
# p2 = np.polyfit(xxl, EE[3][1:], 1)
# p3 = np.polyfit(xxl, EE[5][1:], 1)
# p4 = np.polyfit(xxl, EE[7][1:], 1)
# print(p1[0], p2[0], p3[0], p4[0])
# plt.plot(xxl, EE[1][1:], 'o', color='red', label='$J/t=0.2$')
# plt.plot(xxl, xxl*p1[0]+p1[1], '--', color='red')
# plt.plot(xxl, EE[3][1:], '>', color='blue', label='$J/t=0.4$')
# plt.plot(xxl, xxl*p2[0]+p2[1], '--', color='blue')
# plt.plot(xxl, EE[5][1:], 'd', color='green', label='$J/t=0.6$')
# plt.plot(xxl, xxl*p3[0]+p3[1], '--', color='green')
# # plt.plot(xx, EE[7], 'x', color='cyan', label='$J/t=0.7$')
# # plt.plot(xx, xx*p4[0]+p4[1], '--', color='cyan')

# # plt.plot(xx, EE[2], 'o', color='red', linestyle='--', label='$J/t=0.2$')
# # plt.plot(xx, EE[4], '>', color='blue', linestyle='--', label='$J/t=0.4$')
# # plt.plot(xx, EE[6], 'd', color='green', linestyle='--', label='$J/t=0.6$')
# # # plt.plot(xx, EE[7], 'x', color='cyan', linestyle='--', label='$J/t=0.7$')

# ylimLow = 1.5
# ylimUp = 2.4
# plt.xlabel('$N$')
# # plt.ylabel('Mutual EE')
# plt.xlim([2.3, 3.0])
# # plt.xlim([-0.15, 0.5])
# plt.ylim([ylimLow, ylimUp])
# plt.legend(loc='lower right', frameon=False)
# plt.xticks(np.log([10, 12, 14, 16, 18, 20]), [10, 12, 14, 16, 18, 20])
# plt.yticks(np.arange(ylimLow, ylimUp, 0.2)) # Change the ticks' frequency.
# # plt.setp(axin1.get_yticklabels(), fontsize=fs)
# plt.title('$\sigma\cdot{t}$-$J$', x=0.2, y=0.8)
# # axins1.text(0.1, 0.8, '$\sigma\cdot{t}$-$J$', transform=ax1.transAxes)

# axins2 = inset_axes(ax1,
        # width='45%',
        # height='35%',
        # loc=1,
        # bbox_to_anchor=(-0.05, -0.55, 1, 1), # (x, y, width, height of the bbox)
        # bbox_transform=ax1.transAxes,
 # )

# EE2 = np.load('ee-scale.npy')
# plt.plot(xxl, EE2[2][1:], 'o', color='red', linestyle='--', label='$J/t=0.2$')
# plt.plot(xxl, EE2[4][1:], '>', color='blue', linestyle='--', label='$J/t=0.4$')
# plt.plot(xxl, EE2[6][1:], 'd', color='green', linestyle='--', label='$J/t=0.6$')
# # plt.plot(xx, EE[7], 'x', color='cyan', linestyle='--', label='$J/t=0.7$')

# ylimLow = -0.01
# ylimUp = 0.15
# plt.xlabel('$N$')
# # plt.ylabel('Mutual EE')
# plt.xlim([2.3, 3.0])
# plt.ylim([ylimLow, ylimUp])
# plt.legend(loc='best', frameon=False)
# plt.xticks(np.log([10, 12, 14, 16, 18, 20]), [10, 12, 14, 16, 18, 20])
# plt.yticks(np.arange(ylimLow, ylimUp, 0.04)) # Change the ticks' frequency.
# plt.title('$t$-$J$', x=0.2, y=0.8)
# # plt.setp(axins2.get_yticklabels(), fontsize=fs)
# # axins1.text(0.4, 0.8, '$\sigma\cdot{t}$-$J$', fontsize=fs, transform=ax1.transAxes)

# plt.annotate('', xy=(35.0, 1.2), xycoords='data',
        # xytext=(35.0, 1.3), textcoords='data',
        # arrowprops=dict(arrowstyle='->'))
# plt.annotate('', xy=(35.0, 1.05), xycoords='data',
        # xytext=(35.0, 0.95), textcoords='data',
        # arrowprops=dict(arrowstyle='->'))

# plt.annotate('', xy=(2.0, 0.7), xycoords='data',
        # xytext=(2.0, 1.2), textcoords='data',
        # arrowprops=dict(arrowstyle='<->'))

# ax1.annotate('',
        # xy=(18.0, 1.6), xycoords='data', # The critical point is at x = 28.5.
        # xytext=(1.0, 1.7), textcoords='data',
        # arrowprops=dict(arrowstyle='fancy',
            # facecolor=(0.6, 0.6, 0.6),
            # edgecolor='none',
            # connectionstyle='arc3, rad=-0.1'),
        # )

# ax1.annotate('',
        # xy=(18.0, 0.5), xycoords='data', # The critical point is at x = 28.5.
        # xytext=(1.0, 0.1), textcoords='data',
        # arrowprops=dict(arrowstyle='fancy',
            # facecolor=(0.6, 0.6, 0.6),
            # edgecolor='none',
            # connectionstyle='arc3, rad=-0.1'),
        # )
'''
image = 'entanglement_spectrum_size-%s%s.png'
imageParas = (numSiteX, numSiteY)
fig.tight_layout()
plt.savefig(image % imageParas, format='PNG')
plt.show()
