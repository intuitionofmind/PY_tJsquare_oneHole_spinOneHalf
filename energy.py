import os
import math
import numpy as np
import matplotlib.pyplot as plt
from auxi import numSam, numEval, step
# from auxi import *

dataDir = '/Users/wayne/Downloads/data/'
eigValsFile = 'eigenvalues%s_size%s_J0.3_10.0_%s_BC_%s_sigma%s.dat'
eigVecsFile = 'eigenvectors%s_size%s_J0.3_10.0_%s_BC_%s_sigma%s.dat'
fe = os.path.join(dataDir, eigValsFile)
fv = os.path.join(dataDir, eigVecsFile)

# paras = (numEval, size, numSam, bounCon, sigma)
def LoadEigVal(f):
    rawData = np.fromfile(f, dtype=np.float)
    eigVal = np.zeros((numSam, numEval))
    print(len(rawData))
    for i in range(numSam): 
        for j in range(numEval):
            eigVal[i][j] = rawData[i*numEval+j]
    return eigVal

inter=2
bounCon = 'PP'
size=36

x = np.linspace(0.0, step*numSam, num=numSam)

fig = plt.figure(figsize = (8, 6))
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

ax1 = plt.subplot(121)
paras = (numEval, size, numSam, bounCon, 'False')
ene = LoadEigVal(os.path.join(dataDir, eigValsFile) % paras)
for i in range(numSam):
    print(ene[i, :5])

plt.plot(x[::inter], ene[::inter, 0]-ene[::inter, 0], '-o', color='red', label='$0$')
plt.plot(x[::inter], ene[::inter, 1]-ene[::inter, 0], '->', color='blue', label='$1$')
plt.plot(x[::inter], ene[::inter, 2]-ene[::inter, 0], '-d', color='green', label='$2$')
# plt.plot(x[::inter], ene[::inter, 3]-ene[::inter, 0], '-d', color='green', label='$2$')
# plt.plot(x[::inter], ene[::inter, 3]-ene[::inter, 0], '-x', color='yellow', label='$3$')
# plt.plot(x[::inter], ene[::inter, 4]-ene[::inter, 0], '-h', color='black', label='$4$')
plt.xlabel('$J/t$', fontsize=fs)
plt.ylabel('$E-E_{0}~(t)$', fontsize=fs)
# plt.ylim([-0.15, 0.5])
plt.legend(loc='best', frameon=False, fontsize=fs)
#plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax1.get_yticklabels(), fontsize=fs)
ax1.set_title('(a) $t$-$J$', fontsize=fs, x=0.15, y=0.05)

ax2 = plt.subplot(122)
paras = (numEval, size, numSam, bounCon, 'True')
print(paras)
for i in range(numSam):
    print(ene[i, :5])

ene = LoadEigVal(paras)
plt.plot(x[::inter], ene[::inter, 0]-ene[::inter, 0], '-o', color='red', label='$0$')
plt.plot(x[::inter], ene[::inter, 1]-ene[::inter, 0], '->', color='blue', label='$1$')
plt.plot(x[::inter], ene[::inter, 2]-ene[::inter, 0], '-d', color='green', label='$2$')
# plt.plot(x[::inter], ene[::inter, 3]-ene[::inter, 0], '-x', color='yellow', label='$3$')
# plt.plot(x[::inter], ene[::inter, 4]-ene[::inter, 0], '-h', color='black', label='$4$')
plt.xlabel('$J/t$', fontsize=fs)
# plt.ylim([-0.2, 1.2])
plt.legend(loc='best', frameon=False, fontsize=fs)
# plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax2.get_yticklabels(), visible=False)
ax2.set_title('(b) $\sigma\cdot{t}$-$J$', fontsize=fs, x=0.2, y=0.05)

image = 'energy_size-%s%s_%s.png'
paras_image = (numSiteX, numSiteY, bounCon)
fig.tight_layout()
plt.savefig(image % paras_image, format='PNG')
plt.show()
