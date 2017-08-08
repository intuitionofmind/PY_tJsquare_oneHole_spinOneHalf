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

dataDir = '/Users/wayne/Downloads/data/'

meeFile = 'mutualEE%s_len%s_J%s_%s_sigma%s.npy'

l = 4
J = 40.3

num = 20

f = 'chargeDensity_num%s_J%s_%s_sigma%s.npy'
CDParas = (num, J, 'OO', 'False')
cd = np.load(os.path.join(dataDir, f) % CDParas)

xt = [0, 1, 2, 3]
yt = [0, 1, 2, 3]
fig = plt.figure(figsize=(12, 10))
# fig, axes = plt.subplots(nrows=3, ncols=3)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# plt.suptitle('$\sigma\cdot{t}$-$J$, $J=0.3$, OBC', fontsize=fs)
fig.suptitle('${t}$-$J$ model, $J=0.3$, PBC', fontsize=fs)

import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(3, 3)

i = 0
for s in gs:
    ax = fig.add_subplot(s)
    print(cd[i])
    im = ax.imshow(cd[i], vmin=0.0, vmax=0.16)
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    i += 1

gs.tight_layout(fig, rect=[0.0, 0., 0.8, 0.95]) # [left, bottom, right, top] that the whole subplot area will fit into. Default is [0, 0, 1, 1] 
cax = fig.add_axes([0.9, 0.1, 0.04, 0.8])
fig.colorbar(im, cax=cax, use_gridspec=True)

image = 'chargeDensity%s_J%s_boundary%s_sigma%s.pdf'
plt.savefig(image % CDParas, format='pdf')

# fig.tight_layout()
# plt.show()

