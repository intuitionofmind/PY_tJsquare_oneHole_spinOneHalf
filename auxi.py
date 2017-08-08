import os
import math 
import numpy as np
from numpy import linalg as la

numSiteX = 4
numSiteY = 4
numSite = numSiteX*numSiteY
numEval = 20
numSam = 5
step = 10.0
mz = 0.5  # total magnetization, a specific sector
flux = 0.0

fs = 16
ls = 12

max = int(math.pow(2, numSite-1)) # maxmium dimension of the Hilbert space for the sub-space in terms of pure spin configuration
n = 0  # counter for states in a specific magnetic sector
spinBasis = np.zeros(0, dtype=int)
for i in range(max):
    if bin(i).count('1') == int(mz+(numSite-1)/2.0):
        spinBasis = np.append(spinBasis, i)
        n += 1

holeBasis = np.zeros(numSite, dtype=int)
for i in range(numSite):
    holeBasis[i] = i

subDim = n  # dimension of the sub-space with a hole fixed
dim = numSite*n  # dimension of the Hilbert space

print(subDim, dim)

dataDir = '/Users/wayne/Downloads/data/'
eigValsFile = 'eigenvalues%s_size%s_J0.3_10.0_%s_BC_%s_sigma%s.dat'
eigVecsFile = 'eigenvectors%s_size%s_J0.3_10.0_%s_BC_%s_sigma%s.dat'
fe = os.path.join(dataDir, eigValsFile)
fv = os.path.join(dataDir, eigVecsFile)

# paras = (numSiteX, numSiteY, numSam, bounCon, sigma)
# paras = (numEval, size, numSam, bounCon, sigma)
def LoadEigVal(paras):
    n = paras[2]
    rawData = np.fromfile(fe % paras, dtype=np.float)
    print(fe % paras)
    eigVal = np.zeros((n, numEval))
    print(len(rawData))
    for i in range(n): 
        for j in range(numEval):
            eigVal[i][j] = rawData[i*numEval+j]
    return eigVal

def LoadEigVec(paras, k):
    n = paras[2]
    rawData = np.fromfile(fv % paras, dtype=np.float)
    print(len(rawData))
    eigVec = np.zeros((n, dim), dtype=complex)
    for i in range(n): 
        for j in range(dim):
            eigVec[i][j] = complex(rawData[i*(numEval*dim*2)+k*dim*2+j*2+0], rawData[i*(numEval*dim*2)+k*dim*2+j*2+1])
    print(eigVec[0])
    return eigVec

def FindState(s, lower, upper):
    if lower == upper:
        return upper
    else:
        mid = (lower+upper)//2
        if s > spinBasis[mid]:
            return FindState(s, mid+1, upper)
        else:
            return FindState(s, lower, mid)

def Flip(s, i, j):
    f = int(np.power(2, i))+int(np.power(2, j))
    return s^f

# Translation() is only valid for periodic boundary condition. Note that there is also a fermion sign arised here if the hole is not located in N-1.

def Translation(v):
    w = np.zeros(dim, dtype=complex)
    for i in range(dim):
        h = holeBasis[i // subDim]
        hh = (h+1) % numSite
        if h == numSite-1:
            w[hh*subDim+(i % subDim)] = v[i]
        else:
            s = spinBasis[i % subDim]
            b = bin(s)[2:]
            b = b.rjust(numSite, '0')
            b = b[::-1]
            bb = b[numSite-2]+b[:numSite-2]
            bb = bb[::-1]
            ss = int(bb, 2)
            l = FindState(ss, 0, subDim-1)
            w[hh*subDim+l] = np.power(-1.0, numSite)*v[i]
    return w

def TranslationX(v):
    w = np.zeros(dim, dtype=complex)
    for i in range(dim):
        h = holeBasis[i // subDim]
        hh = (h+1) % numSiteX
        s = spinBasis[i % subDim]
        b = bin(s)[2:]
        b = b.rjust(numSite-1, '0')
        b = b[::-1]
        config = b[:h]+'0'+b[h:]
        temp = config
        for j in (numSiteY):
            temp[j*numSiteX:(j+1)*numSiteX] = config[(j+1)*numSiteX-1]+config[j*numSiteX:(j+1)*numSiteX-1]
        bb = temp[:hh]+temp[hh+1:]
        bb = bb[::-1]
        ss = int(bb, 2)
        l = FindState(ss, 0, subDim-1)
        if (h % numSiteX) == numSiteX-1:
            sign = 1.0
        else:
            sign = np.power(-1.0, numSiteX)
        w[hh*subDim+l] = sign*(np.power(-1.0, numSiteX-1))*(numSiteY-1)*v[i]
 
def MarshallSign(i):
    s = spinBasis[i % subDim]
    h = holeBasis[i // subDim]
    b = bin(s)[2:]
    b = '%0*.0f' % (numSite, int(b))
    b = b[::-1]
    n = 0
    for j in range(numSite):
        if 0 == j % 2:
            if j == h:
                n += 1
            elif j < h and '0' == b[j]:
                n += 1
            elif j > h and '0' == b[j-1]:
                n += 1
    return np.power(-1, n)

# ---------------------Test begin--------------------

'''
PrintHam(1.2, True)
print(test_paras)
ene = LoadEigVal(test_paras)
wf = LoadEigVec(test_paras, 0)
wfT = LoadTransEigVec(test_paras, 0)

v = wf[slice]
print('Check energy: ', ene[slice][0], np.vdot(v, Hamiltonian(v, (slice+1)*0.1, False)))
vv = Translation(v)
for i in range(dim):
    print(vv[i]-wfT[slice][i])
'''

'''
# Test Translation() and diagonalization.

ene = LoadEigVal(test_paras)
wf0 = LoadEigVec(test_paras, 0)
wf1 = LoadEigVec(test_paras, 1)
temp = np.zeros((2, 2), dtype=complex)
if np.absolute(ene[slice][0]-ene[slice][1]) < 0.0000000001:
    wf0t = Translation(wf0[slice])
    wf1t = Translation(wf1[slice])
    temp[0][0] = np.vdot(wf0[slice], wf0t) # vdot for complex conjugate...
    temp[0][1] = np.vdot(wf0[slice], wf1t)
    temp[1][0] = np.vdot(wf1[slice], wf0t)
    temp[1][1] = np.vdot(wf1[slice], wf1t)
    w, v = la.eig(temp)
    print(slice, np.angle(w, deg=True))
    if w[0].imag > 0: # choose the eige
        choice = 0
    else:
        choice = 1
    wf = v[:, choice][0]*wf0[slice]+v[:, choice][1]*wf1[slice]  #construct eigenstates with specific momentum
    wff = Translation(wf)
    print('Test trsanlation:')
    for i in range(10):
        print(np.angle(wf[i]/wff[i], deg=True))
'''
# ----------------------Test end----------------------
