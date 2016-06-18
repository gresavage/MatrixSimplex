__author__="Tom"

"""
PRIMAL
maximize z = 2x1 - x2 + x3
subject to
3x1 + x2 + x3 <= 60
x1 - x2 + 2x3 <= 10
x1 + x2 - x3 <= 20

DUAL
minimize w = 60y1 + 10y2 + 20y3
positive objective
subject to
3y1 + y2 + y3 >= 2
y1 - y2 + y3 >= -1
y1 + 2y2 - y3 >= 1
"""

import numpy as np
import numpy.linalg as la

"PRIMAL"
method = 'max'
N = np.array([[3., 1., 1.], [1., -1., 2.], [1., 1., -1.]])
c = np.array([[2.], [-1.], [1.], [0.], [0.], [0.]])
B = np.eye(3)
cN = c[:3]
cB = c[3:]
b = np.array([[60.], [10.], [20.]])
var = ['x1', 'x2', 'x3', 's1', 's2', 's3']
BV = var[3:]
NBV = var[:3]

"DUAL"
# method = 'min'
# N = np.array([[3., 1., 1.], [1., -1., 2.], [1., 1., -1.]]).transpose()
# c = np.array([[60.], [10.], [20.], [0.], [0.], [0.]])
# B = -np.eye(3)
# cN = c[:3]
# cB = c[3:]
# b = np.array([[2], [-1.], [1.]])
# var = ['y1', 'y2', 'y3', 'e1', 'e2', 'e3']
# BV = var[3:]
# NBV = var[:3]
print "Non Basic Variables"
print NBV
print N
print cN
print "Basic Variables"
print BV
print B
print cB
print "RHS"
print b
print
Binv = la.inv(B)
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#-------------------------------------DUAL SIMPLEX---------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
"Dual Simplex"
"Make e1, e2 positive -> mult row 1, 3 by -1"
print "Beginning Dual Simplex"
bindex = np.where(B==-1.)
for inx in list(bindex[0]):
    "Multiply negative basis rows by -1 to make all basis entries positive"
    B[inx, :] *= -1.
    N[inx, :] *= -1.
    b[inx] *= -1.
dual = 0
while True:
    print "Dual Iteration %r" %dual
    dual += 1
    "Find most negative value index on RHS (RHS MUST be positive)"
    Ntab = Binv.dot(N)
    btab = Binv.dot(b)
    Bindex = np.argmin(btab)
    if btab[Bindex] >= 0:
        # If no RHS value is negative then continue with normal simplex
        print "No negative RHS values"
        break
    # Get indeces of negative entries in pivot row
    Nindex = np.where(Ntab[Bindex, :] < 0)
    if len(Nindex[0]) > 1:
        # Get the index of the most negative value in row Bindex
        Nindex = np.argmin(Ntab[Bindex, :])
    else:
        # Get the variable with the lowest index
        Nindex = np.min(Nindex[0])
    Bswap = np.copy(B[:, Bindex])
    Nswap = np.copy(N[:, Nindex])
    cBswap = np.copy(cB[Bindex])
    cNswap = np.copy(cN[Nindex])
    BVswap = BV[Bindex]
    NBVswap = NBV[Nindex]
    B[:, Bindex] = Nswap
    N[:, Nindex] = Bswap
    cB[Bindex] = cNswap
    cN[Nindex] = cBswap
    BV[Bindex] = NBVswap
    NBV[Nindex] = BVswap

    Binv = la.inv(B)
print "Non Basic Variables"
print NBV
print Binv.dot(N)
print cN.transpose()-cB.transpose().dot(Binv.dot(N))
print "Basic Variables"
print BV
print Binv.dot(B)
print cB
print "RHS"
print Binv.dot(b)
print "z"
print cB.transpose().dot(Binv.dot(b))
print "-"*60
print
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#------------------------------------------SIMPLEX---------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
print "Continuing with regular simplex"
"minimize -> pos obj tableau (neg cost)"
outer, inner = 0, 0
if method == 'max':
    arg = lambda x: np.argmax(x)
    cNcheck = lambda x, y: y[x] <= 0
elif method == 'min':
    arg = lambda x: np.argmin(x)
    cNcheck = lambda x, y: y[x] >= 0
while True:
    btab = Binv.dot(b)
    Ntab = Binv.dot(N)
    cNtab = np.transpose(cN.transpose()-cB.transpose().dot(Ntab))

    Nindex = arg(cNtab)
    z = cB.transpose().dot(btab)
    if cNcheck(Nindex, cNtab):
        print "Optimal Solution found"
        break
    r = np.array([btab[i] / Ntab[i, Nindex] for i in range(len(btab))])
    Bindex = np.argmin(r)
    _r = list(r)
    flag = False
    while True:
        if _r[Bindex] == np.inf or _r[Bindex] == -np.inf or _r[Bindex] <= 0:
            _r.pop(Bindex)
            try:
                Bindex = np.argmin(_r)
            except ValueError:
                print "No attractive variables"
                flag = True
                break
        else:
            break
    if flag:
        print "Solution is not optimal"
        break
    r = list(r)
    Bindex = r.index(_r[Bindex])

    Nswap = np.copy(N[:, Nindex])
    Bswap = np.copy(B[:, Bindex])

    cBswap = np.copy(cB[Bindex])
    cNswap = np.copy(cN[Nindex])
    NBVswap = NBV[Nindex]
    BVswap = BV[Bindex]

    B[:, Bindex] = Nswap
    N[:, Nindex] = Bswap
    cB[Bindex] = cNswap
    cN[Nindex] = cBswap
    NBV[Nindex] = BVswap
    BV[Bindex] = NBVswap
    Binv = la.inv(B)
    outer += 1
btab = la.inv(B).dot(b)
Ntab = la.inv(B).dot(N)
cNtab = np.transpose(cN.transpose()-cB.transpose().dot(la.inv(B).dot(N)))
print "Basis:"
print BV
print btab.transpose()
print
print "Non-Basic Variables"
print NBV
print Ntab
print "Shadow Prices"
print NBV
print cNtab.transpose()
print
print "Objective"
print cB.transpose().dot(la.inv(B).dot(b))