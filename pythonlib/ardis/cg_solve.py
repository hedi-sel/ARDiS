import numpy as np
import random
from scipy.sparse import *
import scipy.sparse.linalg as spLnal
import time

prepData = 0
matMult = 0
vectSum = 0
vectDot = 0

def PrintProf():
    print("Profiling diffusion")
    print("prepData\n", prepData)
    print("matMult\n", matMult)
    print("vectSum\n", vectSum)
    print("vectDot\n", vectDot)
    
def CGNaiveSolve(M, b, x, epsilon=1e-1):
    global prepData
    global matMult 
    global vectSum 
    global vectDot 
    time3 = time.time()
    
    r = b-M.dot(x)
    p = r.copy()
    diff = abs(r.dot(p))
    diff0 = diff
    if (diff <= 0):
        return 0
    nIter = 0
    
    prepData += time.time() - time3

    while diff >= epsilon*epsilon*diff0 and nIter < 10000:
    # last_nIter
        time3 = time.time()

        q = M.dot(p)
        matMult += time.time() - time3
        time3 = time.time()

        value = q.dot(p)
        vectDot += time.time() - time3
        time3 = time.time()

        if value == 0:
            return nIter
        alpha = diff / value

        x += alpha * p
        r -= alpha * q
        vectSum += time.time() - time3
        time3 = time.time()

        diffnew = r.dot(r)
        beta = diffnew / diff
        diff = diffnew
        if diff == 0:
            return nIter
        vectDot += time.time() - time3
        time3 = time.time()

        p = r + beta * p
        nIter += 1
        vectSum += time.time() - time3
        time3 = time.time()

    if nIter == 10000:
        print("Error didnt converge")
    # print("niter: ", nIter, " : ", diff)
    return nIter
