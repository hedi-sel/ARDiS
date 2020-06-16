import modulePython.dna as dna
from modulePython.read_mtx import *

import numpy as np
from scipy.sparse import *
import scipy.sparse.linalg as spLnal
import time
import matplotlib.pyplot as plt


plt.plot([0.1, 0.19, 0.37, 0.46, 0.55, 0.64, 0.73, 0.82, 1.0],
         [0, 0, 1.1, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0])
plt.ylabel('Arrival Time')
plt.xlabel('Thickness of the tube')

plt.show()
