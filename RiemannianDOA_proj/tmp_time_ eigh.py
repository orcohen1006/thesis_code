import numpy as np
from time import time

M = 12
A = np.arange(1,M**2+1).reshape((M,M))
A = A@A.T

num_trys = int(1e4)
t0 = time()
for i in range(num_trys):
    tmp1,tmp2 = np.linalg.eigh(A)
    # tmp1,tmp2,tmp3 = np.linalg.svd(A)

dt = time() - t0
print(f"avg time = {dt/num_trys *1e3} [millisec]")