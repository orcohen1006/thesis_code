import numpy as np
from scipy.linalg import cholesky, solve_triangular
import time
np.random.seed(42)  # For reproducibility

M, D = 12, 360
A = np.random.randn(M, D) + 1j * np.random.randn(M, D)
Q = A @ A.conj().T + M * np.eye(M)  # Make Q Hermitian PD

NUM_ITERS = 1000
# Direct solve
t0 = time.time()
for _ in range(NUM_ITERS):
    tmp = np.linalg.solve(Q, A)
    p_direct = np.sum(A.conj() * tmp, axis=0).real
t1 = time.time()

# Cholesky + triangular solve
t2 = time.time()
for _ in range(NUM_ITERS):
    L = cholesky(Q, lower=True)
    Y = solve_triangular(L, A, lower=True)
    p_chol = np.sum(np.abs(Y)**2, axis=0)
t3 = time.time()

# Naive with for loop
t4 = time.time()
p_naive = np.zeros(D)
for _ in range(NUM_ITERS):
    invQ = np.linalg.inv(Q)
    for d in range(D):
        p_naive[d] = np.sum(A[:, d].conj() * invQ @ A[:, d]).real

t5 = time.time()

print(f"Direct solve time:   {t1 - t0:.4f} sec")
print(f"Cholesky solve time: {t3 - t2:.4f} sec")
print(f"Naive solve time:    {t5 - t4:.4f} sec")
print(f"Max diff 1:            {np.max(np.abs(p_direct - p_chol)):.4e}")
print(f"Max diff 2:            {np.max(np.abs(p_direct - p_naive)):.4e}")
