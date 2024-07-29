import numpy as np
from cholesky_decomposer import CholeskyDecomposer

# Initial matrix A and vector v for rank update
A = np.array([[4, 1], [1, 3]], dtype=float)
v = np.array([1, 1], dtype=float)
alpha = 0.1  # Scalar for the rank update

decomposer = CholeskyDecomposer()

# Compute the initial Cholesky decomposition
decomposer.decompose(A)
L = decomposer.get_L()

# Perform a rank update using the pybind11 extension
# Pass the already computed Cholesky factor L, instead of the original matrix A
decomposer.rank_update(v, alpha)
L_updated = decomposer.get_L()

# Manually update matrix A to A + alpha * v * v^T
A_updated = A + alpha * np.outer(v, v)

# Compute Cholesky decomposition of the manually updated matrix
L_updated_v2 = np.linalg.cholesky(A_updated)

# Print both results to compare
print("Cholesky factor from rank update:\n", L_updated)
print("Cholesky factor from direct decomposition of updated A:\n", L_updated_v2)

