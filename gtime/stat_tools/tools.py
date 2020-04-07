import numpy as np

def mat_square(X, M):
    return np.linalg.multi_dot([M, X, np.transpose(M)])