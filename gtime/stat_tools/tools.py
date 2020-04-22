import numpy as np

def mat_square(X, M):
    return np.linalg.multi_dot([M, X, np.transpose(M)])

def loglikelihood_ns(nu, F):
    return -0.5 * (np.log(2 * np.pi * np.abs(F)) + mat_square(np.linalg.inv(F), nu))

def arma_polynomial_roots(x, len_p):
    phi = x[2:2 + len_p]
    theta = x[2 + len_p:]
    phi_roots = np.abs(np.roots(np.r_[-phi[::-1], 1.0]))
    theta_roots = np.abs(np.roots(np.r_[theta[::-1], 1.0]))
    return np.r_[2.0, 2.0, phi_roots, theta_roots] # TODO refactor 2.0
