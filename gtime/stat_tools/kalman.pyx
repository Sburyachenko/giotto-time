from scipy.linalg.cython_blas cimport dsymm, dgemm, dgemv, daxpy, dsyr, dcopy
from scipy.linalg.cython_lapack cimport dpotrf
from numpy cimport NPY_DOUBLE, npy_intp, PyArray_ZEROS, import_array, float64_t, ndarray
from numpy import zeros_like
from libc.math cimport log, M_PI

import_array()


cdef double llf_cp(double nu, double F):
    return -0.5 * (log(2 * M_PI * abs(F)) + nu * nu / F)


cdef void add_mat(double[::1, :] a, double[::1, :] b, int r, int c):

    cdef int i = 0
    cdef int j = 0

    for i in range(r):
        for j in range(c):
            a[i, j] = a[i, j] + b[i, j]


cdef void mat_square_c(double[::1, :] x, double[::1, :] q, int n):
    cdef int ldt = x.strides[1] // sizeof(float64_t)
    cdef double one = 1.0
    cdef double zero = 0.0
    cdef npy_intp r2shape[2]
    r2shape[0] = <npy_intp> n
    r2shape[1] = <npy_intp> n
    cdef double[::1, :] res = PyArray_ZEROS(2, r2shape, NPY_DOUBLE, 1)
    dsymm('R', 'U', &n, &n, &one, &q[0, 0], &ldt, &x[0, 0], &ldt, &zero, &res[0, 0], &ldt)
    dgemm("N", "T", &n, &n, &n, &one, &res[0, 0], &ldt, &x[0, 0], &ldt, &zero, &x[0, 0], &ldt)


def roll(double[::1] x, double[::1] a, double[::1, :] p, double[::1] eps,
            double mu, double F, double nu,
            double[::1, :] rqr, double[::1, :] K,
           int r, int n):

    cdef int lda = a.strides[0] // sizeof(float64_t)
    cdef int ldk = K.strides[1] // sizeof(float64_t)
    cdef int ldp = p.strides[1] // sizeof(float64_t)
    cdef double one = 1.0
    cdef double zero = 0.0
    cdef double F_c = F
    cdef float64_t F_inv
    cdef float64_t nf
    cdef float64_t loglikelihood = 0.0
    cdef bint converged = False
    cdef int i = 0
    cdef double[::1] a_t = zeros_like(a, order='F')

    # for i, x in enumerate(X[1:]):
    for i in range(n-1):

        if F_c == rqr[0, 0]:
            converged = True

        F_inv = -1.0 / F_c
        nf = -nu * F_inv

        if converged:
            # a = a + p[:, 0] * nu / F
            # a = np.matmul(K, a)
            daxpy(&r, &nf, &p[0, 0], &ldp, &a[0], &lda)
            dgemv('N', &r, &r, &one, &K[0, 0], &ldk, &a[0], &lda, &zero, &a_t[0], &lda)
            dcopy(&r, &a_t[0], &lda, &a[0], &lda)

        else:
            # a = a + p[:, 0] * nu / F
            # p = p - np.dot(gain, pz.T)

            daxpy(&r, &nf, &p[0, 0], &ldp, &a[0], &lda)
            dsyr('L', &r, &F_inv, &p[0, 0], &ldp, &p[0, 0], &r)

            # a = np.matmul(K, a)
            # p = mat_square(p, K) + rqr
            # F = p[0, 0]

            dgemv('N', &r, &r, &one, &K[0, 0], &ldk, &a[0], &lda, &zero, &a_t[0], &lda)
            dcopy(&r, &a_t[0], &lda, &a[0], &lda)
            mat_square_c(p, K, r)
            add_mat(p, rqr, r, r)
            F_c = p[0, 0]

        nu = x[i+1] - a[0] - mu
        eps[i+1] = nu
        ll_last = llf_cp(nu, F_c)
        loglikelihood = loglikelihood + ll_last

    return loglikelihood, eps


def llf_c(double nu, double F):
    return llf_cp(nu, F)