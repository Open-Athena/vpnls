# cython: boundscheck=False, wraparound=False, cdivision=True
"""Pure C-level Cython grid solver for VPNLS."""

from libc.math cimport exp, fabs, INFINITY
import numpy as np


# ── 3x3 Cramer's rule ───────────────────────────────────────────────────────

cdef inline double det3(double* A) noexcept nogil:
    return (A[0] * (A[4] * A[8] - A[5] * A[7])
          - A[1] * (A[3] * A[8] - A[5] * A[6])
          + A[2] * (A[3] * A[7] - A[4] * A[6]))


cdef inline void solve3x3(double* A, double* b, double* x) noexcept nogil:
    cdef double d = det3(A)
    cdef double M[9]
    cdef int i, col
    if d == 0.0:
        x[0] = 0.0; x[1] = 0.0; x[2] = 0.0
        return
    for col in range(3):
        for i in range(9):
            M[i] = A[i]
        M[col] = b[0]; M[3 + col] = b[1]; M[6 + col] = b[2]
        x[col] = det3(M) / d


# ── 2x2 Cramer's rule ───────────────────────────────────────────────────────

cdef inline void solve2x2(double a, double b, double c, double d,
                           double e, double f,
                           double* x0, double* x1) noexcept nogil:
    cdef double det = a * d - b * c
    if det == 0.0:
        x0[0] = 0.0; x1[0] = 0.0
        return
    x0[0] = (e * d - b * f) / det
    x1[0] = (a * f - e * c) / det


# ── NNLS inner solve ────────────────────────────────────────────────────────

cdef inline void nnls_solve(
    int n,
    double* col0, double* col1, double* col2, double* y,
    double* w,
    double* params,
    double* rss_out,
    int* mask_out,
) noexcept nogil:
    cdef double* cols[3]
    cols[0] = col0; cols[1] = col1; cols[2] = col2

    cdef int active[3]
    active[0] = 1; active[1] = 1; active[2] = 1

    cdef int iteration, n_active, j, k, i, any_neg
    cdef int free_idx[3]
    cdef double sub_A[9], sub_b[3], sub_x[3]
    cdef double wi, pred, r, rss

    for iteration in range(4):
        n_active = 0
        for j in range(3):
            if active[j]:
                free_idx[n_active] = j
                n_active += 1

        if n_active == 0:
            params[0] = 0.0; params[1] = 0.0; params[2] = 0.0
            break

        # Build normal equations for active variables
        for j in range(n_active * n_active):
            sub_A[j] = 0.0
        for j in range(n_active):
            sub_b[j] = 0.0

        for i in range(n):
            wi = 1.0 if w == NULL else w[i]
            for j in range(n_active):
                sub_b[j] += wi * cols[free_idx[j]][i] * y[i]
                for k in range(n_active):
                    sub_A[j * n_active + k] += wi * cols[free_idx[j]][i] * cols[free_idx[k]][i]

        if n_active == 3:
            solve3x3(sub_A, sub_b, sub_x)
        elif n_active == 2:
            solve2x2(sub_A[0], sub_A[1], sub_A[2], sub_A[3],
                     sub_b[0], sub_b[1], &sub_x[0], &sub_x[1])
        else:
            sub_x[0] = sub_b[0] / sub_A[0] if sub_A[0] != 0.0 else 0.0

        # Map back
        params[0] = 0.0; params[1] = 0.0; params[2] = 0.0
        for j in range(n_active):
            params[free_idx[j]] = sub_x[j]

        # Check for negatives
        any_neg = 0
        for j in range(3):
            if active[j] and params[j] < 0.0:
                active[j] = 0
                params[j] = 0.0
                any_neg = 1
        if not any_neg:
            break

    # RSS
    rss = 0.0
    for i in range(n):
        pred = params[0] * col0[i] + params[1] * col1[i] + params[2] * col2[i]
        r = y[i] - pred
        rss += r * r
    rss_out[0] = rss

    mask_out[0] = 0
    for j in range(3):
        if not active[j]:
            mask_out[0] |= (1 << j)


# ── Grid search (Python-callable) ───────────────────────────────────────────

def grid_search(
    double[:] log_N, double[:] log_D, double[:] L,
    double alpha_lo, double alpha_hi,
    double beta_lo, double beta_hi,
    double resolution,
    int loss_type,
    double huber_delta,
    int max_irls_iter,
):
    cdef int n = log_N.shape[0]
    cdef int n_alpha = <int>((alpha_hi - alpha_lo) / resolution) + 1
    cdef int n_beta = <int>((beta_hi - beta_lo) / resolution) + 1

    # Work arrays (allocated once in Python, used via memoryviews)
    cdef double[:] col0 = np.ones(n, dtype=np.float64)
    cdef double[:] col1 = np.empty(n, dtype=np.float64)
    cdef double[:] col2 = np.empty(n, dtype=np.float64)
    cdef double[:] wt = np.empty(n, dtype=np.float64)

    cdef double best_obj = INFINITY
    cdef double best_E = 0.0, best_A = 0.0, best_B = 0.0
    cdef double best_alpha = alpha_lo, best_beta = beta_lo
    cdef double best_rss = INFINITY
    cdef int best_mask = 0, best_ai = 0, best_bi = 0

    cdef double alpha, beta, obj, rss
    cdef double params[3]
    cdef int mask, ai, bi, i, it
    cdef double pred, r, ar

    with nogil:
        for ai in range(n_alpha):
            alpha = alpha_lo + ai * resolution

            for i in range(n):
                col1[i] = exp(-alpha * log_N[i])

            for bi in range(n_beta):
                beta = beta_lo + bi * resolution

                for i in range(n):
                    col2[i] = exp(-beta * log_D[i])

                if loss_type == 0:
                    # MSE path: single NNLS solve
                    nnls_solve(n, &col0[0], &col1[0], &col2[0], &L[0],
                               NULL, params, &rss, &mask)
                    obj = rss / n
                else:
                    # Huber path: IRLS
                    nnls_solve(n, &col0[0], &col1[0], &col2[0], &L[0],
                               NULL, params, &rss, &mask)

                    for it in range(max_irls_iter):
                        for i in range(n):
                            pred = params[0] * col0[i] + params[1] * col1[i] + params[2] * col2[i]
                            r = L[i] - pred
                            ar = fabs(r)
                            wt[i] = 1.0 if ar <= huber_delta else huber_delta / ar

                        nnls_solve(n, &col0[0], &col1[0], &col2[0], &L[0],
                                   &wt[0], params, &rss, &mask)

                    # Huber objective + RSS
                    obj = 0.0
                    rss = 0.0
                    for i in range(n):
                        pred = params[0] * col0[i] + params[1] * col1[i] + params[2] * col2[i]
                        r = L[i] - pred
                        rss += r * r
                        ar = fabs(r)
                        if ar <= huber_delta:
                            obj = obj + 0.5 * r * r
                        else:
                            obj = obj + huber_delta * (ar - 0.5 * huber_delta)
                    obj = obj / n

                if obj < best_obj:
                    best_obj = obj
                    best_E = params[0]
                    best_A = params[1]
                    best_B = params[2]
                    best_alpha = alpha
                    best_beta = beta
                    best_rss = rss
                    best_mask = mask
                    best_ai = ai
                    best_bi = bi

    return (best_E, best_A, best_B, best_alpha, best_beta,
            best_obj, best_rss, best_mask,
            best_ai, best_bi, n_alpha, n_beta)
