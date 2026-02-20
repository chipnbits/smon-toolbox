"""Stochastic trace estimators: Hutchinson and Hutch++."""

import numpy as np


def hutchinson(A, num_queries, dist="rademacher"):
    """Hutchinson's stochastic trace estimator.

    Estimates tr(A) using random matrix-vector products.
    Convergence: O(1/sqrt(m)) relative error.

    Args:
        A: Square matrix or linear operator supporting @ (matmul).
        num_queries: Number of matrix-vector products (budget m).
        dist: Sketching distribution, 'rademacher' or 'gaussian'.

    Returns:
        Scalar trace estimate.
    """
    n = A.shape[0]
    if dist == "rademacher":
        Omega = np.random.randint(0, 2, size=(n, num_queries)) * 2 - 1
    elif dist == "gaussian":
        Omega = np.random.randn(n, num_queries)
    else:
        raise ValueError("dist must be 'rademacher' or 'gaussian'")

    Y = A @ Omega
    return np.sum(Omega * Y) / num_queries


def hutch_plus_plus(A, num_queries, dist="rademacher"):
    """Hutch++ trace estimator (Meyer et al., 2021).

    Deflates the top eigenspace then applies Hutchinson on the residual.
    Convergence: O(1/m) relative error for matrices with decaying spectrum.

    Args:
        A: Square matrix or linear operator supporting @ (matmul).
        num_queries: Total matrix-vector product budget (m). Split into
            sketching, low-rank trace, and residual phases.
        dist: Sketching distribution, 'rademacher' or 'gaussian'.
            Controls the sketch split ratio (1/3 for Rademacher, 1/4 for Gaussian).

    Returns:
        Scalar trace estimate.
    """
    n = A.shape[0]
    m = num_queries

    # Split size depends on distribution for optimal theoretical bounds
    if dist == "rademacher":
        s = max(2, int(m / 3))
    elif dist == "gaussian":
        s = max(2, int((m + 2) / 4))
    else:
        raise ValueError("dist must be 'rademacher' or 'gaussian'")

    # Sketching phase (cost: s queries)
    if dist == "rademacher":
        Omega = np.random.randint(0, 2, size=(n, s)) * 2 - 1
    else:
        Omega = np.random.randn(n, s)

    Y = A @ Omega
    Q, _ = np.linalg.qr(Y)

    # Exact trace of low-rank approximation (cost: s queries)
    AQ = A @ Q
    tr_low_rank = np.trace(Q.T @ AQ)

    # Residual Hutchinson (cost: m - 2s queries)
    m_rem = m - 2 * s
    if m_rem <= 0:
        return tr_low_rank

    Omega_res = np.random.randint(0, 2, size=(n, m_rem)) * 2 - 1
    A_Omega_res = A @ Omega_res
    Y_res = A_Omega_res - Q @ (Q.T @ A_Omega_res)
    tr_res = np.sum(Omega_res * Y_res) / m_rem

    return tr_low_rank + tr_res
