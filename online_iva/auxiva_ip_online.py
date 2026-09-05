#  Online (frame-recursive) auxiliary function based independent vector
#  analysis with iterative projection (AuxIVA-IP).
#
#  This is the inversion-based baseline that online AuxIVA-ISS
#  (auxiva_iss_online.py) is meant to replace. It is written to be
#  structurally identical to that module -- same recursive weighted
#  covariance, same auxiliary variable, same per-frame iteration
#  scheme -- so that the only thing differing between the two is the
#  demixing-vector update rule itself:
#
#    IP  : solve (W V_k) w_k = e_k, then normalize     (needs a solve)
#    ISS : rank-1 steering update                      (inverse-free)
#
#  That makes a runtime comparison between them a fair test of the
#  update rule rather than of implementation style.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

"""
Online AuxIVA-IP: frame-recursive counterpart to the batch overiva_py
in auxiva.py, and the baseline for online AuxIVA-ISS.

The batch algorithm builds each source's weighted covariance from every
frame of the recording at once and sweeps over the whole signal many
times. This version instead keeps that covariance as a running,
exponentially-forgotten estimate and applies a few iterations per
incoming frame, so frame t's output depends only on frames <= t.
"""
import numpy as np

from .projection_back import project_back
from .auxiva_iss_online import _phi


def auxiva_ip_online(
    X,
    n_src=None,
    n_iter=3,
    alpha=0.96,
    model="laplace",
    proj_back=True,
    W0=None,
    V0=None,
    eps=1e-10,
    callback=None,
):
    """
    Online blind source separation based on independent vector analysis
    with recursive iterative projection updates.

    Parameters are the same as auxiva_iss_online, except that ``V0``
    initializes the per-source weighted covariance used by the IP
    update. See auxiva_iss_online for details.

    Returns
    -------
    ndarray (n_frames, n_freq, n_src)
        The separated signal, computed causally frame by frame.
    """
    n_frames, n_freq, n_chan = X.shape

    if n_src is None:
        n_src = n_chan
    assert n_src == n_chan, "auxiva_ip_online only supports the determined case"

    W = (
        np.tile(np.eye(n_chan, dtype=X.dtype), (n_freq, 1, 1))
        if W0 is None
        else W0.copy()
    )
    V = (
        np.tile(0.01 * np.eye(n_chan, dtype=X.dtype), (n_src, n_freq, 1, 1))
        if V0 is None
        else V0.copy()
    )

    Y_out = np.zeros((n_frames, n_freq, n_src), dtype=X.dtype)

    eyes = np.eye(n_chan, dtype=X.dtype)

    for t in range(n_frames):
        x_t = X[t]  # (n_freq, n_chan)

        xxH = x_t[:, :, None] * np.conj(x_t[:, None, :])  # (n_freq, n_chan, n_chan)

        # Snapshot of V_{t-1}: the recursion is anchored to the previous
        # frame, not to the running iterate. See the equivalent comment
        # in auxiva_iss_online.py.
        V_prev = V.copy()

        for _ in range(n_iter):
            for k in range(n_src):

                # --- auxiliary variable and recursive covariance for
                # --- source k, identical to the ISS version
                w_k = W[:, k, :]  # stored row k, i.e. w_k^H
                y_k = np.sum(w_k * x_t, axis=-1)
                r_k = np.sqrt(np.sum(np.abs(y_k) ** 2))
                phi_k = _phi(r_k, model, n_freq, eps)

                V[k] = alpha * V_prev[k] + (1 - alpha) * phi_k * xxH

                # --- iterative projection update. This is the step
                # --- ISS removes: a linear solve per frequency bin.
                WV = W @ V[k]  # (n_freq, n_chan, n_chan), row m is w_m^H V_k

                e_k = np.broadcast_to(eyes[:, k], (n_freq, n_chan))
                w_new = np.linalg.solve(WV, e_k)  # (n_freq, n_chan), true steering vec

                # normalize: w_k <- w_k / sqrt(w_k^H V_k w_k). w_new is
                # the true (unconjugated) vector here, so this is the
                # plain Hermitian form.
                Vw = (V[k] @ w_new[:, :, None])[..., 0]
                denom = np.real(np.sum(np.conj(w_new) * Vw, axis=-1))
                denom = np.maximum(denom, eps)
                w_new = w_new / np.sqrt(denom)[:, None]

                # store back as a row, i.e. as w_k^H
                W[:, k, :] = np.conj(w_new)

        y_t = (W @ x_t[:, :, None])[..., 0]
        Y_out[t] = y_t

        if callback is not None:
            callback(y_t, t)

    if proj_back:
        Y_out = project_back(Y_out, X[:, :, 0])

    return Y_out
