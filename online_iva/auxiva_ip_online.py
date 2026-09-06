#  Online (frame-recursive) auxiliary function based independent vector
#  analysis with iterative projection (AuxIVA-IP).
#
#  This is the inversion-based baseline that online AuxIVA-ISS
#  (auxiva_iss_online.py) is meant to replace. It is written to match
#  that module as closely as the two rules allow -- same recursive
#  weighted covariance, same auxiliary variable, same per-frame
#  iteration scheme -- so a runtime comparison tests the update rule
#  rather than implementation style:
#
#    IP  : solve (W V_k) w_k = e_k, then normalize     (needs a solve)
#    ISS : rank-1 steering update                      (inverse-free)
#
#  One asymmetry remains and is deliberate. IP carries diagonal loading;
#  ISS carries none. That is not an unfair advantage handed to ISS but a
#  real cost of the inversion-based approach: a singular solve fails
#  hard, where ISS degrades gracefully through the floor on its scalar
#  denominator. Removing the loading would not equalize the comparison,
#  it would just make IP crash on quiet input.
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
    update; it defaults to a scale-matched multiple of the identity
    derived from the input. See auxiva_iss_online for details.

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
    # Scale-matched prior, computed identically to the ISS module's.
    # See the comment there for why the input power alone is the wrong
    # scale. Change one, change both.
    #
    # The diagonal loading below is scaled to whichever covariance is
    # actually in use: an absolute constant does not track the input
    # gain and would not prevent a LinAlgError on a quiet stretch, which
    # in a streaming run is fatal. A caller-supplied V0 may sit at a
    # different scale than the input-derived prior, so it is measured
    # rather than assumed.
    if V0 is None:
        Xw = X[: min(n_frames, 50)]
        p0 = np.mean(np.abs(Xw) ** 2)                 # mean per-bin power
        e0 = np.mean(np.sum(np.abs(Xw) ** 2, axis=1))  # across-frequency energy
        if model == "laplace":
            u0 = p0 / (2.0 * np.sqrt(e0) + eps)        # phi ~ 1 / (2r)
        else:
            u0 = n_freq * p0 / (e0 + eps)              # phi ~ n_freq / r^2
        V = np.tile(
            (u0 + eps) * np.eye(n_chan, dtype=X.dtype), (n_src, n_freq, 1, 1)
        )
        reg = 1e-6 * u0
    else:
        V = V0.copy()
        reg = 1e-6 * np.mean(
            np.real(np.diagonal(V, axis1=-2, axis2=-1))
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

            # ---- pass 1: byte-identical to the ISS module's pass 1, so
            # ---- the two modules differ only in the sweep below
            Y_all = (W @ x_t[:, :, None])[..., 0]                # (n_freq, n_src)
            r_all = np.sqrt(np.sum(np.abs(Y_all) ** 2, axis=0))  # (n_src,)
            phi_all = _phi(r_all, model, n_freq, eps)            # (n_src,)

            V = alpha * V_prev + (1 - alpha) * phi_all[:, None, None, None] * xxH[None]

            # ---- pass 2: IP sweep. This is the step ISS removes: a
            # ---- linear solve per frequency bin.
            for k in range(n_src):

                # The loading belongs on V[k], which is Hermitian
                # positive semi-definite, not on the product W @ V[k],
                # which is neither. Applied distributively so the loaded
                # matrix is never materialized:
                #     W @ (V[k] + reg I) = W @ V[k] + reg W
                #
                # Not folded into pass 1: V_prev would then inherit the
                # loading each frame and it would compound to
                # reg / (1 - alpha), a 25x inflation at alpha = 0.96.
                WV = W @ V[k]
                WV += reg * W

                e_k = np.broadcast_to(eyes[:, k], (n_freq, n_chan))
                # The trailing axis is explicit: numpy 2.0 changed how a
                # right-hand side of shape (n_freq, n_chan) is
                # interpreted, and without it solve raises there while
                # working on numpy 1.x.
                w_new = np.linalg.solve(WV, e_k[..., None])[..., 0]

                # normalize: w_k <- w_k / sqrt(w_k^H V_k w_k). w_new is
                # the true (unconjugated) vector here, so this is the
                # plain Hermitian form.
                # likewise (V[k] + reg I) @ w = V[k] @ w + reg w, so the
                # normalization uses the same loaded matrix as the solve
                Vw = (V[k] @ w_new[:, :, None])[..., 0] + reg * w_new
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
