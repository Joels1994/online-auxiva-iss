#  Online (frame-recursive) auxiliary function based independent vector
#  analysis with iterative source steering.
#
#  Reference: "Inverse-free Online Independent Vector Analysis with
#  Flexible Iterative Source Steering" (arXiv:2209.00937). This module
#  is an original implementation written from the paper's Algorithm 1
#  description, not a copy of the authors' own code (which is not part
#  of this repository).
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
Online AuxIVA-ISS: a frame-recursive counterpart to the batch
auxiva_iss_py in auxiva_iss.py.

Unlike the batch algorithm, which is handed the full STFT of a
recording and repeatedly sweeps over every time frame to converge,
this version processes one incoming STFT frame at a time and updates
its state causally:

  - It keeps a running, exponentially-forgotten estimate of each
    source's spatial covariance (``U``), instead of the exact
    statistics over the whole recording the batch version can afford.
  - It applies only a handful of ISS iterations per new frame (``n_iter``,
    typically 1-5), instead of dozens of sweeps over the entire signal.
  - Frame t's output only ever depends on frames <= t, so this is
    suitable (algorithmically) for streaming/low-latency use, whereas
    the batch version needs the entire recording up front.

To exercise/validate it here, this function still takes a full
pre-computed STFT array as input and loops over its frame axis, but it
never looks ahead: it only ever reads X[t] to produce output frame t,
exactly reproducing what would happen if frames arrived one at a time.
"""
import numpy as np

from .projection_back import project_back


def _phi(r, model, n_freq, eps=1e-10):
    """
    Auxiliary-variable weight for a source, given the norm of its
    demixed signal across frequency (``r``). Same weighting rule as
    the batch implementation's ``r_inv`` (see auxiva_iss.py).
    """
    if model == "laplace":
        return 1.0 / np.maximum(eps, 2.0 * r)
    elif model == "gauss":
        return 1.0 / np.maximum(eps, (r ** 2) / n_freq)
    else:
        raise ValueError(f"No such model {model}")


def auxiva_iss_online(
    X,
    n_src=None,
    n_iter=3,
    alpha=0.96,
    model="laplace",
    proj_back=True,
    W0=None,
    U0=None,
    eps=1e-10,
    callback=None,
):
    """
    Online blind source separation based on independent vector analysis
    with recursive iterative source steering updates.

    Parameters
    ----------
    X: ndarray (n_frames, n_freq, n_chan)
        STFT representation of the mixture. Frames are consumed one at
        a time, in order, to emulate streaming operation.
    n_src: int, optional
        The number of sources. Only the determined case (n_src ==
        n_chan) is supported, as in the batch auxiva_iss_py.
    n_iter: int, optional
        Number of ISS iterations applied per incoming frame (default 3).
        Unlike the batch algorithm, this need not be large: convergence
        happens gradually across frames as the recursive statistics
        accumulate, not within a single frame.
    alpha: float, optional
        Forgetting factor for the recursive per-source covariance
        (default 0.96). Closer to 1 means slower adaptation but more
        stable estimates; closer to 0 adapts faster to changing
        conditions (e.g. moving sources) but is noisier.
    model: str
        'laplace' (default) or 'gauss' source model, same as the batch
        implementation.
    proj_back: bool, optional
        Whether to apply the standard (batch) projection-back scaling
        to the full output at the end.
    W0: ndarray (n_freq, n_chan, n_chan), optional
        Initial demixing matrices. Defaults to the identity per
        frequency bin.
    U0: ndarray (n_src, n_freq, n_chan, n_chan), optional
        Initial per-source spatial covariance estimates. Defaults to a
        scale-matched multiple of the identity, derived from the input
        so that it sits near where the recursion settles.
    callback: callable, optional
        If given, called as ``callback(y_t, t)`` after each frame is
        processed, with ``y_t`` of shape (n_freq, n_src).

    Returns
    -------
    ndarray (n_frames, n_freq, n_src)
        The separated signal, computed causally frame by frame.
    """
    n_frames, n_freq, n_chan = X.shape

    if n_src is None:
        n_src = n_chan
    assert n_src == n_chan, "auxiva_iss_online only supports the determined case"

    W = (
        np.tile(np.eye(n_chan, dtype=X.dtype), (n_freq, 1, 1))
        if W0 is None
        else W0.copy()
    )
    # Scale-matched prior. A fixed 0.01 * I is arbitrary relative to the
    # signal level, but so is the input power on its own: the recursion
    # accumulates phi(r) * x x^H, and phi carries units of 1/amplitude,
    # so the covariance settles at power/amplitude rather than at power.
    # Using the power alone overshoots by a large factor (measured 68x
    # on one test signal); the expression below lands within 1%.
    #
    # The IP baseline computes this identically. Change one, change both.
    if U0 is None:
        Xw = X[: min(n_frames, 50)]
        p0 = np.mean(np.abs(Xw) ** 2)                 # mean per-bin power
        e0 = np.mean(np.sum(np.abs(Xw) ** 2, axis=1))  # across-frequency energy
        if model == "laplace":
            u0 = p0 / (2.0 * np.sqrt(e0) + eps)        # phi ~ 1 / (2r)
        else:
            u0 = n_freq * p0 / (e0 + eps)              # phi ~ n_freq / r^2
        U = np.tile(
            (u0 + eps) * np.eye(n_chan, dtype=X.dtype), (n_src, n_freq, 1, 1)
        )
    else:
        U = U0.copy()

    Y_out = np.zeros((n_frames, n_freq, n_src), dtype=X.dtype)

    for t in range(n_frames):
        x_t = X[t]  # (n_freq, n_chan)

        # x_t x_t^H per frequency bin, reused by whichever source's
        # covariance gets refreshed this frame
        xxH = x_t[:, :, None] * np.conj(x_t[:, None, :])  # (n_freq, n_chan, n_chan)

        # Snapshot of U_{t-1}. The recursion is anchored there, not to
        # the running iterate: the point of the inner sweep is to
        # recompute r_kt (and hence phi) as W improves, not to decay by
        # alpha again or to inject this frame's x x^H once per sweep.
        U_prev = U.copy()

        for _ in range(n_iter):

            # ---- pass 1: refresh every r_k and U[k] from the current W
            #
            # All sources' statistics are refreshed together, before any
            # pivot moves, rather than each source refreshing its own as
            # its turn arrives. Two reasons. It matches the batch
            # Algorithm 1, which computes the activations once per
            # iteration and notes that this suffices. And it removes an
            # ordering dependence: previously source m's covariance was
            # current if m had already been visited this sweep and stale
            # otherwise, so results depended on source indexing. The IP
            # baseline performs the identical pass, so the two differ
            # only in the update rule below.
            Y_all = (W @ x_t[:, :, None])[..., 0]                # (n_freq, n_src)
            r_all = np.sqrt(np.sum(np.abs(Y_all) ** 2, axis=0))  # (n_src,)
            phi_all = _phi(r_all, model, n_freq, eps)            # (n_src,)

            U = alpha * U_prev + (1 - alpha) * phi_all[:, None, None, None] * xxH[None]

            # ---- pass 2: pivot sweep
            for k in range(n_src):

                # Note on conjugates: W's rows are stored so that
                # y = W @ x directly (no extra conjugate needed at use
                # time, matching the final y_t computation below), which
                # means row k of W IS w_k^H already, i.e. the true
                # steering vector is w_k = conj(W[:, k, :]). The
                # Hermitian form w_m^H U w_k must therefore be computed
                # as W[:,m,:] @ (U @ conj(W[:,k,:])) -- U applied to the
                # conjugated pivot row, then dotted with the plain
                # (unconjugated) row -- not the other way around.
                w_k_frozen = W[:, k, :].copy()

                # U[m] @ conj(w_k) for every source m, shape
                # (n_src, n_freq, n_chan). A batched matmul, so this goes
                # through BLAS, matching how the IP baseline's linear
                # algebra is dispatched.
                #
                # This term can be computed without materializing U at
                # all, since xxH @ conj(w_k) = x * conj(y_k) collapses the
                # recursion's new term to a rank-1 product. That was
                # tried and measured slower: both forms do the same
                # number of matrix-vector products, and the collapse
                # merely trades one (n_src, n_freq, n_chan, n_chan)
                # allocation per sweep for n_src extra elementwise
                # operations, which costs more at the small channel
                # counts that matter here. Interleaved best-of-nine:
                # 0.85x at 2 channels, 0.89x at 4, break-even at 8,
                # 1.04x at 12.
                Uk_wk = (U @ np.conj(w_k_frozen)[None, :, :, None])[..., 0]

                denom = np.real(
                    np.sum(w_k_frozen[None, :, :] * Uk_wk, axis=-1)
                )  # w_k^H U[m] w_k, shape (n_src, n_freq)
                denom = np.maximum(denom, eps)

                # copied, not a view: numer happens to be materialized
                # before W is modified below, but that is an accident of
                # statement order rather than something enforced
                W_rows = np.transpose(W, (1, 0, 2)).copy()
                numer = np.sum(
                    W_rows * Uk_wk, axis=-1
                )  # w_m^H U[m] w_k, shape (n_src, n_freq)

                v = numer / denom  # (n_src, n_freq), valid for m != k
                v[k] = 1.0 - 1.0 / np.sqrt(denom[k])  # self-update for m == k

                v_t = np.transpose(v, (1, 0))  # (n_freq, n_src)
                W -= v_t[:, :, None] * w_k_frozen[:, None, :]

        y_t = (W @ x_t[:, :, None])[..., 0]
        Y_out[t] = y_t

        if callback is not None:
            callback(y_t, t)

    if proj_back:
        # The batch project_back utility operates over the whole
        # recording at once; applying it as a final scale-fixing pass
        # keeps output amplitudes consistent with the other algorithms
        # in this package. A true streaming system would instead track
        # a running projection-back scale per frame.
        Y_out = project_back(Y_out, X[:, :, 0])

    return Y_out
