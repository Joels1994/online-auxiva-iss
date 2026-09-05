#  Batch (offline) AuxIVA-ISS and AuxIVA-IP, for comparison against the
#  online implementations in this package.
#
#  Both functions are taken, with minor adaptation, from piva by Robin
#  Scheibler (https://github.com/fakufaku/piva), which is GPL-3.0. They
#  are included here so the batch and online formulations can be
#  benchmarked side by side without needing a piva checkout or its
#  compiled extension.
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
The batch counterparts of auxiva_iss_online and auxiva_ip_online.

The contrast with the online versions is the point of this module:

  * Batch is handed the whole recording and sweeps over every frame
    n_iter times. Nothing is causal.
  * Batch ISS never forms a covariance matrix. It works directly on the
    already-separated outputs and contracts over time, which is what
    makes it cheaper than batch IP by roughly a factor of the channel
    count. The online formulation cannot do this, because a single
    incoming frame gives nothing to sum over, so it must carry M-by-M
    covariances instead and loses that advantage.
"""
import numpy as np

from .projection_back import project_back


def tensor_H(X):
    """Conjugate transpose of the last two axes."""
    return np.conj(X).swapaxes(-2, -1)


def auxiva_iss_batch(
    X, n_src=None, n_iter=20, proj_back=True, model="laplace", eps=1e-10
):
    """
    Batch AuxIVA with iterative source steering.

    Parameters mirror auxiva_iss_online, except that n_iter counts
    sweeps over the entire recording rather than per-frame iterations.
    """
    n_frames, n_freq, n_chan = X.shape
    if n_src is None:
        n_src = X.shape[2]
    assert n_chan == n_src, "auxiva_iss_batch only supports the determined case"

    r_inv = np.zeros((n_src, n_frames))
    v = np.zeros((n_freq, n_src), dtype=X.dtype)

    X_original = X
    X = X.transpose([1, 2, 0]).copy()   # frequencies first
    Y = X.copy()

    for _ in range(n_iter):

        if model == "laplace":
            r_inv[:, :] = 1.0 / np.maximum(eps, 2.0 * np.linalg.norm(Y, axis=0))
        elif model == "gauss":
            r_inv[:, :] = 1.0 / np.maximum(
                eps, (np.linalg.norm(Y, axis=0) ** 2) / n_freq
            )
        else:
            raise ValueError(f"No such model {model}")

        for s in range(n_src):
            # Contractions over TIME, no M-by-M matrix anywhere: this is
            # what makes batch ISS cheaper than batch IP in the channel
            # count.
            #
            # Written as a fused einsum rather than the more obvious
            # (Y * r_inv) @ conj(Y[:, s]). That form materializes a full
            # (n_freq, n_src, n_frames) temporary on every pivot, about
            # 10 MB at 12 channels, and the resulting memory traffic
            # dominates at larger channel counts: it measured 2.5x
            # slower and was enough to hide the algorithm's advantage
            # entirely.
            v_num = np.einsum(
                "fmt,mt,ft->fm", Y, r_inv, np.conj(Y[:, s, :]), optimize=True
            )
            v_denom = r_inv[None, :, :] @ np.abs(Y[:, s, :, None]) ** 2

            v[:, :] = v_num / v_denom[:, :, 0]
            v[:, s] -= 1 / np.sqrt(v_denom[:, s, 0])

            Y[:, :, :] -= v[:, :, None] * Y[:, s, None, :]

    Y = Y.transpose([2, 0, 1]).copy()

    if proj_back:
        Y = project_back(Y, X_original[:, :, 0])

    return Y


def auxiva_ip_batch(
    X, n_src=None, n_iter=20, proj_back=True, model="laplace", eps=1e-10
):
    """
    Batch AuxIVA with iterative projection, the inversion-based
    baseline. Same interface as auxiva_iss_batch.
    """
    n_frames, n_freq, n_chan = X.shape
    if n_src is None:
        n_src = n_chan
    assert n_chan == n_src, "auxiva_ip_batch only supports the determined case"

    Y = np.zeros((n_freq, n_src, n_frames), dtype=X.dtype)
    X_original = X
    X = X.transpose([1, 2, 0]).copy()

    eyes = np.eye(n_chan, dtype=X.dtype)[None, :, :]
    W = np.zeros((n_freq, n_chan, n_chan), dtype=X.dtype)
    W[:, :, :] = eyes

    r_inv = np.zeros((n_src, n_frames))
    Y[:, :, :] = X[:, :n_src, :]

    for _ in range(n_iter):

        if model == "laplace":
            r_inv[:, :] = 1.0 / np.maximum(eps, 2.0 * np.linalg.norm(Y, axis=0))
        elif model == "gauss":
            r_inv[:, :] = 1.0 / np.maximum(
                eps, (np.linalg.norm(Y, axis=0) ** 2) / n_freq
            )
        else:
            raise ValueError(f"No such model {model}")

        for s in range(n_src):
            # an M-by-M weighted covariance, then a solve, per source
            V = (X * r_inv[None, s, None, :]) @ tensor_H(X) / n_frames

            WV = W @ V
            W[:, s, :] = np.conj(np.linalg.solve(WV, eyes[:, :, s]))

            denom = np.real(
                W[:, None, s, :] @ V[:, :, :] @ np.conj(W[:, s, :, None])
            )
            W[:, s, :] /= np.sqrt(np.maximum(eps, denom[:, :, 0]))

        Y[:, :, :] = W[:, :n_src, :] @ X

    Y = Y.transpose([2, 0, 1]).copy()

    if proj_back:
        Y = project_back(Y, X_original[:, :, 0])

    return Y
