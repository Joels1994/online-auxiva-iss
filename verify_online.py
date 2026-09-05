"""
Independent verification of the vectorized online implementations.

Reimplements both algorithms with fully explicit scalar loops, written
straight from the Algorithm 1 pseudocode with no shared code, then
compares against the vectorized versions in online_iva/.
"""
import numpy as np

from online_iva import auxiva_iss_online, auxiva_ip_online


def phi_of(r, model, n_freq, eps=1e-10):
    if model == "laplace":
        return 1.0 / max(eps, 2.0 * r)
    return 1.0 / max(eps, (r ** 2) / n_freq)


def iss_bruteforce(X, n_iter, alpha, model="laplace", eps=1e-10):
    n_frames, n_freq, n_chan = X.shape
    K = n_chan
    W = np.zeros((n_freq, n_chan, n_chan), dtype=complex)
    U = np.zeros((K, n_freq, n_chan, n_chan), dtype=complex)
    for f in range(n_freq):
        W[f] = np.eye(n_chan)
        for k in range(K):
            U[k, f] = 0.01 * np.eye(n_chan)

    Y = np.zeros((n_frames, n_freq, K), dtype=complex)

    for t in range(n_frames):
        x = X[t]
        U_prev = U.copy()

        for _ in range(n_iter):
            for k in range(K):

                # r_kt = sqrt(sum_f |w_k^H x_f|^2)
                acc = 0.0
                for f in range(n_freq):
                    yk = 0j
                    for i in range(n_chan):
                        yk += W[f, k, i] * x[f, i]
                    acc += abs(yk) ** 2
                phi = phi_of(np.sqrt(acc), model, n_freq, eps)

                # U_k <- alpha U_k(t-1) + (1-alpha) phi x x^H
                for f in range(n_freq):
                    for i in range(n_chan):
                        for j in range(n_chan):
                            U[k, f, i, j] = (
                                alpha * U_prev[k, f, i, j]
                                + (1 - alpha) * phi * x[f, i] * np.conj(x[f, j])
                            )

                a_k = W[:, k, :].copy()  # stored row k == w_k^H
                v = np.zeros((K, n_freq), dtype=complex)

                for f in range(n_freq):
                    for m in range(K):
                        # U_m @ w_k, with true steering vector w_k = conj(a_k)
                        Uw = np.zeros(n_chan, dtype=complex)
                        for i in range(n_chan):
                            s = 0j
                            for j in range(n_chan):
                                s += U[m, f, i, j] * np.conj(a_k[f, j])
                            Uw[i] = s

                        den = 0j
                        for i in range(n_chan):
                            den += a_k[f, i] * Uw[i]
                        den = max(den.real, eps)

                        if m != k:
                            num = 0j
                            for i in range(n_chan):
                                num += W[f, m, i] * Uw[i]
                            v[m, f] = num / den
                        else:
                            v[m, f] = 1.0 - 1.0 / np.sqrt(den)

                # W <- W - v_k w_k^H
                for f in range(n_freq):
                    for m in range(K):
                        for i in range(n_chan):
                            W[f, m, i] -= v[m, f] * a_k[f, i]

        for f in range(n_freq):
            for i in range(K):
                s = 0j
                for j in range(n_chan):
                    s += W[f, i, j] * x[f, j]
                Y[t, f, i] = s

    return Y


def ip_bruteforce(X, n_iter, alpha, model="laplace", eps=1e-10):
    n_frames, n_freq, n_chan = X.shape
    K = n_chan
    W = np.zeros((n_freq, n_chan, n_chan), dtype=complex)
    V = np.zeros((K, n_freq, n_chan, n_chan), dtype=complex)
    for f in range(n_freq):
        W[f] = np.eye(n_chan)
        for k in range(K):
            V[k, f] = 0.01 * np.eye(n_chan)

    Y = np.zeros((n_frames, n_freq, K), dtype=complex)

    for t in range(n_frames):
        x = X[t]
        V_prev = V.copy()

        for _ in range(n_iter):
            for k in range(K):

                acc = 0.0
                for f in range(n_freq):
                    yk = 0j
                    for i in range(n_chan):
                        yk += W[f, k, i] * x[f, i]
                    acc += abs(yk) ** 2
                phi = phi_of(np.sqrt(acc), model, n_freq, eps)

                for f in range(n_freq):
                    for i in range(n_chan):
                        for j in range(n_chan):
                            V[k, f, i, j] = (
                                alpha * V_prev[k, f, i, j]
                                + (1 - alpha) * phi * x[f, i] * np.conj(x[f, j])
                            )

                for f in range(n_freq):
                    # WV = W V_k
                    WV = np.zeros((n_chan, n_chan), dtype=complex)
                    for m in range(n_chan):
                        for i in range(n_chan):
                            s = 0j
                            for j in range(n_chan):
                                s += W[f, m, j] * V[k, f, j, i]
                            WV[m, i] = s

                    e = np.zeros(n_chan, dtype=complex)
                    e[k] = 1.0
                    w = np.linalg.solve(WV, e)  # true steering vector

                    # normalize by sqrt(w^H V_k w)
                    Vw = np.zeros(n_chan, dtype=complex)
                    for i in range(n_chan):
                        s = 0j
                        for j in range(n_chan):
                            s += V[k, f, i, j] * w[j]
                        Vw[i] = s
                    den = 0j
                    for i in range(n_chan):
                        den += np.conj(w[i]) * Vw[i]
                    den = max(den.real, eps)
                    w = w / np.sqrt(den)

                    for i in range(n_chan):
                        W[f, k, i] = np.conj(w[i])

        for f in range(n_freq):
            for i in range(K):
                s = 0j
                for j in range(n_chan):
                    s += W[f, i, j] * x[f, j]
                Y[t, f, i] = s

    return Y


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    X = (
        rng.normal(size=(8, 5, 3)) + 1j * rng.normal(size=(8, 5, 3))
    ).astype(np.complex128)

    for model in ["laplace", "gauss"]:
        for n_iter in [1, 3]:
            Yv = auxiva_iss_online(
                X, n_iter=n_iter, alpha=0.96, model=model, proj_back=False
            )
            Yb = iss_bruteforce(X, n_iter=n_iter, alpha=0.96, model=model)
            d_iss = np.max(np.abs(Yv - Yb))

            Yv = auxiva_ip_online(
                X, n_iter=n_iter, alpha=0.96, model=model, proj_back=False
            )
            Yb = ip_bruteforce(X, n_iter=n_iter, alpha=0.96, model=model)
            d_ip = np.max(np.abs(Yv - Yb))

            print(
                f"model={model:8s} n_iter={n_iter}   "
                f"ISS max|vec-brute| = {d_iss:.3e}   "
                f"IP max|vec-brute| = {d_ip:.3e}"
            )
