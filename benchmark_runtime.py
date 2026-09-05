"""
Runtime comparison of online AuxIVA-IP against online AuxIVA-ISS.

Uses synthetic complex Gaussian data, so it needs nothing but NumPy and
runs anywhere. Separation quality is meaningless on random input by
construction; the point here is purely how the cost of each update rule
grows with the channel count.

Run:  python benchmark_runtime.py
"""
import time

import numpy as np

from online_iva import algorithms

N_ITER = 3
ALPHA = 0.96
MODEL = "laplace"


def main(n_freq=257, n_frames=60, n_trials=3):
    print("online AuxIVA-IP vs AuxIVA-ISS, ms per frame")
    print(f"(n_freq={n_freq}, n_frames={n_frames}, n_iter={N_ITER}, alpha={ALPHA})\n")
    print(f"{'n_chan':>7} {'IP':>10} {'ISS':>10} {'ISS speedup':>13}")

    rng = np.random.default_rng(0)

    for n_chan in [2, 4, 6, 8, 10, 12]:

        X = (
            rng.normal(size=(n_frames, n_freq, n_chan))
            + 1j * rng.normal(size=(n_frames, n_freq, n_chan))
        ).astype(np.complex128)

        timings = {}
        for name, fn in algorithms.items():
            best = np.inf
            for _ in range(n_trials):
                tic = time.perf_counter()
                fn(
                    X, n_src=n_chan, n_iter=N_ITER, alpha=ALPHA,
                    model=MODEL, proj_back=False,
                )
                best = min(best, time.perf_counter() - tic)
            timings[name] = 1000 * best / n_frames

        ip = timings["auxiva_ip_online"]
        iss = timings["auxiva_iss_online"]
        print(f"{n_chan:>7} {ip:>10.2f} {iss:>10.2f} {ip / iss:>12.2f}x")


if __name__ == "__main__":
    main()
