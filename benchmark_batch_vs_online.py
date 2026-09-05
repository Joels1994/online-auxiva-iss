#  Runtime comparison of batch and online AuxIVA, ISS against IP.
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
Four algorithms, two formulations times two update rules:

                ISS (inverse-free)      IP (needs a solve)
    batch       auxiva_iss_batch        auxiva_ip_batch
    online      auxiva_iss_online       auxiva_ip_online

The question this answers: why does the ISS speed advantage grow with
channel count offline, but flatten out online?

Batch ISS contracts over time and never forms a covariance matrix,
costing roughly F*M^2*N per sweep. Batch IP builds an M-by-M weighted
covariance per source and solves against it, costing about
F*M^3*max(M,N). The ratio grows with M, so the gap widens.

Online, that trick is unavailable: one incoming frame gives nothing to
sum over, so statistics are carried recursively as M-by-M covariances
and recovering the same quantities needs a matrix-vector product per
source at every pivot. Both rules land at the same order and the ratio
collapses toward a constant.

Synthetic complex Gaussian input is used throughout. Separation quality
is meaningless on random data, which is the point: this isolates the
cost of the update rule from everything else.

Run:  python benchmark_batch_vs_online.py
      python benchmark_batch_vs_online.py --quick
      python benchmark_batch_vs_online.py --channels 2 4 8 16 --plot out.png
"""
import argparse
import time

import numpy as np

from online_iva import auxiva_iss_online, auxiva_ip_online
from online_iva.batch_reference import auxiva_iss_batch, auxiva_ip_batch

# ------------------------------------------------------------------
# N_ITER_BATCH is deliberately large and N_TRIALS deliberately high.
# At small channel counts a batch sweep finishes in a few milliseconds,
# too short to time reliably, and the measured ratio then reflects
# scheduling noise rather than the algorithms. With smaller settings
# this benchmark produced a spurious 3.0x spike at 16 channels that
# vanished once the workload was raised.
# ------------------------------------------------------------------
N_FREQ = 129           # frequency bins
N_FRAMES = 400         # STFT frames
N_ITER_BATCH = 40      # sweeps over the whole recording
N_ITER_ONLINE = 2      # sweeps per incoming frame
ALPHA = 0.96
N_TRIALS = 5           # repeats, fastest kept

# matches the sweep in Figure 3 of the ISS paper
CHANNELS = [2, 4, 6, 8, 10, 12, 14, 16]


def best_time(fn, X, n_trials, **kwargs):
    """
    Fastest of n_trials runs. The minimum rather than the mean, because
    timing noise only ever makes a run slower, so the fastest run is the
    closest estimate of true cost.
    """
    best = np.inf
    for _ in range(n_trials):
        tic = time.perf_counter()
        fn(X, proj_back=False, **kwargs)
        best = min(best, time.perf_counter() - tic)
    return best


def run(channels, n_freq, n_frames, n_iter_batch, n_iter_online, n_trials):
    rng = np.random.default_rng(0)
    results = {k: [] for k in ("batch_iss", "batch_ip", "online_iss", "online_ip")}

    print(
        f"{n_freq} bins, {n_frames} frames, batch {n_iter_batch} sweeps, "
        f"online {n_iter_online} per frame, best of {n_trials}"
    )
    print(
        f"\n{'chan':>5} | {'batch ISS':>10} {'batch IP':>10} {'ratio':>7} "
        f"| {'online ISS':>11} {'online IP':>10} {'ratio':>7}   (seconds)"
    )
    print("-" * 78)

    for M in channels:
        X = (
            rng.normal(size=(n_frames, n_freq, M))
            + 1j * rng.normal(size=(n_frames, n_freq, M))
        ).astype(np.complex128)

        b_iss = best_time(auxiva_iss_batch, X, n_trials, n_src=M, n_iter=n_iter_batch)
        b_ip = best_time(auxiva_ip_batch, X, n_trials, n_src=M, n_iter=n_iter_batch)
        o_iss = best_time(
            auxiva_iss_online, X, n_trials, n_src=M,
            n_iter=n_iter_online, alpha=ALPHA,
        )
        o_ip = best_time(
            auxiva_ip_online, X, n_trials, n_src=M,
            n_iter=n_iter_online, alpha=ALPHA,
        )

        results["batch_iss"].append(b_iss)
        results["batch_ip"].append(b_ip)
        results["online_iss"].append(o_iss)
        results["online_ip"].append(o_ip)

        print(
            f"{M:>5} | {b_iss:>10.3f} {b_ip:>10.3f} {b_ip / b_iss:>6.2f}x "
            f"| {o_iss:>11.3f} {o_ip:>10.3f} {o_ip / o_iss:>6.2f}x"
        )

    return results


def summarize(channels, results):
    ch = np.array(channels)
    b_ratio = np.array(results["batch_ip"]) / np.array(results["batch_iss"])
    o_ratio = np.array(results["online_ip"]) / np.array(results["online_iss"])

    print("\n" + "=" * 78)
    print("HOW MUCH FASTER ISS IS THAN IP")
    print("=" * 78)
    print(f"{'channels':>10}{'batch':>12}{'online':>12}")
    print("-" * 78)
    for M, b, o in zip(ch, b_ratio, o_ratio):
        print(f"{M:>10}{b:>11.2f}x{o:>11.2f}x")
    print("-" * 78)
    print(
        f"batch  : {b_ratio[0]:.2f}x at {ch[0]} channels -> "
        f"{b_ratio[-1]:.2f}x at {ch[-1]}   (advantage grows)"
    )
    print(
        f"online : {o_ratio[0]:.2f}x at {ch[0]} channels -> "
        f"{o_ratio[-1]:.2f}x at {ch[-1]}   (advantage decays)"
    )
    print("=" * 78)
    return ch, b_ratio, o_ratio


def make_plot(ch, b_ratio, o_ratio, results, path):
    import matplotlib
    matplotlib.use("Agg")           # no display needed
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(ch, b_ratio, "o-", label="batch (offline)", linewidth=2)
    ax1.plot(ch, o_ratio, "s-", label="online", linewidth=2)
    ax1.axhline(1.0, color="grey", linestyle=":", linewidth=1)
    ax1.set_xlabel("number of channels / sources")
    ax1.set_ylabel("IP time / ISS time")
    ax1.set_title("How much faster ISS is than IP")
    ax1.legend()
    ax1.grid(alpha=0.3)

    for key, style, lab in [
        ("batch_iss", "o-", "batch ISS"),
        ("batch_ip", "o--", "batch IP"),
        ("online_iss", "s-", "online ISS"),
        ("online_ip", "s--", "online IP"),
    ]:
        ax2.plot(ch, results[key], style, label=lab, linewidth=1.8)
    ax2.set_xlabel("number of channels / sources")
    ax2.set_ylabel("seconds")
    ax2.set_yscale("log")
    ax2.set_title("Absolute runtime (log scale)")
    ax2.legend()
    ax2.grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"\nplot written to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runtime comparison of batch and online AuxIVA."
    )
    parser.add_argument(
        "--channels", type=int, nargs="+", default=CHANNELS,
        help="channel counts to sweep (default: %(default)s)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="smaller workload, fewer trials; noisy, for a fast smoke test",
    )
    parser.add_argument(
        "--plot", metavar="PATH", nargs="?", const="batch_vs_online_runtime.png",
        help="also write the figure to PATH (needs matplotlib)",
    )
    args = parser.parse_args()

    if args.quick:
        results = run(args.channels, 65, 150, 10, 1, 2)
    else:
        results = run(
            args.channels, N_FREQ, N_FRAMES,
            N_ITER_BATCH, N_ITER_ONLINE, N_TRIALS,
        )

    ch, b_ratio, o_ratio = summarize(args.channels, results)

    if args.plot:
        make_plot(ch, b_ratio, o_ratio, results, args.plot)
