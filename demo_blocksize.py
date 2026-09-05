#  Block-size study for the online AuxIVA implementations.
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
Runs the online algorithms at different STFT block sizes.

Framing scheme (unchanged from demo_audio.py, only the size differs):
a block of `block` samples is taken every `block // 2` samples, so
consecutive blocks overlap by half and every input sample falls inside
exactly 2 blocks. With block = 1024 at 16 kHz that is a 64 ms window
advancing 32 ms at a time.

    block:    |------1024------|
    next:              |------1024------|
    overlap:           |--512--|

Smaller blocks trade frequency resolution for time resolution: fewer
bins per frame, but more frames per second, so the recursive statistics
update more often. In reverberant rooms the narrowband approximation
that IVA relies on needs the window to be long relative to the room
impulse response, so shrinking it too far costs separation quality.

Run:  python demo_blocksize.py
"""
import time

import numpy as np
import pyroomacoustics as pra

from online_iva import auxiva_iss_online, auxiva_ip_online, metrics
from demo_audio import get_samples, make_room, SNR_DB

try:
    from pyroomacoustics.transform.stft import (
        analysis as stft_analysis,
        synthesis as stft_synthesis,
        compute_synthesis_window,
    )
except (ImportError, AttributeError):  # pragma: no cover - old versions
    stft_analysis = pra.transform.analysis
    stft_synthesis = pra.transform.synthesis
    compute_synthesis_window = pra.transform.compute_synthesis_window

FS = 16000
N_SRC = 2
N_ITER = 3
ALPHA = 0.96
N_REPEAT_SIGNAL = 4

BLOCK_SIZES = [1024, 2048, 4096]


def build_mixture():
    """One reverberant two-speaker mixture, shared by every block size."""
    np.random.seed(0)
    source_signals = [s[1] for s in get_samples()]

    room = make_room(
        [10.0, 7.5, 3.2], N_SRC, source_signals,
        fs=FS, max_order=17, absorption=0.35, radius=0.05,
    )

    premix = room.simulate(return_premix=True)
    premix /= np.std(premix[:, 0, :], axis=1)[:, None, None]

    clean = np.sum(premix, axis=0)
    sigma_n = np.std(clean[0]) * 10 ** (-SNR_DB / 20)
    mix = clean + sigma_n * np.random.randn(*premix.shape[1:])

    return premix, mix


def run(block, premix, mix):
    hop = block // 2
    win_a = np.hamming(block)
    win_s = compute_synthesis_window(win_a, hop)

    one_pass_len = mix.shape[1]
    X = stft_analysis(
        np.tile(mix, (1, N_REPEAT_SIGNAL)).T, block, hop, win=win_a
    ).astype(np.complex128)

    eval_start = (N_REPEAT_SIGNAL - 1) * one_pass_len

    rows = []
    for label, fn in [("ISS", auxiva_iss_online), ("IP", auxiva_ip_online)]:
        tic = time.perf_counter()
        Y = fn(
            X, n_src=N_SRC, n_iter=N_ITER, alpha=ALPHA,
            model="laplace", proj_back=True,
        )
        elapsed = time.perf_counter() - tic

        y = stft_synthesis(Y, block, hop, win=win_s)
        y = y[block - hop:, :].astype(np.float64)

        m = min(y.shape[0] - eval_start, premix.shape[2])
        sdr, sir, sar, perm = metrics.si_bss_eval(
            premix[:N_SRC, 0, :m].T, y[eval_start:eval_start + m, :]
        )

        rows.append(
            (label, X.shape[0], X.shape[1], elapsed,
             1000 * elapsed / X.shape[0], np.mean(sdr), np.mean(sir))
        )
    return rows


def main():
    premix, mix = build_mixture()

    print(f"{'block':>6} {'hop':>6} {'win ms':>7} {'blocks':>7} {'bins':>6} "
          f"{'algo':>5} {'ms/block':>9} {'SDR':>7} {'SIR':>7}")

    for block in BLOCK_SIZES:
        for label, n_frames, n_bins, _, ms, sdr, sir in run(block, premix, mix):
            print(
                f"{block:>6} {block // 2:>6} {1000 * block / FS:>7.0f} "
                f"{n_frames:>7} {n_bins:>6} {label:>5} {ms:>9.2f} "
                f"{sdr:>7.2f} {sir:>7.2f}"
            )


if __name__ == "__main__":
    main()
