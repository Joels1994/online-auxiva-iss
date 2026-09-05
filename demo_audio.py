#  Audio demo for the online AuxIVA implementations.
#  Copyright (C) 2026
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
Separates a simulated two-speaker mixture with both online algorithms
and writes the audio to ./output_audio/ so the result can be listened
to.

Speech comes from the CMU ARCTIC corpus, downloaded on first run from
the pyroomacoustics example data (six short utterances, two speakers).

Needs pyroomacoustics and scipy in addition to numpy.

Run:  python demo_audio.py
"""
import os
from urllib.request import urlretrieve

import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile

from online_iva import auxiva_iss_online, auxiva_ip_online, metrics

SAMPLE_URL = (
    "https://github.com/LCAV/pyroomacoustics/raw/master/examples/input_samples"
)
SAMPLE_FILES = [
    "cmu_arctic_us_aew_a0001.wav",
    "cmu_arctic_us_axb_a0004.wav",
]
SAMPLE_DIR = "samples"
OUTPUT_DIR = "output_audio"

# the online algorithms adapt over time, so the mixture is repeated to
# give them enough frames to converge; only the last pass is scored
N_REPEAT_SIGNAL = 4


def get_samples():
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    signals = []
    for fn in SAMPLE_FILES:
        path = os.path.join(SAMPLE_DIR, fn)
        if not os.path.exists(path):
            print(f"downloading {fn} ...")
            urlretrieve(f"{SAMPLE_URL}/{fn}", filename=path)
        fs, audio = wavfile.read(path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float64) / (2 ** 15)
        signals.append((fs, audio))
    return signals


def write_wav(path, signal, fs, peak=0.9):
    """
    Write a mono float signal as 16-bit wav, peak-normalized so it is
    audible without clipping. Normalization is per file, so loudness is
    not comparable between files; the printed SDR/SIR values are the
    quantitative comparison.
    """
    x = np.asarray(signal, dtype=np.float64)
    m = np.max(np.abs(x))
    if m > 0:
        x = peak * x / m
    wavfile.write(path, int(fs), (x * (2 ** 15 - 1)).astype(np.int16))


def make_room(room_dim, n_mics, source_signals, fs, max_order, absorption, radius):
    room_dim = np.array(room_dim, dtype=float)
    room = pra.ShoeBox(room_dim, fs=fs, absorption=absorption, max_order=max_order)

    # circular microphone array, near the middle of the room
    center = room_dim / 2 * (0.995 + 0.01 * np.random.rand(*room_dim.shape))
    theta = 2 * np.pi * np.arange(n_mics) / n_mics
    R = center[:, None] + radius * np.array(
        [np.cos(theta), np.sin(theta), np.zeros(n_mics)]
    )
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

    # sources placed at random
    locs = np.random.rand(len(source_signals), *room_dim.shape) * room_dim[None, :]
    for loc, sig in zip(locs, source_signals):
        room.add_source(loc, signal=sig)

    return room


def main():
    fs = 16000
    framesize, n_src, n_mics = 4096, 2, 2
    hop = framesize // 2
    win_a = np.hamming(framesize)
    win_s = pra.transform.compute_synthesis_window(win_a, hop)

    np.random.seed(0)
    source_signals = [s[1] for s in get_samples()]

    room = make_room(
        [10.0, 7.5, 3.2], n_mics, source_signals,
        fs=fs, max_order=17, absorption=0.35, radius=0.05,
    )

    premix = room.simulate(return_premix=True)
    premix /= np.std(premix[:, 0, :], axis=1)[:, None, None]

    sigma_n = 10 ** (-15 / 10) * premix.shape[0]
    mix = np.sum(premix, axis=0) + sigma_n * np.random.randn(*premix.shape[1:])

    one_pass_len = mix.shape[1]
    X = pra.transform.analysis(
        np.tile(mix, (1, N_REPEAT_SIGNAL)).T, framesize, hop, win=win_a
    ).astype(np.complex128)

    print("simulation done")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_wav(os.path.join(OUTPUT_DIR, "mixture_mic0.wav"), mix[0, :], fs)
    for i in range(n_src):
        write_wav(
            os.path.join(OUTPUT_DIR, f"reference_src{i + 1}.wav"), premix[i, 0, :], fs
        )

    eval_start = (N_REPEAT_SIGNAL - 1) * one_pass_len

    for label, fn in [
        ("online_iss", auxiva_iss_online),
        ("online_ip", auxiva_ip_online),
    ]:
        Y = fn(X, n_src=n_src, n_iter=3, alpha=0.96, model="laplace", proj_back=True)

        y = pra.transform.synthesis(Y, framesize, hop, win=win_s)
        y = y[framesize - hop:, :].astype(np.float64)

        m = min(y.shape[0] - eval_start, premix.shape[2])
        y_eval = y[eval_start:eval_start + m, :]

        sdr, sir, sar, perm = metrics.si_bss_eval(premix[:n_src, 0, :m].T, y_eval)
        y_eval = y_eval[:, perm]  # line outputs up with references

        for i in range(n_src):
            write_wav(
                os.path.join(OUTPUT_DIR, f"{label}_src{i + 1}.wav"), y_eval[:, i], fs
            )

        print(f"{label:12s} SDR={np.round(sdr, 2)}  SIR={np.round(sir, 2)}")

    print(f"\naudio written to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
