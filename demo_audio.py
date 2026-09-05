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
import argparse
import glob
import os
import shutil
import tarfile
import tempfile
import time
from urllib.request import urlretrieve

import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile

from online_iva import auxiva_iss_online, auxiva_ip_online, metrics

# pyroomacoustics moved the STFT helpers from pra.transform into the
# pra.transform.stft submodule after the 0.1.x series. Support both so
# this runs on an old pinned install and on a current one alike.
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

SAMPLE_URL = (
    "https://github.com/LCAV/pyroomacoustics/raw/master/examples/input_samples"
)
SAMPLE_FILES = [
    "cmu_arctic_us_aew_a0001.wav",
    "cmu_arctic_us_axb_a0004.wav",
]
# ------------------------------------------------------------------
# Audio source switch. Override from the command line with
#   python demo_audio.py --long
#
#   "short" : the two CMU ARCTIC clips above, about 4.4 s each,
#             repeated N_REPEAT_SIGNAL times so the online algorithms
#             have enough frames to adapt. This is the default and
#             uses the files already in samples/.
#
#   "long"  : two genuine 2-minute recordings, one male speaker and
#             one female, built by concatenating consecutive CMU
#             ARCTIC utterances from the concat15 release. No
#             repetition is needed, the signal is already long enough
#             for the recursive statistics to converge.
# ------------------------------------------------------------------
AUDIO_SOURCE = "short"

LONG_SECONDS = 120
LONG_SPEAKERS = ("female_axb", "male_bdl")

SAMPLE_DIR = "samples"
LONG_SAMPLE_DIR = "samples_long"

# Where the concat15 corpus may already exist locally, checked before
# falling back to downloading it.
LONG_LOCAL_CANDIDATES = [
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        "code_2020ICASSP_iss-master",
        "samples",
    ),
]
LONG_CORPUS_URL = "https://zenodo.org/record/3066489/files/cmu_arctic_concat15.tar.gz"

# separate output folders so the two modes never overwrite each other
OUTPUT_DIR = "output_audio"

# the short clips need repeating; the long ones do not
N_REPEAT_SIGNAL = 4

# Target signal-to-noise ratio at the reference microphone, in dB.
# The noise standard deviation is derived from the measured signal
# level below. Note that dB for an amplitude ratio uses 20 log10, not
# 10 log10; getting that wrong silently makes the mixture far cleaner
# than intended.
SNR_DB = 15

# Projection back resolves the per-frequency scale ambiguity by fitting
# one scalar per bin against the reference microphone. Over a long
# recording a single scalar cannot track a demixing matrix that keeps
# adapting: on the 2-minute material, fitting once over the whole signal
# gives about -3 dB SDR while refitting every 2 s gives about +2 dB.
# None means fit once, which is what "short" uses.
PROJ_BACK_SEGMENT_S = None


def configure(audio_source):
    """
    Apply the short/long choice to the module-level settings. Kept in
    one place so the two modes cannot drift apart.
    """
    global AUDIO_SOURCE, N_REPEAT_SIGNAL, OUTPUT_DIR, PROJ_BACK_SEGMENT_S

    if audio_source not in ("short", "long"):
        raise ValueError(
            f"audio_source must be 'short' or 'long', got {audio_source!r}"
        )

    AUDIO_SOURCE = audio_source
    if audio_source == "short":
        N_REPEAT_SIGNAL = 4
        OUTPUT_DIR = "output_audio"
        PROJ_BACK_SEGMENT_S = None
    else:
        N_REPEAT_SIGNAL = 1
        OUTPUT_DIR = "output_audio_long"
        PROJ_BACK_SEGMENT_S = 2.0


def _find_long_corpus():
    """
    Locate the concat15 corpus of longer CMU ARCTIC recordings, using a
    local copy if one exists and downloading it (about 47 MB) otherwise.
    """
    for cand in LONG_LOCAL_CANDIDATES:
        cand = os.path.abspath(cand)
        if glob.glob(os.path.join(cand, "cmu_arctic_*_*.wav")):
            return cand

    if glob.glob(os.path.join(LONG_SAMPLE_DIR, "cmu_arctic_*_*.wav")):
        return LONG_SAMPLE_DIR

    os.makedirs(LONG_SAMPLE_DIR, exist_ok=True)
    print("downloading the concat15 corpus (about 47 MB), one time only ...")
    with tempfile.TemporaryDirectory() as tmp:
        arc = os.path.join(tmp, "concat15.tar.gz")
        urlretrieve(LONG_CORPUS_URL, filename=arc)
        with tarfile.open(arc) as tf:
            tf.extractall(tmp)
        for root, _, files in os.walk(tmp):
            for f in files:
                if f.endswith(".wav") or f == "metadata.json":
                    shutil.copy(os.path.join(root, f), LONG_SAMPLE_DIR)
    return LONG_SAMPLE_DIR


def get_long_samples(target_seconds=LONG_SECONDS, speakers=LONG_SPEAKERS):
    """
    Build one continuous recording per speaker by concatenating that
    speaker's consecutive utterances until the target duration is
    reached. Returns [(fs, signal), ...], one entry per speaker.
    """
    corpus = _find_long_corpus()
    signals = []

    for spk in speakers:
        files = glob.glob(os.path.join(corpus, f"cmu_arctic_{spk}_*.wav"))
        if not files:
            raise FileNotFoundError(f"no recordings for speaker '{spk}' in {corpus}")
        files.sort(
            key=lambda p: int(os.path.basename(p).rsplit("_", 1)[1].split(".")[0])
        )

        parts, total, fs = [], 0.0, None
        for f in files:
            fs_f, audio = wavfile.read(f)
            fs = fs_f if fs is None else fs
            if fs_f != fs:
                raise ValueError(f"sampling rate mismatch in {f}")
            if audio.dtype == np.int16:
                audio = audio.astype(np.float64) / (2 ** 15)
            if audio.ndim > 1:
                audio = audio[:, 0]
            parts.append(audio)
            total += len(audio) / fs
            if total >= target_seconds:
                break

        sig = np.concatenate(parts)
        want = int(target_seconds * fs)
        if len(sig) < want:
            raise ValueError(
                f"speaker '{spk}' only yields {len(sig) / fs:.0f} s, "
                f"needed {target_seconds} s"
            )
        signals.append((fs, sig[:want]))
        print(f"  {spk}: using {len(parts)} files -> {want / fs:.0f} s")

    return signals


def apply_projection_back(Y, X, fs, hop, segment_seconds=None):
    """
    Resolve the output scale against microphone 0. With
    segment_seconds=None this fits a single scalar per frequency bin
    over the whole recording, exactly as proj_back=True does inside the
    algorithms. With a value set it refits every segment, which tracks a
    slowly drifting demixing matrix far better on long signals.
    """
    from online_iva import project_back

    if segment_seconds is None:
        return project_back(Y, X[:, :, 0])

    Y = Y.copy()
    step = max(1, int(round(segment_seconds * fs / hop)))
    for a in range(0, Y.shape[0], step):
        sl = slice(a, min(a + step, Y.shape[0]))
        Y[sl] = project_back(Y[sl], X[sl, :, 0])
    return Y


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
    framesize, n_src, n_mics = 1024, 2, 2
    hop = framesize // 2
    win_a = np.hamming(framesize)
    win_s = compute_synthesis_window(win_a, hop)

    np.random.seed(0)

    if AUDIO_SOURCE == "short":
        source_signals = [s[1] for s in get_samples()]
    else:
        source_signals = [s[1] for s in get_long_samples()]

    print(
        f"audio source    : {AUDIO_SOURCE} "
        f"({len(source_signals[0]) / fs:.1f} s per speaker, "
        f"repeated {N_REPEAT_SIGNAL}x)"
    )

    room = make_room(
        [10.0, 7.5, 3.2], n_mics, source_signals,
        fs=fs, max_order=17, absorption=0.35, radius=0.05,
    )

    premix = room.simulate(return_premix=True)
    premix /= np.std(premix[:, 0, :], axis=1)[:, None, None]

    clean = np.sum(premix, axis=0)
    sigma_n = np.std(clean[0]) * 10 ** (-SNR_DB / 20)
    mix = clean + sigma_n * np.random.randn(*premix.shape[1:])

    one_pass_len = mix.shape[1]
    X = stft_analysis(
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
        # Time the separation itself only. The STFT, the projection back
        # and the metrics are excluded, so this measures the update rule
        # rather than the surrounding plumbing.
        tic = time.perf_counter()
        Y = fn(X, n_src=n_src, n_iter=3, alpha=0.96, model="laplace", proj_back=False)
        elapsed = time.perf_counter() - tic

        n_blocks = X.shape[0]
        ms_per_block = 1000 * elapsed / n_blocks
        rtf = elapsed / (n_blocks * hop / fs)

        # scaling is applied here so it can be done in segments when the
        # recording is long enough for the demixing scale to drift
        Y = apply_projection_back(Y, X, fs, hop, PROJ_BACK_SEGMENT_S)

        y = stft_synthesis(Y, framesize, hop, win=win_s)
        y = y[framesize - hop:, :].astype(np.float64)

        m = min(y.shape[0] - eval_start, premix.shape[2])
        y_eval = y[eval_start:eval_start + m, :]

        sdr, sir, sar, perm = metrics.si_bss_eval(premix[:n_src, 0, :m].T, y_eval)
        y_eval = y_eval[:, perm]  # line outputs up with references

        for i in range(n_src):
            write_wav(
                os.path.join(OUTPUT_DIR, f"{label}_src{i + 1}.wav"), y_eval[:, i], fs
            )

        print(
            f"{label:12s} SDR={np.round(sdr, 2)}  SIR={np.round(sir, 2)}  "
            f"time={elapsed:6.2f} s  {ms_per_block:5.2f} ms/block  "
            f"realtime factor={rtf:.3f}"
        )

    print(f"\naudio written to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Separate a simulated two-speaker mixture with the "
        "online AuxIVA algorithms."
    )
    parser.add_argument(
        "--long",
        action="store_true",
        help="use two genuine 2-minute recordings instead of the short "
        "clips (writes to output_audio_long/)",
    )
    args = parser.parse_args()

    configure("long" if args.long else "short")
    main()
