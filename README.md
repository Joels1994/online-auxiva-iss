# Online AuxIVA: ISS and IP

Frame-recursive (online) independent vector analysis, in pure NumPy.

Two algorithms are provided. They share the same recursive weighted
covariance, the same auxiliary variable, and the same per-frame
iteration structure, so the only thing that differs between them is the
demixing update rule:

| Module | Update rule | Needs a matrix solve |
| --- | --- | --- |
| `online_iva/auxiva_ip_online.py` | iterative projection (baseline) | yes |
| `online_iva/auxiva_iss_online.py` | iterative source steering | no |

Reference for the ISS variant: T. Nakashima and N. Ono, *Inverse-free
Online Independent Vector Analysis with Flexible Iterative Source
Steering*, arXiv:2209.00937.

## Relation to the batch algorithms

The batch AuxIVA takes the whole recording up front and sweeps over
every frame many times. These process one STFT frame at a time,
keeping each source's spatial covariance as a running, exponentially
forgotten estimate (`alpha`) and applying only a few iterations per
incoming frame (`n_iter`). Frame `t`'s output depends only on frames
`<= t`.

## Usage

```python
import numpy as np
from online_iva import auxiva_iss_online

# X: STFT of the mixture, shape (n_frames, n_freq, n_chan)
Y = auxiva_iss_online(X, n_iter=3, alpha=0.96, model="laplace")
```

Both take the same arguments. `model` is `"laplace"` or `"gauss"`.

## Audio demo

```bash
python demo_audio.py
```

Simulates a two-speaker reverberant mixture, separates it with both
algorithms, and writes wav files to `output_audio/`. Speech is two CMU
ARCTIC utterances, downloaded on first run. Needs `pyroomacoustics`
and `scipy` as well as NumPy.

Committed output of that script is in `output_audio/`:

| File | What it is |
| --- | --- |
| `mixture_mic0.wav` | what the first microphone hears, both speakers overlapping |
| `reference_src<i>.wav` | the clean source at that same mic, the target |
| `online_iss_src<i>.wav` | separated by online AuxIVA-ISS |
| `online_ip_src<i>.wav` | separated by online AuxIVA-IP |

Outputs are permutation-aligned, so `..._src1.wav` corresponds to
`reference_src1.wav` in every file. Each file is peak-normalized
individually so it is audible, which means loudness is not comparable
between files. Scores for that run:

| Method | SDR (dB) | SIR (dB) |
| --- | --- | --- |
| Online ISS | 2.60 / 3.79 | 15.24 / 12.33 |
| Online IP | 2.57 / 3.78 | 15.26 / 12.30 |

The saved segment is the final repetition of the mixture, since the
online methods need time to adapt.

## Benchmark

```bash
python benchmark_runtime.py
```

Measured on an Apple M-series machine, milliseconds per frame,
representative of three runs (the ratio at 10 and 12 channels varies by
roughly +/- 0.05x between runs):

| Channels | IP | ISS | ISS speedup |
| --- | --- | --- | --- |
| 2 | 0.62 | 0.42 | 1.46x |
| 4 | 1.79 | 1.36 | 1.32x |
| 6 | 4.49 | 3.65 | 1.23x |
| 8 | 9.03 | 8.42 | 1.07x |
| 10 | 17.30 | 17.21 | 1.00x |
| 12 | 30.95 | 29.59 | 1.05x |

So ISS is clearly ahead up to about 6 channels and roughly break-even
from 8 channels upward. Note that both algorithms snapshot their
covariance once per frame (required for the recursion to be anchored
correctly, see below), which adds the same overhead to each and so
compresses the ratio toward 1.0.

The same shape holds at `n_iter=1`, so the crossover is a property of
the online formulation rather than of the sweep count:

| Channels | IP | ISS | ISS speedup |
| --- | --- | --- | --- |
| 2 | 0.22 | 0.15 | 1.42x |
| 4 | 0.61 | 0.47 | 1.29x |
| 6 | 1.49 | 1.25 | 1.19x |
| 8 | 3.00 | 2.96 | 1.01x |
| 10 | 5.64 | 5.62 | 1.00x |
| 12 | 9.48 | 10.36 | 0.91x |

## Choosing n_iter

On the two-source speech mixture of `demo_audio.py`:

| n_iter | Algorithm | ms/frame | Mean SDR (dB) | Mean SIR (dB) |
| --- | --- | --- | --- | --- |
| 1 | ISS | 0.93 | 2.59 | 14.52 |
| 1 | IP | 1.49 | 2.10 | 13.97 |
| 3 | ISS | 2.64 | 3.20 | 13.78 |
| 3 | IP | 4.12 | 3.17 | 13.78 |

A single sweep per frame is a good operating point. It gives up about
0.6 dB of SDR but gains roughly 0.7 dB of SIR, at just under a third
of the cost. For a streaming system, where per-frame cost is the
binding constraint, that trade is usually worth taking.

ISS also leads IP on quality at `n_iter=1` (2.59 against 2.10 SDR),
while at `n_iter=3` the two are indistinguishable. With one sweep the
updates have not yet reached the same fixed point, so the difference in
per-sweep progress is visible, and it favours ISS. That is the regime
an online system actually runs in.

The default in both functions is `n_iter=3`, kept so the numbers
elsewhere in this README reproduce; pass `n_iter=1` for streaming use.

On simulated two-source speech mixtures the two reach essentially the
same separation quality, as expected since they optimize the same cost
function:

| Method | SDR (dB) | SIR (dB) |
| --- | --- | --- |
| Online ISS | 2.21 / 0.55 | 8.59 / 12.43 |
| Online IP | 2.29 / 0.56 | 8.79 / 12.45 |

## Known limitations

These are real and worth knowing before relying on this code.

- **The ISS speed advantage here is modest, and shrinks with channel
  count.** In the online formulation both methods must carry an
  M-by-M covariance per source, and ISS still needs one matrix-vector
  product per source at each pivot, so both end up the same asymptotic
  order per sweep. ISS avoids the solve, which is a constant-factor
  win. This is unlike the batch case, where ISS works directly on the
  separated outputs and is genuinely one order cheaper.
- **The flexible update of the paper's Section IV-B is not
  implemented.** Every source is updated on every sweep. Restricting
  updates to only the moving sources is where the larger online saving
  is expected to come from.
- **Projection back is not causal.** `proj_back=True` applies a
  batch projection back over the whole output at the end, which reads
  future frames. The separation loop itself is strictly causal; this
  final scaling step is not. A true streaming system needs a recursive
  projection back. Turning it off costs about 17 dB of SDR, since the
  per-frequency scale ambiguity is then unresolved.
- **Determined case only**, meaning the number of microphones must
  equal the number of sources.

## Implementation note: anchoring the covariance recursion

Within a frame's inner sweeps, the covariance recursion is anchored to
the *previous frame's* covariance, not to the running iterate. Both
modules snapshot it on entry to each frame and rebuild from that
snapshot. Recomputing the auxiliary variable as `W` improves is the
point of the inner loop; decaying by `alpha` again is not. Applying the
recursion to the running iterate instead gives an effective forgetting
factor of `alpha ** n_iter` and injects the frame's `x x^H` once per
sweep. Getting this wrong is silent: it still separates, just worse. It
cost about 1.5 dB of SDR here before being corrected.

## Install and run on a fresh machine

```bash
git clone https://github.com/Joels1994/online-auxiva-iss.git
cd online-auxiva-iss
pip install numpy scipy pyroomacoustics

python benchmark_runtime.py   # runtime comparison, numpy only
python demo_audio.py          # separates speech, writes output_audio/
```

No build step and no compiled extension. `demo_audio.py` downloads its
two speech samples on first run, so that one needs network access;
nothing else does.

Verified working on two combinations: numpy 1.24 with pyroomacoustics
0.10, and numpy 1.23 with pyroomacoustics 0.1.23. The STFT helpers
moved between those pyroomacoustics releases, so the demo detects
their location at import.

## Requirements

NumPy, and SciPy for the SDR/SIR metrics in `online_iva/metrics.py`.
The runtime benchmark needs only NumPy. The audio demo additionally
needs `pyroomacoustics`.

## License and attribution

GPL-3.0, see `LICENSE`.

`online_iva/projection_back.py` and `online_iva/metrics.py` are taken
from [piva](https://github.com/fakufaku/piva) by Robin Scheibler, which
is GPL-3.0. The online algorithms in `auxiva_iss_online.py` and
`auxiva_ip_online.py` were written for this repository, following the
paper's Algorithm 1 and the batch implementations in piva.

Speech samples used by the demo are from the
[CMU ARCTIC](http://festvox.org/cmu_arctic/) corpus, fetched from the
pyroomacoustics example data.
