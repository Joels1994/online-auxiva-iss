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

## Requirements

NumPy, and SciPy for the SDR/SIR metrics in `online_iva/metrics.py`.
The runtime benchmark needs only NumPy.
