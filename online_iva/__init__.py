"""
Online (frame-recursive) independent vector analysis.

Two algorithms, both pure NumPy, sharing the same recursive weighted
covariance, auxiliary variable and per-frame iteration structure, so
that the only thing differing between them is the demixing update:

    auxiva_ip_online   iterative projection, needs a linear solve
    auxiva_iss_online  iterative source steering, inverse-free

Reference for the ISS variant:
    T. Nakashima and N. Ono, "Inverse-free Online Independent Vector
    Analysis with Flexible Iterative Source Steering", arXiv:2209.00937.
"""

from .auxiva_iss_online import auxiva_iss_online
from .auxiva_ip_online import auxiva_ip_online
from .projection_back import project_back
from . import metrics

algorithms = {
    "auxiva_iss_online": auxiva_iss_online,
    "auxiva_ip_online": auxiva_ip_online,
}
