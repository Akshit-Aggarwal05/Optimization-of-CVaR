"""Benchmark data and scenario generators from Rockafellar & Uryasev (2000), Section 3.

The paper's numerical experiment builds an optimal portfolio from three
instruments - the S&P 500, a portfolio of long-term US government bonds, and a
portfolio of small-cap stocks - whose monthly returns are modelled as jointly
normal.  Tables 1-4 of the paper give the inputs and the minimum-variance
benchmark, and Tables 5-6 give the LP results under pseudo-random and
quasi-random (Sobol) sampling respectively.

This module reproduces those inputs verbatim so that the risk engine can be
regression-tested against published numbers.  The Proposition in Section 3
guarantees the benchmark is exact: when the loss is normally distributed,
:math:`\\beta \\ge 0.5`, and the expected-return constraint (15) is active, the
minimum-variance, minimum-VaR and minimum-CVaR portfolios coincide.  So the
scenario-based LP *must* converge to Table 3, and its VaR/CVaR to Table 4.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.stats import norm, qmc

__all__ = [
    "INSTRUMENTS",
    "MEAN_RETURNS",
    "COVARIANCE",
    "MIN_VARIANCE_WEIGHTS",
    "MIN_VARIANCE_VARIANCE",
    "TARGET_RETURN",
    "REFERENCE_VAR",
    "REFERENCE_CVAR",
    "sample_normal_returns",
]

FloatArray = npt.NDArray[np.float64]

#: Instrument labels, in the column order used by every array in this module.
INSTRUMENTS: Final[tuple[str, str, str]] = ("S&P 500", "Gov Bond", "Small Cap")

#: Table 1 - mean monthly returns ``m``.
MEAN_RETURNS: Final[FloatArray] = np.array(
    [0.0101110, 0.0043532, 0.0137058],
    dtype=np.float64,
)

#: Table 2 - covariance matrix ``V`` of monthly returns.
COVARIANCE: Final[FloatArray] = np.array(
    [
        [0.00324625, 0.00022983, 0.00420395],
        [0.00022983, 0.00049937, 0.00019247],
        [0.00420395, 0.00019247, 0.00764097],
    ],
    dtype=np.float64,
)

#: Table 3 - the unique minimum-variance portfolio, i.e. the solution of (P3).
MIN_VARIANCE_WEIGHTS: Final[FloatArray] = np.array(
    [0.452013, 0.115573, 0.432414],
    dtype=np.float64,
)

#: Variance of the Table 3 portfolio, quoted in the text of Section 3.
MIN_VARIANCE_VARIANCE: Final[float] = 0.00378529

#: The expected-return floor ``R`` of eq. (15) used throughout Section 3.
#: The constraint is active at the optimum, which is what makes the
#: Proposition's equivalence apply.
TARGET_RETURN: Final[float] = 0.011

#: Table 4 - beta-VaR of the minimum-variance portfolio, by confidence level.
REFERENCE_VAR: Final[dict[float, float]] = {
    0.90: 0.067847,
    0.95: 0.090200,
    0.99: 0.132128,
}

#: Table 4 - beta-CVaR of the minimum-variance portfolio, by confidence level.
REFERENCE_CVAR: Final[dict[float, float]] = {
    0.90: 0.096975,
    0.95: 0.115908,
    0.99: 0.152977,
}


def sample_normal_returns(
    n_scenarios: int,
    mean: npt.ArrayLike = MEAN_RETURNS,
    covariance: npt.ArrayLike = COVARIANCE,
    engine: str = "sobol",
    seed: int | None = 20000,
) -> FloatArray:
    """Draw scenario returns from the multivariate normal law ``N(m, V)``.

    Both sampling schemes compared in the paper are provided:

    ``"pseudo"``
        Conventional Monte Carlo (Table 5).  Convergence of the CVaR estimate
        is "slow at best" in the paper's words - errors of 1-3% persist even at
        20,000 scenarios.
    ``"sobol"``
        A scrambled Sobol low-discrepancy sequence mapped through the inverse
        normal CDF (Table 6).  The paper reports sub-1% agreement with the
        analytic benchmark once the sample exceeds 10,000 points.  Scrambling
        (Owen) is used here so that the sequence remains a valid unbiased
        estimator while retaining the low-discrepancy property.

    Correlation is imposed by the Cholesky factor ``C`` of ``V``: if ``Z`` has
    iid standard normal rows then ``m + Z @ C.T`` has mean ``m`` and covariance
    ``C C^T = V``.

    Parameters
    ----------
    n_scenarios
        Number of scenarios ``q`` to draw.  For ``engine="sobol"`` the value is
        rounded **up** to the next power of two, since balance properties of a
        Sobol sequence only hold on such prefixes.
    mean
        Mean return vector, shape ``(n,)``.
    covariance
        Covariance matrix, shape ``(n, n)``; must be positive definite.
    engine
        ``"sobol"`` or ``"pseudo"``.
    seed
        Seed for the scrambling / pseudo-random stream, for reproducibility.

    Returns
    -------
    FloatArray
        Scenario return matrix ``Y`` of shape ``(q, n)``.

    Raises
    ------
    ValueError
        For a non-positive scenario count, a shape mismatch, an unknown engine,
        or a covariance matrix that is not positive definite.
    """
    m = np.asarray(mean, dtype=np.float64).ravel()
    v = np.asarray(covariance, dtype=np.float64)
    n = m.size
    if n_scenarios <= 0:
        raise ValueError(f"'n_scenarios' must be positive, got {n_scenarios}.")
    if v.shape != (n, n):
        raise ValueError(f"'covariance' must be ({n}, {n}), got {v.shape}.")

    try:
        chol = np.linalg.cholesky(v)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
        raise ValueError("'covariance' must be positive definite.") from exc

    if engine == "pseudo":
        rng = np.random.default_rng(seed)
        standard = rng.standard_normal(size=(n_scenarios, n))
    elif engine == "sobol":
        power = int(np.ceil(np.log2(n_scenarios)))
        sampler = qmc.Sobol(d=n, scramble=True, seed=seed)
        uniforms = sampler.random_base2(m=power)
        # Guard the inverse CDF against exact 0/1, which map to +/-inf.
        eps = np.finfo(np.float64).eps
        standard = norm.ppf(np.clip(uniforms, eps, 1.0 - eps))
    else:
        raise ValueError(f"Unknown engine {engine!r}; expected 'sobol' or 'pseudo'.")

    return m + standard @ chol.T
