"""Core tail-risk metrics and the Rockafellar-Uryasev CVaR optimisation programme.

This module is the mathematical foundation of the risk engine.  It implements,
in order:

1.  **Historical (empirical) VaR and CVaR** estimated from a finite scenario set.
2.  **Parametric VaR and CVaR** under Gaussian and Student-t loss distributions.
3.  **The Rockafellar-Uryasev (2000) linear programme**, which minimises
    :math:`\\beta`-CVaR and recovers :math:`\\beta`-VaR *simultaneously* as the
    auxiliary variable :math:`\\alpha^{*}`.

--------------------------------------------------------------------------------
Mathematical framework  (Rockafellar & Uryasev, *Optimization of Conditional
Value-at-Risk*, Journal of Risk 2(3), 2000)
--------------------------------------------------------------------------------

Let ``x`` be the decision vector (portfolio positions) and ``y`` the random
vector of market outcomes.  The loss is ``f(x, y)``.  For the classical
portfolio problem the paper takes, in equation (12),

.. math::

    f(x, y) \\;=\\; -[x_1 y_1 + \\dots + x_n y_n] \\;=\\; -x^{T} y ,

i.e. the loss is the *negative* of the portfolio return.  **Losses are positive
numbers throughout this module.**  A profit is a negative loss.

The cumulative loss distribution is :math:`\\Psi(x, \\alpha) = P[f(x,y) \\le \\alpha]`
(eq. 1).  For a confidence level :math:`\\beta \\in (0, 1)`:

.. math::

    \\alpha_\\beta(x) &= \\min\\{\\alpha : \\Psi(x, \\alpha) \\ge \\beta\\}
        &&\\quad \\text{(eq. 2, } \\beta\\text{-VaR)} \\\\
    \\phi_\\beta(x)   &= (1-\\beta)^{-1} \\int_{f(x,y) \\ge \\alpha_\\beta(x)}
                        f(x, y)\\, p(y)\\, dy
        &&\\quad \\text{(eq. 3, } \\beta\\text{-CVaR)}

The central device of the paper is the auxiliary function (eq. 4)

.. math::

    F_\\beta(x, \\alpha) \\;=\\; \\alpha \\;+\\; (1-\\beta)^{-1}
        \\int_{y} [f(x, y) - \\alpha]^{+} \\, p(y)\\, dy ,

which is **convex and continuously differentiable in** :math:`\\alpha`.
Theorem 1 states

.. math::

    \\phi_\\beta(x) = \\min_{\\alpha} F_\\beta(x, \\alpha), \\qquad
    \\alpha_\\beta(x) = \\text{left endpoint of } \\arg\\min_{\\alpha} F_\\beta(x, \\alpha),

so CVaR is obtained *without* first computing VaR, and VaR falls out as a
by-product.  Theorem 2 states that minimising :math:`\\phi_\\beta(x)` over
:math:`x \\in X` is equivalent to the *joint* minimisation of
:math:`F_\\beta(x, \\alpha)` over :math:`X \\times \\mathbb{R}`.  Because
:math:`f(x, y) = -x^{T}y` is linear (hence convex) in ``x``, and ``X`` is
polyhedral, the joint problem is convex - in fact, after scenario sampling, a
*linear programme*.

Sampling :math:`y^1, \\dots, y^q` from :math:`p(y)` gives the estimator (eq. 17)

.. math::

    \\tilde{F}_\\beta(x, \\alpha) \\;=\\; \\alpha \\;+\\;
        \\frac{1}{q(1-\\beta)} \\sum_{k=1}^{q} \\big[-x^{T} y^{k} - \\alpha\\big]^{+},

piecewise linear and convex in :math:`(x, \\alpha)`.  Introducing auxiliary
variables :math:`u_k` to linearise the positive part yields the LP given in the
paper immediately after (17) - implemented here by :class:`CVaRLinearProgram`.

References
----------
Rockafellar, R.T. and Uryasev, S. (2000).  *Optimization of Conditional
Value-at-Risk*.  Journal of Risk, 2(3), 21-41.

Rockafellar, R.T. and Uryasev, S. (2002).  *Conditional value-at-risk for
general loss distributions*.  Journal of Banking & Finance, 26(7), 1443-1471.
(Source of the exact discrete-atom CVaR decomposition used in
:func:`historical_var_cvar`.)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, Sequence

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.optimize import OptimizeResult, linprog
from scipy.stats import norm, t as student_t

__all__ = [
    "QuantileConvention",
    "LossDistribution",
    "RiskEstimate",
    "PortfolioConstraints",
    "CVaRSolution",
    "RiskEstimator",
    "HistoricalRiskEstimator",
    "ParametricRiskEstimator",
    "CVaRLinearProgram",
    "portfolio_loss_scenarios",
    "ru_auxiliary_function",
    "historical_var",
    "historical_cvar",
    "historical_var_cvar",
    "historical_var_cvar_batch",
    "parametric_var",
    "parametric_cvar",
    "parametric_var_cvar",
    "cvar_efficient_frontier",
]

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

#: Below this many scenarios in the tail the empirical estimator is dominated by
#: sampling noise.  ``q * (1 - beta)`` is the *expected* number of observations
#: strictly beyond VaR, i.e. the effective sample size of the tail average.
MIN_EFFECTIVE_TAIL_SIZE: Final[float] = 10.0

#: Absolute tolerance used when checking that solver output satisfies the
#: budget / sign constraints to within LP feasibility tolerances.
_FEASIBILITY_ATOL: Final[float] = 1e-8


# ==============================================================================
# Enumerations
# ==============================================================================


class QuantileConvention(str, Enum):
    """Estimator used for the empirical :math:`\\beta`-quantile of the losses.

    Attributes
    ----------
    ORDER_STATISTIC
        The *left-continuous inverse* of the empirical CDF,
        :math:`\\hat{\\alpha}_\\beta = L_{(\\lceil \\beta q \\rceil)}`, where
        :math:`L_{(1)} \\le \\dots \\le L_{(q)}` are the sorted losses.  This is
        the definition in eq. (2) applied to the empirical measure: it is the
        smallest sample point whose empirical CDF value is at least
        :math:`\\beta`.  It is the convention consistent with the LP of
        Theorem 2 and is therefore the default.
    INTERPOLATED
        Linear interpolation between adjacent order statistics
        (``numpy.quantile(..., method="linear")``).  Smoother in ``beta`` and
        commonly used for reporting, but it is *not* an attainable value of the
        loss and is not the minimiser of :math:`\\tilde{F}_\\beta`.
    """

    ORDER_STATISTIC = "order_statistic"
    INTERPOLATED = "interpolated"


class LossDistribution(str, Enum):
    """Parametric family assumed for the portfolio loss distribution."""

    NORMAL = "normal"
    STUDENT_T = "student_t"


# ==============================================================================
# Input validation helpers
# ==============================================================================


def _as_1d(array: npt.ArrayLike, name: str) -> FloatArray:
    """Coerce ``array`` to a finite 1-D float64 vector.

    Parameters
    ----------
    array
        Anything array-like.
    name
        Argument name, used to build informative error messages.

    Returns
    -------
    FloatArray
        A contiguous 1-D ``float64`` copy.

    Raises
    ------
    ValueError
        If the input is not 1-D, is empty, or contains NaN/inf.
    """
    values = np.asarray(array, dtype=np.float64).ravel() if np.ndim(array) <= 1 else None
    if values is None:
        raise ValueError(f"'{name}' must be 1-dimensional, got ndim={np.ndim(array)}.")
    if values.size == 0:
        raise ValueError(f"'{name}' must not be empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"'{name}' contains NaN or infinite values.")
    return values


def _as_2d(array: npt.ArrayLike, name: str) -> FloatArray:
    """Coerce ``array`` to a finite 2-D float64 matrix of shape ``(q, n)``.

    Raises
    ------
    ValueError
        If the input is not 2-D, has a zero dimension, or is not finite.
    """
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"'{name}' must be 2-dimensional (q, n), got ndim={values.ndim}.")
    if values.size == 0:
        raise ValueError(f"'{name}' must not be empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"'{name}' contains NaN or infinite values.")
    return values


def _validate_beta(beta: float) -> float:
    """Validate the confidence level :math:`\\beta \\in (0, 1)`.

    The paper considers ``beta`` in {0.90, 0.95, 0.99}; any interior value is
    admissible.  ``beta = 0`` degenerates CVaR to the unconditional mean loss
    and ``beta = 1`` makes :math:`(1-\\beta)^{-1}` undefined, so both endpoints
    are rejected.
    """
    beta = float(beta)
    if not np.isfinite(beta) or not (0.0 < beta < 1.0):
        raise ValueError(f"'beta' must lie strictly inside (0, 1), got {beta!r}.")
    return beta


def _warn_on_thin_tail(n_scenarios: int, beta: float) -> float:
    """Return the effective tail sample size and log a warning if it is small.

    The CVaR estimator averages roughly ``q * (1 - beta)`` observations.  With
    ``beta = 0.99`` and ``q = 500`` that is five points: the estimate is then
    driven by a handful of extreme draws and its standard error is large.  This
    is exactly the sampling-noise effect the paper observes in Table 5, where
    pseudo-random Monte Carlo converges slowly compared with the quasi-random
    Sobol sequence of Table 6.
    """
    effective = n_scenarios * (1.0 - beta)
    if effective < MIN_EFFECTIVE_TAIL_SIZE:
        LOGGER.warning(
            "Thin tail: q*(1-beta) = %.2f < %.0f (q=%d, beta=%.4f). "
            "CVaR/VaR estimates will carry substantial sampling error; "
            "increase the scenario count or lower beta.",
            effective,
            MIN_EFFECTIVE_TAIL_SIZE,
            n_scenarios,
            beta,
        )
    return effective


# ==============================================================================
# Result containers
# ==============================================================================


@dataclass(frozen=True)
class RiskEstimate:
    """Immutable container for a single VaR/CVaR estimate.

    Attributes
    ----------
    beta
        Confidence level of the estimate.
    var
        :math:`\\beta`-VaR, eq. (2).  Expressed as a **positive loss** in the
        same units as the input (percentage return, or currency).
    cvar
        :math:`\\beta`-CVaR, eq. (3).  Always ``>= var`` by construction.
    cvar_plus
        "Upper CVaR" - the conditional expectation of losses *strictly* greater
        than VaR.  For continuous distributions ``cvar == cvar_plus``; for a
        finite scenario set they differ whenever the VaR atom carries mass.
        ``nan`` when no scenario exceeds VaR.
    var_weight
        The weight :math:`\\lambda \\in [0, 1]` in the exact decomposition
        ``cvar = lambda * var + (1 - lambda) * cvar_plus`` (Rockafellar &
        Uryasev 2002).  ``lambda = (Psi(x, VaR) - beta) / (1 - beta)`` is the
        probability mass of the VaR atom that must be split to make the tail
        probability exactly ``1 - beta``.  ``nan`` for parametric estimates.
    mean_loss
        Unconditional expected loss :math:`\\mu(x) = -x^{T}m`, eq. (14).
    n_scenarios
        Number of scenarios behind the estimate; ``None`` for parametric.
    effective_tail_size
        ``n_scenarios * (1 - beta)``; ``None`` for parametric estimates.
    method
        Free-form provenance label, e.g. ``"historical/order_statistic"``.
    """

    beta: float
    var: float
    cvar: float
    cvar_plus: float = float("nan")
    var_weight: float = float("nan")
    mean_loss: float = float("nan")
    n_scenarios: int | None = None
    effective_tail_size: float | None = None
    method: str = ""

    @property
    def expected_return(self) -> float:
        """Expected portfolio *return*, i.e. the negative of the mean loss."""
        return -self.mean_loss

    @property
    def excess_cvar(self) -> float:
        """``cvar - var``: the mean severity of losses beyond the VaR threshold."""
        return self.cvar - self.var

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"[{self.method or 'risk'}] beta={self.beta:.4f}  "
            f"VaR={self.var:.6f}  CVaR={self.cvar:.6f}  "
            f"E[loss]={self.mean_loss:.6f}"
        )


@dataclass(frozen=True)
class CVaRSolution:
    """Outcome of the Rockafellar-Uryasev CVaR minimisation.

    Attributes
    ----------
    weights
        Optimal portfolio :math:`x^{*}`.
    var
        :math:`\\alpha^{*}`, the optimal auxiliary variable.  By Theorem 2 this
        is a member of :math:`A_\\beta(x^{*})`, the argmin interval, and hence
        the :math:`\\beta`-VaR of the optimal portfolio.  See
        :attr:`var_is_left_endpoint` for the caveat about degenerate optima.
    cvar
        :math:`\\tilde{F}_\\beta(x^{*}, \\alpha^{*})` - the optimal
        :math:`\\beta`-CVaR (eq. 10).
    expected_return
        :math:`\\bar{y}^{T}x^{*}`, the in-sample expected portfolio return.
    empirical
        VaR/CVaR of :math:`x^{*}` recomputed directly from the scenario losses
        by :func:`historical_var_cvar`.  This is an independent cross-check of
        the LP: ``empirical.cvar`` must agree with :attr:`cvar` to solver
        tolerance, and ``empirical.var`` is the *left endpoint* of the argmin
        interval even when the LP returns an interior vertex.
    var_is_left_endpoint
        ``True`` when :math:`\\alpha^{*}` from the LP coincides with the
        empirical VaR.  When ``False`` the argmin interval :math:`A_\\beta(x^{*})`
        is non-degenerate ("flat spot" in the loss CDF, eq. 6-7) and
        ``empirical.var`` should be reported as the VaR.
    success, status, message, n_iterations
        Solver diagnostics propagated from :func:`scipy.optimize.linprog`.
    beta, n_scenarios, n_assets
        Problem dimensions, retained for audit trails.
    tail_slacks
        The solver's auxiliary variables ``u_1 ... u_q``, shape ``(q,)``.
        Exposed so that the LP's primal feasibility can be audited directly
        against the two constraints printed after eq. (17) - ``u_k >= 0`` and
        ``x'y_k + alpha + u_k >= 0`` - rather than inferred from the objective
        value.  At the optimum each ``u_k`` equals the positive part
        ``max(f(x, y_k) - alpha, 0)`` exactly, which is what makes the
        linearisation tight.
    """

    weights: FloatArray
    var: float
    cvar: float
    expected_return: float
    empirical: RiskEstimate
    var_is_left_endpoint: bool
    success: bool
    status: int
    message: str
    n_iterations: int
    beta: float
    n_scenarios: int
    n_assets: int
    tail_slacks: FloatArray

    @property
    def reported_var(self) -> float:
        """VaR to publish: the left endpoint of :math:`A_\\beta(x^{*})` (eq. 7)."""
        return self.empirical.var

    def __str__(self) -> str:  # pragma: no cover - presentation only
        weights = np.array2string(self.weights, precision=6, suppress_small=True)
        return (
            f"CVaRSolution(beta={self.beta:.4f}, success={self.success})\n"
            f"  weights = {weights}\n"
            f"  VaR     = {self.reported_var:.6f}\n"
            f"  CVaR    = {self.cvar:.6f}\n"
            f"  E[R]    = {self.expected_return:.6f}\n"
            f"  solver  = {self.message}"
        )


@dataclass
class PortfolioConstraints:
    """Polyhedral feasible set :math:`X` for the decision vector ``x``.

    The baseline set of the paper is eq. (16): ``X = {x : (11) and (15)}`` with

    * eq. (11) - budget and no short sales: :math:`x_j \\ge 0`,
      :math:`\\sum_j x_j = 1`;
    * eq. (15) - minimum expected return: :math:`\\mu(x) = -x^{T}m \\le -R`.

    Attributes
    ----------
    budget
        Right-hand side of the budget equality :math:`\\sum_j x_j = \\text{budget}`.
        Set to ``None`` to drop the equality entirely (e.g. zero-net-investment
        hedging books, where the percentage-return interpretation of Section 3
        no longer applies).
    lower_bounds, upper_bounds
        Per-asset box constraints.  Scalars are broadcast to all ``n`` assets.
        ``lower_bounds = 0.0`` reproduces the no-short-sale condition of
        eq. (11); the hedging application of Section 4 instead uses
        :math:`-|z_j| \\le x_j \\le |z_j|` (eq. 20), which is expressed by
        passing vectors.  ``None`` means unbounded on that side.
    min_expected_return
        The scalar ``R`` of eq. (15).  ``None`` disables the constraint, in
        which case the LP returns the global minimum-CVaR portfolio.
    linear_inequalities
        Optional extra constraints ``(A, b)`` imposing ``A @ x <= b`` - sector
        caps, turnover budgets, factor-exposure limits.  ``A`` has shape
        ``(m, n)`` and ``b`` shape ``(m,)``.
    linear_equalities
        Optional extra constraints ``(A, b)`` imposing ``A @ x == b``.
    """

    budget: float | None = 1.0
    lower_bounds: float | npt.ArrayLike | None = 0.0
    upper_bounds: float | npt.ArrayLike | None = None
    min_expected_return: float | None = None
    linear_inequalities: tuple[npt.ArrayLike, npt.ArrayLike] | None = None
    linear_equalities: tuple[npt.ArrayLike, npt.ArrayLike] | None = None

    def bounds_for(self, n_assets: int) -> list[tuple[float | None, float | None]]:
        """Materialise the per-asset box constraints as a ``linprog`` bounds list.

        Parameters
        ----------
        n_assets
            Dimension ``n`` of the decision vector.

        Returns
        -------
        list of (float or None, float or None)
            One ``(low, high)`` pair per asset.

        Raises
        ------
        ValueError
            If a bound vector has the wrong length or if any lower bound
            exceeds the corresponding upper bound (empty feasible box).
        """
        lower = self._broadcast(self.lower_bounds, n_assets, "lower_bounds")
        upper = self._broadcast(self.upper_bounds, n_assets, "upper_bounds")
        pairs: list[tuple[float | None, float | None]] = []
        for j in range(n_assets):
            low = None if lower is None else float(lower[j])
            high = None if upper is None else float(upper[j])
            if low is not None and high is not None and low > high:
                raise ValueError(
                    f"Infeasible box for asset {j}: lower={low} > upper={high}."
                )
            pairs.append((low, high))
        return pairs

    @staticmethod
    def _broadcast(
        value: float | npt.ArrayLike | None, n_assets: int, name: str
    ) -> FloatArray | None:
        """Broadcast a scalar or validate a vector bound to length ``n_assets``."""
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(n_assets, float(arr))
        arr = arr.ravel()
        if arr.size != n_assets:
            raise ValueError(
                f"'{name}' has length {arr.size}, expected {n_assets} (one per asset)."
            )
        return arr


# ==============================================================================
# Loss construction and the auxiliary function F_beta
# ==============================================================================


def portfolio_loss_scenarios(weights: npt.ArrayLike, returns: npt.ArrayLike) -> FloatArray:
    """Map scenario returns to portfolio losses via eq. (12): ``f(x, y) = -x^T y``.

    Parameters
    ----------
    weights
        Portfolio positions ``x``, shape ``(n,)``.
    returns
        Scenario return matrix ``Y``, shape ``(q, n)``; row ``k`` is the return
        vector :math:`y^{k}` of scenario ``k``.

    Returns
    -------
    FloatArray
        Loss vector of shape ``(q,)`` with ``L[k] = -x^T y^k``.  Positive
        entries are losses, negative entries are gains.

    Raises
    ------
    ValueError
        If the shapes are inconsistent or the inputs are not finite.

    Examples
    --------
    >>> portfolio_loss_scenarios([0.5, 0.5], [[0.02, -0.04], [0.10, 0.10]])
    array([ 0.01, -0.1 ])
    """
    x = _as_1d(weights, "weights")
    y = _as_2d(returns, "returns")
    if y.shape[1] != x.size:
        raise ValueError(
            f"Shape mismatch: 'returns' has {y.shape[1]} columns but "
            f"'weights' has {x.size} entries."
        )
    return -(y @ x)


def ru_auxiliary_function(
    losses: npt.ArrayLike, alpha: float | npt.ArrayLike, beta: float
) -> float | FloatArray:
    """Evaluate the Rockafellar-Uryasev auxiliary function, eq. (9)/(17).

    .. math::

        \\tilde{F}_\\beta(x, \\alpha) \\;=\\; \\alpha \\;+\\;
            \\frac{1}{q(1-\\beta)} \\sum_{k=1}^{q} \\big[f(x, y^{k}) - \\alpha\\big]^{+}

    Convex and piecewise linear in ``alpha`` with kinks at the sample losses.
    Its minimum value is CVaR and its argmin interval contains VaR (Theorem 1),
    a property exercised directly by the unit tests.

    Parameters
    ----------
    losses
        Realised losses :math:`f(x, y^{k})` for a *fixed* portfolio, shape ``(q,)``.
    alpha
        Threshold(s) at which to evaluate; scalar or array of any shape.
    beta
        Confidence level in ``(0, 1)``.

    Returns
    -------
    float or FloatArray
        ``F_beta`` evaluated at each ``alpha``; scalar in, scalar out.
    """
    loss = _as_1d(losses, "losses")
    beta = _validate_beta(beta)
    a = np.asarray(alpha, dtype=np.float64)
    # Broadcast losses (q,) against alpha (...,) -> excess matrix (..., q).
    excess = np.maximum(loss - a[..., np.newaxis], 0.0)
    value = a + excess.sum(axis=-1) / (loss.size * (1.0 - beta))
    return float(value) if value.ndim == 0 else value


# ==============================================================================
# Historical (empirical / scenario-based) estimators
# ==============================================================================


def historical_var(
    losses: npt.ArrayLike,
    beta: float,
    convention: QuantileConvention = QuantileConvention.ORDER_STATISTIC,
) -> float:
    """Empirical :math:`\\beta`-VaR of a loss sample, eq. (2).

    The order-statistic convention returns
    :math:`L_{(\\lceil \\beta q \\rceil)}`, the smallest observed loss whose
    empirical CDF value reaches :math:`\\beta`.  This is the exact empirical
    analogue of eq. (2) and the value that minimises
    :func:`ru_auxiliary_function`, hence the default.

    Parameters
    ----------
    losses
        Loss sample, shape ``(q,)``.  Positive values are losses.
    beta
        Confidence level in ``(0, 1)``.
    convention
        Quantile estimator; see :class:`QuantileConvention`.

    Returns
    -------
    float
        The :math:`\\beta`-VaR.

    Examples
    --------
    >>> float(historical_var([0.0, 1.0, 2.0, 3.0, 10.0], 0.80))
    3.0
    """
    loss = _as_1d(losses, "losses")
    beta = _validate_beta(beta)
    if convention is QuantileConvention.INTERPOLATED:
        return float(np.quantile(loss, beta, method="linear"))
    # ceil(beta * q) is 1-based; clip guards beta*q < 1 and floating-point
    # values of beta*q that land microscopically above an integer.
    index = int(np.clip(np.ceil(beta * loss.size), 1, loss.size))
    return float(np.partition(loss, index - 1)[index - 1])


def historical_cvar(
    losses: npt.ArrayLike,
    beta: float,
    convention: QuantileConvention = QuantileConvention.ORDER_STATISTIC,
) -> float:
    """Empirical :math:`\\beta`-CVaR of a loss sample.

    Thin wrapper over :func:`historical_var_cvar`; see that function for the
    exact discrete-atom formula.

    Parameters
    ----------
    losses
        Loss sample, shape ``(q,)``.
    beta
        Confidence level in ``(0, 1)``.
    convention
        Quantile estimator used for the VaR threshold.

    Returns
    -------
    float
        The :math:`\\beta`-CVaR, guaranteed ``>=`` the corresponding VaR.
    """
    return historical_var_cvar(losses, beta, convention=convention).cvar


def historical_var_cvar(
    losses: npt.ArrayLike,
    beta: float,
    convention: QuantileConvention = QuantileConvention.ORDER_STATISTIC,
) -> RiskEstimate:
    """Joint empirical VaR/CVaR estimate with the exact discrete-atom formula.

    For a finite scenario set the naive "average of losses above VaR" is
    *biased*: unless :math:`\\beta q` is an integer, the losses strictly beyond
    the VaR order statistic carry probability mass ``(q - k) / q`` which does
    not equal ``1 - beta``.  The estimator used here splits the probability atom
    sitting exactly at VaR so that the tail mass is exactly ``1 - beta``.  With
    :math:`k = \\lceil \\beta q \\rceil` and sorted losses :math:`L_{(1)} \\le
    \\dots \\le L_{(q)}`:

    .. math::

        \\widehat{\\text{CVaR}}_\\beta \\;=\\; \\frac{1}{1-\\beta}
        \\left[ \\left(\\frac{k}{q} - \\beta\\right) L_{(k)}
        \\;+\\; \\frac{1}{q}\\sum_{i=k+1}^{q} L_{(i)} \\right] .

    The two weights sum to exactly :math:`1 - \\beta`.  This value is identical
    to :math:`\\min_\\alpha \\tilde{F}_\\beta(x, \\alpha)` of eq. (17) - which is
    what makes it the estimator consistent with the LP of Theorem 2 - and it
    equals the convex combination
    ``lambda * VaR + (1 - lambda) * CVaR_plus``.

    Parameters
    ----------
    losses
        Loss sample, shape ``(q,)``.
    beta
        Confidence level in ``(0, 1)``.
    convention
        Quantile estimator for the VaR threshold.  ``INTERPOLATED`` keeps the
        exact-mass CVaR definition but reports the interpolated VaR; the two
        are then not tied to the same order statistic, so ``ORDER_STATISTIC``
        should be preferred whenever the estimate feeds an optimiser.

    Returns
    -------
    RiskEstimate
        Populated with VaR, CVaR, upper CVaR, the atom weight ``lambda`` and
        the mean loss.

    Notes
    -----
    Complexity is ``O(q log q)`` (a full sort is required for the tail sum).
    """
    loss = _as_1d(losses, "losses")
    beta = _validate_beta(beta)
    q = loss.size
    effective = _warn_on_thin_tail(q, beta)

    ordered = np.sort(loss)
    k = int(np.clip(np.ceil(beta * q), 1, q))  # 1-based index of the VaR atom
    var_order_stat = float(ordered[k - 1])

    # Probability mass of the VaR atom retained in the tail, so that
    # (k/q - beta) + (q - k)/q == 1 - beta exactly.
    atom_mass = k / q - beta
    tail_sum = float(ordered[k:].sum())  # strictly beyond the VaR order statistic
    cvar = (atom_mass * var_order_stat + tail_sum / q) / (1.0 - beta)

    # Upper CVaR: E[L | L > VaR].  Uses a strict mask so that ties at the VaR
    # level are excluded, per the definition in Rockafellar & Uryasev (2002).
    strictly_above = ordered[ordered > var_order_stat]
    cvar_plus = float(strictly_above.mean()) if strictly_above.size else float("nan")
    # lambda in  CVaR = lambda * VaR + (1 - lambda) * CVaR_plus.
    var_weight = atom_mass / (1.0 - beta)

    var = (
        float(np.quantile(loss, beta, method="linear"))
        if convention is QuantileConvention.INTERPOLATED
        else var_order_stat
    )

    return RiskEstimate(
        beta=beta,
        var=var,
        cvar=float(cvar),
        cvar_plus=cvar_plus,
        var_weight=float(var_weight),
        mean_loss=float(loss.mean()),
        n_scenarios=q,
        effective_tail_size=float(effective),
        method=f"historical/{convention.value}",
    )


def historical_var_cvar_batch(
    loss_matrix: npt.ArrayLike, beta: float
) -> tuple[FloatArray, FloatArray]:
    """Vectorised VaR/CVaR across many portfolios at once.

    Population-based optimisers evaluate CVaR tens of thousands of times, so
    the per-portfolio Python overhead of :func:`historical_var_cvar` dominates.
    This kernel applies the identical exact-atom formula to every column of a
    loss matrix in one pass and is the function NSGA-II calls.

    Two performance notes:

    * ``np.partition`` replaces a full sort.  Only the ``k``-th order statistic
      and the *sum* of everything beyond it are needed, and partitioning
      guarantees all entries after position ``k-1`` are ``>=`` the pivot.  Cost
      drops from ``O(q log q)`` to ``O(q)`` per portfolio.
    * The result is bit-identical to looping :func:`historical_var_cvar`; the
      test suite asserts this rather than trusting it.

    Parameters
    ----------
    loss_matrix
        Shape ``(q, P)``.  Column ``p`` holds the ``q`` scenario losses of
        portfolio ``p`` - i.e. ``L @ X.T`` for a population matrix ``X`` of
        shape ``(P, n)``.
    beta
        Confidence level in ``(0, 1)``.

    Returns
    -------
    (FloatArray, FloatArray)
        ``(var, cvar)``, each of shape ``(P,)``.
    """
    losses = _as_2d(loss_matrix, "loss_matrix")
    beta = _validate_beta(beta)
    q = losses.shape[0]

    k = int(np.clip(np.ceil(beta * q), 1, q))  # 1-based index of the VaR atom
    partitioned = np.partition(losses, k - 1, axis=0)
    var = partitioned[k - 1]  # (P,)
    tail_sum = partitioned[k:].sum(axis=0)  # (P,) - all entries are >= var

    atom_mass = k / q - beta
    cvar = (atom_mass * var + tail_sum / q) / (1.0 - beta)
    return var, cvar


# ==============================================================================
# Parametric estimators
# ==============================================================================


def _student_t_scale(sigma: float, dof: float, standardised: bool) -> float:
    """Convert a standard deviation to the scale parameter of a Student-t.

    If ``standardised`` is ``True`` the caller supplies ``sigma`` as the
    *standard deviation* of the loss, and the t-scale is
    :math:`c = \\sigma \\sqrt{(\\nu - 2)/\\nu}` (finite only for
    :math:`\\nu > 2`).  Otherwise ``sigma`` is already the scale parameter.
    """
    if not standardised:
        return sigma
    if dof <= 2.0:
        raise ValueError(
            f"Student-t variance is infinite for dof <= 2 (got dof={dof}); "
            "pass standardised=False and supply the scale parameter directly."
        )
    return sigma * float(np.sqrt((dof - 2.0) / dof))


def parametric_var(
    mean_loss: float,
    volatility: float,
    beta: float,
    distribution: LossDistribution = LossDistribution.NORMAL,
    dof: float | None = None,
    standardised: bool = True,
) -> float:
    """Closed-form :math:`\\beta`-VaR of a location-scale loss distribution.

    Under normality the paper's eq. (18) gives

    .. math::

        \\alpha_\\beta(x) = \\mu(x) + c_1(\\beta)\\,\\sigma(x), \\qquad
        c_1(\\beta) = \\sqrt{2}\\,\\mathrm{erf}^{-1}(2\\beta - 1) = \\Phi^{-1}(\\beta),

    with :math:`\\mu(x) = -x^{T}m` and :math:`\\sigma^{2}(x) = x^{T}Vx`
    (eq. 14).  The Student-t case replaces :math:`\\Phi^{-1}` by the
    t-quantile and rescales, capturing the fat tails that make the Gaussian
    assumption unsafe for the option books of Section 4.

    Parameters
    ----------
    mean_loss
        :math:`\\mu(x)`, the expected **loss** (negative of expected return).
    volatility
        :math:`\\sigma(x) > 0`, the loss standard deviation (or the t-scale
        when ``standardised=False``).
    beta
        Confidence level in ``(0, 1)``.
    distribution
        Parametric family; see :class:`LossDistribution`.
    dof
        Degrees of freedom :math:`\\nu`, required for ``STUDENT_T``.
    standardised
        Whether ``volatility`` is a standard deviation (``True``) or the raw
        scale parameter of the t (``False``).  Ignored for the normal case.

    Returns
    -------
    float
        The :math:`\\beta`-VaR.
    """
    beta = _validate_beta(beta)
    mean_loss = float(mean_loss)
    volatility = float(volatility)
    if not np.isfinite(volatility) or volatility < 0.0:
        raise ValueError(f"'volatility' must be finite and non-negative, got {volatility!r}.")

    if distribution is LossDistribution.NORMAL:
        return mean_loss + volatility * float(norm.ppf(beta))

    if dof is None:
        raise ValueError("'dof' is required for the Student-t distribution.")
    dof = float(dof)
    if dof <= 1.0:
        raise ValueError(f"'dof' must exceed 1 for a finite mean, got {dof!r}.")
    scale = _student_t_scale(volatility, dof, standardised)
    return mean_loss + scale * float(student_t.ppf(beta, dof))


def parametric_cvar(
    mean_loss: float,
    volatility: float,
    beta: float,
    distribution: LossDistribution = LossDistribution.NORMAL,
    dof: float | None = None,
    standardised: bool = True,
) -> float:
    """Closed-form :math:`\\beta`-CVaR of a location-scale loss distribution.

    Under normality the paper's eq. (19) gives

    .. math::

        \\phi_\\beta(x) = \\mu(x) + c_2(\\beta)\\,\\sigma(x), \\qquad
        c_2(\\beta) = \\frac{\\varphi\\!\\left(\\Phi^{-1}(\\beta)\\right)}{1-\\beta}
        = \\Big(\\sqrt{2\\pi}\\,
          \\exp\\!\\big[(\\mathrm{erf}^{-1}(2\\beta-1))^{2}\\big](1-\\beta)\\Big)^{-1},

    the two expressions being algebraically identical because
    :math:`\\Phi^{-1}(\\beta) = \\sqrt{2}\\,\\mathrm{erf}^{-1}(2\\beta-1)`.

    For :math:`T \\sim t_\\nu` with :math:`\\nu > 1` the standard result is

    .. math::

        \\mathrm{ES}_\\beta(T) = \\frac{f_\\nu(t_\\beta)}{1-\\beta}
            \\cdot \\frac{\\nu + t_\\beta^{2}}{\\nu - 1},
        \\qquad t_\\beta = F_\\nu^{-1}(\\beta),

    which is then affinely mapped by the location and scale.

    Parameters
    ----------
    See :func:`parametric_var`.

    Returns
    -------
    float
        The :math:`\\beta`-CVaR; always ``>=`` the corresponding VaR.
    """
    beta = _validate_beta(beta)
    mean_loss = float(mean_loss)
    volatility = float(volatility)
    if not np.isfinite(volatility) or volatility < 0.0:
        raise ValueError(f"'volatility' must be finite and non-negative, got {volatility!r}.")

    if distribution is LossDistribution.NORMAL:
        z = float(norm.ppf(beta))
        c2 = float(norm.pdf(z)) / (1.0 - beta)
        return mean_loss + volatility * c2

    if dof is None:
        raise ValueError("'dof' is required for the Student-t distribution.")
    dof = float(dof)
    if dof <= 1.0:
        raise ValueError(f"'dof' must exceed 1 for a finite tail expectation, got {dof!r}.")
    scale = _student_t_scale(volatility, dof, standardised)
    t_beta = float(student_t.ppf(beta, dof))
    es_standard = (
        float(student_t.pdf(t_beta, dof)) / (1.0 - beta) * (dof + t_beta**2) / (dof - 1.0)
    )
    return mean_loss + scale * es_standard


def parametric_var_cvar(
    mean_loss: float,
    volatility: float,
    beta: float,
    distribution: LossDistribution = LossDistribution.NORMAL,
    dof: float | None = None,
    standardised: bool = True,
) -> RiskEstimate:
    """Joint closed-form VaR/CVaR estimate.

    Parameters
    ----------
    See :func:`parametric_var`.

    Returns
    -------
    RiskEstimate
        ``cvar_plus`` equals ``cvar`` and ``var_weight`` is ``0`` because a
        continuous distribution places no probability atom at VaR.
    """
    kwargs = dict(
        beta=beta, distribution=distribution, dof=dof, standardised=standardised
    )
    var = parametric_var(mean_loss, volatility, **kwargs)  # type: ignore[arg-type]
    cvar = parametric_cvar(mean_loss, volatility, **kwargs)  # type: ignore[arg-type]
    label = distribution.value + ("" if dof is None else f"(dof={dof:g})")
    return RiskEstimate(
        beta=_validate_beta(beta),
        var=var,
        cvar=cvar,
        cvar_plus=cvar,  # no atom at VaR for a continuous law
        var_weight=0.0,
        mean_loss=float(mean_loss),
        n_scenarios=None,
        effective_tail_size=None,
        method=f"parametric/{label}",
    )


# ==============================================================================
# Object-oriented estimator interface
# ==============================================================================


class RiskEstimator(ABC):
    """Abstract base class for portfolio-level VaR/CVaR estimators.

    Concrete subclasses bind a *market model* (a scenario set, or a mean vector
    and covariance matrix) and expose a uniform
    ``estimate(weights, beta) -> RiskEstimate`` contract.  Downstream modules -
    the multi-objective optimiser and the stress tester - depend only on this
    interface, so a scenario-based engine and an analytic engine are
    interchangeable at the call site (Liskov substitution).
    """

    @property
    @abstractmethod
    def n_assets(self) -> int:
        """Number of instruments the estimator is defined over."""

    @abstractmethod
    def estimate(self, weights: npt.ArrayLike, beta: float) -> RiskEstimate:
        """Return the VaR/CVaR of ``weights`` at confidence level ``beta``."""

    def var(self, weights: npt.ArrayLike, beta: float) -> float:
        """Convenience accessor for :math:`\\beta`-VaR."""
        return self.estimate(weights, beta).var

    def cvar(self, weights: npt.ArrayLike, beta: float) -> float:
        """Convenience accessor for :math:`\\beta`-CVaR."""
        return self.estimate(weights, beta).cvar

    def _check_weights(self, weights: npt.ArrayLike) -> FloatArray:
        """Validate a weight vector against the estimator's asset dimension."""
        x = _as_1d(weights, "weights")
        if x.size != self.n_assets:
            raise ValueError(
                f"'weights' has {x.size} entries but the estimator is defined "
                f"over {self.n_assets} assets."
            )
        return x


class HistoricalRiskEstimator(RiskEstimator):
    """Scenario-based estimator - the discrete measure of eq. (17).

    Holds a return matrix ``Y`` of shape ``(q, n)`` whose rows are the sampled
    return vectors :math:`y^{k}`.  The scenarios may be genuine historical
    observations, a Monte Carlo draw from :math:`p(y)`, or - as in Table 6 of
    the paper - a low-discrepancy Sobol sequence.  The estimator makes no
    distributional assumption whatsoever.

    Parameters
    ----------
    returns
        Scenario return matrix, shape ``(q, n)``.
    convention
        Quantile estimator passed through to :func:`historical_var_cvar`.
    probabilities
        Optional scenario probabilities, shape ``(q,)``.  Reserved for
        importance-sampled or regime-weighted scenario sets; must be
        non-negative and sum to one.  Currently only equal weighting is
        supported by the estimator, so a non-uniform vector raises.
    """

    def __init__(
        self,
        returns: npt.ArrayLike,
        convention: QuantileConvention = QuantileConvention.ORDER_STATISTIC,
        probabilities: npt.ArrayLike | None = None,
    ) -> None:
        self._returns = _as_2d(returns, "returns")
        self._convention = QuantileConvention(convention)
        if probabilities is not None:
            probs = _as_1d(probabilities, "probabilities")
            if probs.size != self._returns.shape[0]:
                raise ValueError("'probabilities' must have one entry per scenario.")
            if np.any(probs < 0.0) or not np.isclose(probs.sum(), 1.0):
                raise ValueError("'probabilities' must be non-negative and sum to 1.")
            if not np.allclose(probs, 1.0 / probs.size):
                raise NotImplementedError(
                    "Non-uniform scenario probabilities are not yet supported; "
                    "resample the scenario set to an equally weighted one."
                )

    @property
    def returns(self) -> FloatArray:
        """Read-only view of the scenario return matrix, shape ``(q, n)``."""
        view = self._returns.view()
        view.flags.writeable = False
        return view

    @property
    def n_scenarios(self) -> int:
        """Number of scenarios ``q``."""
        return self._returns.shape[0]

    @property
    def n_assets(self) -> int:
        """Number of instruments ``n``."""
        return self._returns.shape[1]

    @property
    def expected_returns(self) -> FloatArray:
        """Sample mean return vector :math:`\\bar{y}`, shape ``(n,)``."""
        return self._returns.mean(axis=0)

    @property
    def covariance(self) -> FloatArray:
        """Sample covariance matrix ``V`` (unbiased), shape ``(n, n)``."""
        return np.cov(self._returns, rowvar=False, ddof=1)

    def losses(self, weights: npt.ArrayLike) -> FloatArray:
        """Scenario losses ``f(x, y^k) = -x^T y^k`` for the given weights."""
        return portfolio_loss_scenarios(self._check_weights(weights), self._returns)

    def estimate(self, weights: npt.ArrayLike, beta: float) -> RiskEstimate:
        """Empirical VaR/CVaR of ``weights``; see :func:`historical_var_cvar`."""
        return historical_var_cvar(self.losses(weights), beta, convention=self._convention)


class ParametricRiskEstimator(RiskEstimator):
    """Analytic estimator under a location-scale return law.

    Implements eq. (14) - :math:`\\mu(x) = -x^{T}m`,
    :math:`\\sigma^{2}(x) = x^{T}Vx` - followed by the closed forms (18)-(19).
    Under normality with :math:`\\beta \\ge 0.5`, the paper's Proposition shows
    that minimising VaR, CVaR or variance subject to an *active* return
    constraint yields the **same** optimal portfolio.  That equivalence is what
    makes this class a valid benchmark for the LP: see the regression tests,
    which reproduce Tables 3 and 4.

    Parameters
    ----------
    expected_returns
        Mean return vector ``m``, shape ``(n,)``.
    covariance
        Covariance matrix ``V``, shape ``(n, n)``; must be symmetric and
        positive semi-definite.
    distribution
        Parametric family for the loss; see :class:`LossDistribution`.
    dof
        Degrees of freedom for ``STUDENT_T``.
    """

    def __init__(
        self,
        expected_returns: npt.ArrayLike,
        covariance: npt.ArrayLike,
        distribution: LossDistribution = LossDistribution.NORMAL,
        dof: float | None = None,
    ) -> None:
        self._mean = _as_1d(expected_returns, "expected_returns")
        cov = _as_2d(covariance, "covariance")
        n = self._mean.size
        if cov.shape != (n, n):
            raise ValueError(
                f"'covariance' must be ({n}, {n}) to match 'expected_returns', "
                f"got {cov.shape}."
            )
        if not np.allclose(cov, cov.T, atol=1e-10):
            raise ValueError("'covariance' must be symmetric.")
        # Positive semi-definiteness: a small negative eigenvalue is numerical
        # noise from an estimated matrix, a large one means the input is wrong.
        min_eigenvalue = float(np.linalg.eigvalsh(cov).min())
        if min_eigenvalue < -1e-10 * max(1.0, float(np.abs(cov).max())):
            raise ValueError(
                f"'covariance' is not positive semi-definite "
                f"(minimum eigenvalue {min_eigenvalue:.3e})."
            )
        self._covariance = cov
        self._distribution = LossDistribution(distribution)
        self._dof = dof
        if self._distribution is LossDistribution.STUDENT_T and dof is None:
            raise ValueError("'dof' is required when distribution is STUDENT_T.")

    @property
    def n_assets(self) -> int:
        """Number of instruments ``n``."""
        return self._mean.size

    @property
    def expected_returns(self) -> FloatArray:
        """Mean return vector ``m``."""
        return self._mean

    @property
    def covariance(self) -> FloatArray:
        """Covariance matrix ``V``."""
        return self._covariance

    def mean_loss(self, weights: npt.ArrayLike) -> float:
        """:math:`\\mu(x) = -x^{T}m`, eq. (14)."""
        return float(-self._check_weights(weights) @ self._mean)

    def volatility(self, weights: npt.ArrayLike) -> float:
        """:math:`\\sigma(x) = \\sqrt{x^{T}Vx}`, eq. (14).

        The quadratic form is clipped at zero before the square root: for a
        PSD covariance the true value is non-negative, and a tiny negative
        result is floating-point cancellation, not a modelling error.
        """
        x = self._check_weights(weights)
        variance = float(x @ self._covariance @ x)
        return float(np.sqrt(max(variance, 0.0)))

    def estimate(self, weights: npt.ArrayLike, beta: float) -> RiskEstimate:
        """Closed-form VaR/CVaR of ``weights`` from eqs. (18)-(19)."""
        return parametric_var_cvar(
            mean_loss=self.mean_loss(weights),
            volatility=self.volatility(weights),
            beta=beta,
            distribution=self._distribution,
            dof=self._dof,
        )


# ==============================================================================
# The Rockafellar-Uryasev linear programme
# ==============================================================================


class CVaRLinearProgram:
    """Minimise :math:`\\beta`-CVaR by linear programming (Theorem 2).

    Decision vector
    ---------------
    The LP variable is the stacked vector

    .. math::

        z \\;=\\; \\big(\\underbrace{x_1, \\dots, x_n}_{\\text{positions}},\\;
                    \\underbrace{\\alpha}_{\\text{VaR}},\\;
                    \\underbrace{u_1, \\dots, u_q}_{\\text{tail slacks}}\\big)
        \\;\\in\\; \\mathbb{R}^{n + 1 + q}.

    Objective
    ---------
    Directly from eq. (17), minimise

    .. math::

        \\alpha \\;+\\; \\frac{1}{q(1-\\beta)}\\sum_{k=1}^{q} u_k .

    When ``rescale_objective`` is set, the paper's remark after Theorem 1 is
    applied: the equivalent objective
    :math:`(1-\\beta)\\alpha + q^{-1}\\sum_k u_k` is minimised instead.  This
    avoids the large factor :math:`(q(1-\\beta))^{-1}` when :math:`1-\\beta` is
    small, improving the conditioning of the constraint/objective scaling; the
    optimal value is divided by :math:`(1-\\beta)` afterwards to recover CVaR.
    The argmin is unchanged because the two objectives differ by a positive
    multiplicative constant.

    Constraints
    -----------
    ``u_k`` must dominate the positive part :math:`[f(x, y^k) - \\alpha]^{+}`.
    Since ``u_k`` is minimised and bounded below by zero, it is enough to
    impose the *linear* inequality

    .. math::

        u_k \\;\\ge\\; f(x, y^{k}) - \\alpha
        \\quad\\Longleftrightarrow\\quad
        \\ell_k^{T} x - \\alpha - u_k \\;\\le\\; 0 ,

    where :math:`\\ell_k` is row ``k`` of the loss-coefficient matrix ``L``.
    For the portfolio application ``L = -Y``, so the row reads
    :math:`-x^{T}y^{k} - \\alpha - u_k \\le 0`, i.e.
    :math:`x^{T}y^{k} + \\alpha + u_k \\ge 0` - precisely the constraint printed
    after eq. (17).  At the optimum the inequality is tight whenever the
    scenario is in the tail, so ``u_k`` equals the positive part exactly.

    The remaining rows encode :math:`X` (eq. 16): the budget equality
    :math:`\\sum_j x_j = 1` (eq. 11), the expected-return floor
    :math:`\\bar{\\ell}^{T}x \\le -R` (eq. 15), the box constraints, and any
    user-supplied linear constraints.

    Sparsity
    --------
    The scenario block is ``[L | -1 | -I_q]``.  The identity block is stored
    sparsely: at ``q = 20,000`` a dense representation would need ~3.2 GB,
    while the CSR form needs ``O(q)`` entries.  HiGHS consumes the sparse form
    natively.

    Parameters
    ----------
    loss_scenarios
        Loss *coefficient* matrix ``L`` of shape ``(q, n)`` such that the
        scenario losses of portfolio ``x`` are ``L @ x``.  Supplying the loss
        coefficients rather than returns keeps the class reusable for the
        hedging application of Section 4, where eq. (23) gives
        :math:`f(x, y) = x^{T}(m - y)` and hence ``L[k] = m - y^k``.  Use
        :meth:`from_returns` for the standard return-based case.
    beta
        Confidence level in ``(0, 1)``.
    constraints
        Feasible set :math:`X`.  Defaults to eq. (11) with no return floor.
    expected_returns
        Mean return vector used in the eq. (15) constraint and in the reported
        expected return.  Defaults to ``-L.mean(axis=0)``, the scenario-implied
        expected return, which is the right choice when ``L`` was built from a
        return sample.
    rescale_objective
        Apply the :math:`(1-\\beta)` scaling described above.  Default ``True``.

    Raises
    ------
    ValueError
        On inconsistent shapes, a non-interior ``beta``, or an empty box.
    """

    def __init__(
        self,
        loss_scenarios: npt.ArrayLike,
        beta: float,
        constraints: PortfolioConstraints | None = None,
        expected_returns: npt.ArrayLike | None = None,
        rescale_objective: bool = True,
    ) -> None:
        self._losses = _as_2d(loss_scenarios, "loss_scenarios")
        self._beta = _validate_beta(beta)
        self._constraints = constraints if constraints is not None else PortfolioConstraints()
        self._rescale = bool(rescale_objective)

        q, n = self._losses.shape
        if expected_returns is None:
            # E[return] = -E[loss]; the column means of L are the per-asset
            # expected losses, so their negation is the expected return vector.
            self._expected_returns = -self._losses.mean(axis=0)
        else:
            self._expected_returns = _as_1d(expected_returns, "expected_returns")
            if self._expected_returns.size != n:
                raise ValueError(
                    f"'expected_returns' has {self._expected_returns.size} entries, "
                    f"expected {n}."
                )
        _warn_on_thin_tail(q, self._beta)

    # -- constructors ---------------------------------------------------------

    @classmethod
    def from_returns(
        cls,
        returns: npt.ArrayLike,
        beta: float,
        constraints: PortfolioConstraints | None = None,
        expected_returns: npt.ArrayLike | None = None,
        rescale_objective: bool = True,
    ) -> "CVaRLinearProgram":
        """Build the programme from a scenario **return** matrix.

        Applies eq. (12), ``f(x, y) = -x^T y``, so the loss-coefficient matrix
        is ``L = -Y``.

        Parameters
        ----------
        returns
            Scenario return matrix ``Y``, shape ``(q, n)``.
        beta, constraints, expected_returns, rescale_objective
            See :class:`CVaRLinearProgram`.

        Returns
        -------
        CVaRLinearProgram
        """
        y = _as_2d(returns, "returns")
        return cls(
            loss_scenarios=-y,
            beta=beta,
            constraints=constraints,
            expected_returns=expected_returns,
            rescale_objective=rescale_objective,
        )

    @classmethod
    def from_estimator(
        cls,
        estimator: HistoricalRiskEstimator,
        beta: float,
        constraints: PortfolioConstraints | None = None,
        rescale_objective: bool = True,
    ) -> "CVaRLinearProgram":
        """Build the programme from a :class:`HistoricalRiskEstimator`."""
        return cls.from_returns(
            returns=estimator.returns,
            beta=beta,
            constraints=constraints,
            rescale_objective=rescale_objective,
        )

    # -- dimensions -----------------------------------------------------------

    @property
    def n_scenarios(self) -> int:
        """Number of scenarios ``q``."""
        return self._losses.shape[0]

    @property
    def n_assets(self) -> int:
        """Number of instruments ``n``."""
        return self._losses.shape[1]

    @property
    def n_variables(self) -> int:
        """Dimension of the LP variable ``z``: ``n + 1 + q``."""
        return self.n_assets + 1 + self.n_scenarios

    @property
    def beta(self) -> float:
        """Confidence level."""
        return self._beta

    # -- LP assembly ----------------------------------------------------------

    def _objective(self) -> FloatArray:
        """Cost vector ``c`` over ``z = (x, alpha, u)``.

        Positions carry zero cost: the portfolio enters the objective only
        through the tail slacks.  See the class docstring for the rescaling.
        """
        q = self.n_scenarios
        scale = (1.0 - self._beta) if self._rescale else 1.0
        c = np.zeros(self.n_variables)
        c[self.n_assets] = scale                      # coefficient on alpha
        c[self.n_assets + 1 :] = scale / (q * (1.0 - self._beta))  # on each u_k
        return c

    def _inequalities(self) -> tuple[sparse.csr_matrix, FloatArray]:
        """Assemble ``A_ub`` and ``b_ub``.

        Row blocks, in order:

        1. ``q`` scenario rows   ``[ L | -1 | -I ] z <= 0``  (linearised kinks),
        2. optional return floor ``[ l_bar | 0 | 0 ] z <= -R``  (eq. 15),
        3. optional user rows    ``[ A | 0 | 0 ] z <= b``.
        """
        q, n = self.n_scenarios, self.n_assets

        # Block 1 - one row per scenario.  csr_matrix keeps the identity block
        # at O(q) storage instead of O(q^2).
        scenario_block = sparse.hstack(
            [
                sparse.csr_matrix(self._losses),          # L
                sparse.csr_matrix(-np.ones((q, 1))),      # -alpha
                -sparse.identity(q, format="csr"),        # -u
            ],
            format="csr",
        )
        blocks: list[sparse.spmatrix] = [scenario_block]
        rhs: list[FloatArray] = [np.zeros(q)]

        # Block 2 - eq. (15): mu(x) <= -R, written on expected losses so that
        # it stays valid for a general loss matrix.
        target = self._constraints.min_expected_return
        if target is not None:
            row = np.concatenate([-self._expected_returns, np.zeros(1 + q)])
            blocks.append(sparse.csr_matrix(row.reshape(1, -1)))
            rhs.append(np.array([-float(target)]))

        # Block 3 - arbitrary user constraints on positions only.
        if self._constraints.linear_inequalities is not None:
            a_user, b_user = self._constraints.linear_inequalities
            a_user = np.atleast_2d(np.asarray(a_user, dtype=np.float64))
            b_user = np.atleast_1d(np.asarray(b_user, dtype=np.float64))
            if a_user.shape[1] != n:
                raise ValueError(
                    f"'linear_inequalities' matrix has {a_user.shape[1]} columns, "
                    f"expected {n}."
                )
            if a_user.shape[0] != b_user.size:
                raise ValueError(
                    "'linear_inequalities' matrix and vector have inconsistent "
                    f"row counts: {a_user.shape[0]} vs {b_user.size}."
                )
            padded = np.hstack([a_user, np.zeros((a_user.shape[0], 1 + q))])
            blocks.append(sparse.csr_matrix(padded))
            rhs.append(b_user)

        return sparse.vstack(blocks, format="csr"), np.concatenate(rhs)

    def _equalities(self) -> tuple[sparse.csr_matrix | None, FloatArray | None]:
        """Assemble ``A_eq`` and ``b_eq``: the budget (eq. 11) plus user rows."""
        q, n = self.n_scenarios, self.n_assets
        blocks: list[sparse.spmatrix] = []
        rhs: list[FloatArray] = []

        if self._constraints.budget is not None:
            row = np.concatenate([np.ones(n), np.zeros(1 + q)])
            blocks.append(sparse.csr_matrix(row.reshape(1, -1)))
            rhs.append(np.array([float(self._constraints.budget)]))

        if self._constraints.linear_equalities is not None:
            a_user, b_user = self._constraints.linear_equalities
            a_user = np.atleast_2d(np.asarray(a_user, dtype=np.float64))
            b_user = np.atleast_1d(np.asarray(b_user, dtype=np.float64))
            if a_user.shape[1] != n:
                raise ValueError(
                    f"'linear_equalities' matrix has {a_user.shape[1]} columns, "
                    f"expected {n}."
                )
            if a_user.shape[0] != b_user.size:
                raise ValueError(
                    "'linear_equalities' matrix and vector have inconsistent "
                    f"row counts: {a_user.shape[0]} vs {b_user.size}."
                )
            padded = np.hstack([a_user, np.zeros((a_user.shape[0], 1 + q))])
            blocks.append(sparse.csr_matrix(padded))
            rhs.append(b_user)

        if not blocks:
            return None, None
        return sparse.vstack(blocks, format="csr"), np.concatenate(rhs)

    def _bounds(self) -> list[tuple[float | None, float | None]]:
        """Variable bounds: box on ``x``, free ``alpha``, non-negative ``u``.

        ``alpha`` must be free - VaR is negative whenever the portfolio makes
        money at the given confidence level, which is routine for
        ``beta = 0.90`` on a strongly trending asset.  Clamping it at zero is a
        classic implementation bug that silently biases CVaR upward.
        """
        return (
            self._constraints.bounds_for(self.n_assets)
            + [(None, None)]
            + [(0.0, None)] * self.n_scenarios
        )

    def build(self) -> dict[str, object]:
        """Return the assembled LP data without solving.

        Exposed for unit testing, for feeding an alternative solver (CPLEX,
        Gurobi, HiGHS via its own API), and for regulatory model
        documentation, where the exact constraint matrix must be auditable.

        Returns
        -------
        dict
            Keys ``c``, ``A_ub``, ``b_ub``, ``A_eq``, ``b_eq``, ``bounds``,
            matching the signature of :func:`scipy.optimize.linprog`.
        """
        a_ub, b_ub = self._inequalities()
        a_eq, b_eq = self._equalities()
        return {
            "c": self._objective(),
            "A_ub": a_ub,
            "b_ub": b_ub,
            "A_eq": a_eq,
            "b_eq": b_eq,
            "bounds": self._bounds(),
        }

    # -- solve ----------------------------------------------------------------

    def solve(
        self,
        method: str = "highs",
        options: dict[str, object] | None = None,
        var_tolerance: float = 1e-6,
    ) -> CVaRSolution:
        """Solve the programme and return the optimal portfolio with VaR/CVaR.

        Parameters
        ----------
        method
            ``linprog`` method.  ``"highs"`` dispatches to the dual simplex or
            interior point solver as appropriate and is the production choice.
        options
            Solver options forwarded to ``linprog`` (e.g.
            ``{"presolve": True, "time_limit": 60}``).
        var_tolerance
            Absolute tolerance for deciding whether :math:`\\alpha^{*}` equals
            the empirical VaR, i.e. whether the argmin interval
            :math:`A_\\beta(x^{*})` collapsed to a point.

        Returns
        -------
        CVaRSolution
            Contains the optimal weights, VaR, CVaR, expected return, an
            independent empirical re-estimate, and solver diagnostics.

        Raises
        ------
        RuntimeError
            If the LP is infeasible or unbounded.  Infeasibility usually means
            ``min_expected_return`` exceeds the largest attainable expected
            return under the box and budget constraints.

        Notes
        -----
        By Theorem 2 the pair :math:`(x^{*}, \\alpha^{*})` solving the joint
        problem has :math:`x^{*}` minimising :math:`\\beta`-CVaR and
        :math:`\\alpha^{*} \\in A_\\beta(x^{*})`.  When that interval is a
        single point - "as is typical" per the paper - :math:`\\alpha^{*}` *is*
        the :math:`\\beta`-VaR.  When it is not, eq. (7) requires the **left**
        endpoint, which the simplex vertex need not deliver; the returned
        object therefore also carries the empirical VaR, and
        :attr:`CVaRSolution.reported_var` always yields the left endpoint.
        """
        problem = self.build()
        result: OptimizeResult = linprog(
            c=problem["c"],
            A_ub=problem["A_ub"],
            b_ub=problem["b_ub"],
            A_eq=problem["A_eq"],
            b_eq=problem["b_eq"],
            bounds=problem["bounds"],
            method=method,
            options=options,
        )

        if not result.success:
            raise RuntimeError(
                f"CVaR linear programme failed (status={result.status}): {result.message}. "
                "A common cause is an infeasible 'min_expected_return' target; check it "
                "against the maximum attainable expected return under the box constraints."
            )

        n = self.n_assets
        z = np.asarray(result.x, dtype=np.float64)
        weights = z[:n]
        alpha_star = float(z[n])

        # Undo the (1 - beta) objective rescaling to recover CVaR itself.
        scale = (1.0 - self._beta) if self._rescale else 1.0
        cvar = float(result.fun) / scale

        # Independent verification: recompute the tail metrics straight from
        # the realised scenario losses of x*.  Agreement between 'cvar' (an LP
        # optimal value) and 'empirical.cvar' (a sorting-based estimate) is a
        # strong end-to-end check on the constraint matrix.
        realised_losses = self._losses @ weights
        empirical = historical_var_cvar(realised_losses, self._beta)

        var_is_left_endpoint = bool(abs(alpha_star - empirical.var) <= var_tolerance)
        if not var_is_left_endpoint:
            LOGGER.info(
                "alpha* = %.8f differs from the empirical VaR %.8f: the argmin "
                "interval A_beta(x*) is non-degenerate. Reporting the left "
                "endpoint per eq. (7).",
                alpha_star,
                empirical.var,
            )

        budget = self._constraints.budget
        if budget is not None and abs(weights.sum() - budget) > 1e-6:
            LOGGER.warning(
                "Budget constraint violated beyond tolerance: sum(x) = %.10f, "
                "target = %.10f.",
                weights.sum(),
                budget,
            )

        return CVaRSolution(
            weights=weights,
            var=alpha_star,
            cvar=cvar,
            expected_return=float(self._expected_returns @ weights),
            empirical=empirical,
            var_is_left_endpoint=var_is_left_endpoint,
            success=bool(result.success),
            status=int(result.status),
            message=str(result.message),
            n_iterations=int(getattr(result, "nit", -1)),
            beta=self._beta,
            n_scenarios=self.n_scenarios,
            n_assets=n,
            tail_slacks=z[n + 1 :],
        )


# ==============================================================================
# Frontier utility
# ==============================================================================


def cvar_efficient_frontier(
    returns: npt.ArrayLike,
    beta: float,
    target_returns: Sequence[float] | npt.ArrayLike,
    constraints: PortfolioConstraints | None = None,
    expected_returns: npt.ArrayLike | None = None,
    skip_infeasible: bool = True,
) -> list[CVaRSolution]:
    """Trace the mean-CVaR efficient frontier by re-solving the LP per target.

    Sweeping ``R`` in the eq. (15) constraint :math:`\\mu(x) \\le -R` and
    minimising CVaR at each level produces the exact frontier: because the
    problem is convex, every point is globally optimal.  This is the
    *reference* frontier against which the NSGA-II Pareto front of
    ``src/optimization/multiobjective.py`` will be scored - a genetic algorithm
    that cannot recover this curve on a two-objective sub-problem is
    mis-specified.

    Parameters
    ----------
    returns
        Scenario return matrix ``Y``, shape ``(q, n)``.
    beta
        Confidence level in ``(0, 1)``.
    target_returns
        Values of ``R`` to sweep.
    constraints
        Base feasible set; ``min_expected_return`` is overridden per target.
    expected_returns
        Mean return vector used in the eq. (15) constraint.  Defaults to the
        scenario sample mean.  Supply the same vector used elsewhere in the
        analysis: a frontier swept against the sample mean is not directly
        comparable with one swept against a forward-looking view, because the
        two impose different constraints at nominally the same target.
    skip_infeasible
        If ``True``, targets whose LP is infeasible are logged and skipped
        rather than raising - convenient when sweeping a generous grid.

    Returns
    -------
    list of CVaRSolution
        One entry per feasible target, in the order supplied.
    """
    base = constraints if constraints is not None else PortfolioConstraints()
    y = _as_2d(returns, "returns")
    frontier: list[CVaRSolution] = []

    for target in np.asarray(target_returns, dtype=np.float64).ravel():
        # dataclasses.replace would also work; an explicit copy keeps any
        # user-supplied constraint arrays shared rather than duplicated.
        point_constraints = PortfolioConstraints(
            budget=base.budget,
            lower_bounds=base.lower_bounds,
            upper_bounds=base.upper_bounds,
            min_expected_return=float(target),
            linear_inequalities=base.linear_inequalities,
            linear_equalities=base.linear_equalities,
        )
        program = CVaRLinearProgram.from_returns(
            y, beta, point_constraints, expected_returns=expected_returns
        )
        try:
            frontier.append(program.solve())
        except RuntimeError as exc:
            if not skip_infeasible:
                raise
            LOGGER.warning("Frontier target R=%.6f is infeasible: %s", target, exc)

    return frontier
