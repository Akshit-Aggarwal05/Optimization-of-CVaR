"""Portfolio objectives for multi-objective optimisation.

The desk balances three conflicting mandates:

======================  ========  ================================================
Objective               Sense     Conflict
======================  ========  ================================================
Expected return         maximise  Return is compensation for bearing tail risk;
                                  pushing it up drags CVaR up with it.
Conditional VaR         minimise  De-risking means rotating into low-return
                                  assets, or buying protection that costs money.
Hedging cost            minimise  Protection suppresses CVaR but is paid for in
                                  spread and carry, and it is paid whether or not
                                  the tail event arrives.
======================  ========  ================================================

No single portfolio optimises all three, so the solution is a *surface* - the
Pareto front - not a point.

LP representability
-------------------
Every objective here is **convex and piecewise linear** in ``x``:

* expected return is linear;
* CVaR is convex piecewise linear by Rockafellar & Uryasev Theorem 2;
* the hedging cost is a positively weighted sum of absolute values.

Each therefore admits an *epigraph* form ``f(x) <= t`` expressible with linear
constraints and auxiliary variables - see :class:`LinearEpigraph`.  That is
what allows the Tchebycheff scalarisation in
:mod:`src.optimization.multiobjective` to be solved to **global optimality** by
linear programming, producing an exact reference front for scoring the genetic
algorithm.

The moment a genuinely non-convex feature enters the mandate - square-root
market impact ``|dx|^{3/2}``, cardinality limits ("hold at most 20 names"),
minimum lot sizes, or a barrier-option payoff - LP representability is lost and
the exact route is closed.  That, and not a preference for biological metaphor,
is the reason NSGA-II is in this codebase: it is the fallback that survives
those features.  Such an objective should simply not implement
:meth:`Objective.lp_epigraph`, and the solvers will route around it.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Final, Sequence

import numpy as np
import numpy.typing as npt
from scipy import sparse

from src.models.risk_metrics import (
    _as_1d,
    _as_2d,
    _validate_beta,
    historical_var_cvar_batch,
)

__all__ = [
    "ObjectiveSense",
    "LinearEpigraph",
    "Objective",
    "ExpectedReturnObjective",
    "CVaRObjective",
    "HedgingCostModel",
    "HedgingCostObjective",
    "VarianceObjective",
    "ObjectiveSet",
]

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]


class ObjectiveSense(str, Enum):
    """Direction of improvement for an objective."""

    MINIMISE = "minimise"
    MAXIMISE = "maximise"

    @property
    def sign(self) -> float:
        """``+1`` for minimisation, ``-1`` for maximisation.

        Multi-objective theory is written for minimisation.  Multiplying a
        maximisation objective by ``-1`` converts it, so the whole engine can
        work in a single convention internally while still reporting numbers in
        their natural, human-readable units.
        """
        return 1.0 if self is ObjectiveSense.MINIMISE else -1.0


@dataclass(frozen=True)
class LinearEpigraph:
    """Linear-programming epigraph form of a convex piecewise-linear objective.

    Represents the **minimisation-sense** value of an objective as

    .. math::

        f(x) \\;=\\; \\min_{w} \\;\\; c_x^{T} x + c_w^{T} w
        \\quad \\text{s.t.} \\quad A_x x + A_w w \\le b, \\;\\; w \\in W,

    where ``w`` are auxiliary variables private to this objective and ``W`` is
    their box.  The inner minimisation is exact only when the objective is
    driven *downward* by the outer problem; the Tchebycheff programme
    guarantees this through a strictly positive augmentation coefficient, so
    every epigraph is tight at the optimum.  See
    :class:`~src.optimization.multiobjective.TchebycheffLPSolver` for the
    argument.

    Attributes
    ----------
    n_aux
        Number of auxiliary variables ``w``.  Zero for a purely linear
        objective such as expected return.
    c_x, c_aux
        Objective coefficients, shapes ``(n,)`` and ``(n_aux,)``.
    A_x, A_aux, b
        Inequality data, shapes ``(r, n)``, ``(r, n_aux)`` and ``(r,)``.
        ``r = 0`` is allowed.  The constraint blocks may be dense arrays *or*
        any :mod:`scipy.sparse` matrix - the CVaR epigraph carries a ``-I_q``
        block that would need 2.1 GB dense at ``q = 16,384`` and O(q) sparse,
        so the assembler treats every block as sparse.
    aux_bounds
        One ``(low, high)`` pair per auxiliary variable; ``None`` means
        unbounded on that side.
    """

    n_aux: int
    c_x: FloatArray
    c_aux: FloatArray
    A_x: FloatArray | sparse.spmatrix
    A_aux: FloatArray | sparse.spmatrix
    b: FloatArray
    aux_bounds: tuple[tuple[float | None, float | None], ...]

    def __post_init__(self) -> None:
        """Validate internal shape consistency at construction time."""
        n_rows = self.b.size
        if self.c_aux.size != self.n_aux:
            raise ValueError(
                f"c_aux has {self.c_aux.size} entries, expected n_aux={self.n_aux}."
            )
        if len(self.aux_bounds) != self.n_aux:
            raise ValueError(
                f"aux_bounds has {len(self.aux_bounds)} entries, "
                f"expected n_aux={self.n_aux}."
            )
        if self.A_x.shape[0] != n_rows or self.A_aux.shape[0] != n_rows:
            raise ValueError(
                f"Constraint blocks disagree on row count: A_x={self.A_x.shape}, "
                f"A_aux={self.A_aux.shape}, b={self.b.shape}."
            )
        if self.A_x.shape[1] != self.c_x.size:
            raise ValueError(
                f"A_x has {self.A_x.shape[1]} columns but c_x has {self.c_x.size}."
            )
        if self.A_aux.shape[1] != self.n_aux:
            raise ValueError(
                f"A_aux has {self.A_aux.shape[1]} columns, expected {self.n_aux}."
            )


# ==============================================================================
# Objective interface
# ==============================================================================


class Objective(ABC):
    """Abstract scalar criterion evaluated on portfolio weights.

    Subclasses report values in **natural units** (expected return as a return,
    CVaR as a loss, cost as a cost) and declare a :class:`ObjectiveSense`.  The
    framework converts to minimisation internally via
    :meth:`minimisation_batch`, so a user never has to remember to negate
    anything.

    Implementations should override :meth:`evaluate_batch` whenever a
    vectorised form exists: NSGA-II makes ``population_size * n_generations``
    evaluations, and the scalar fallback loop is the dominant cost otherwise.
    """

    #: Human-readable label used in reports, plots and result frames.
    name: str = "objective"

    #: Direction of improvement.
    sense: ObjectiveSense = ObjectiveSense.MINIMISE

    @property
    @abstractmethod
    def n_assets(self) -> int:
        """Dimension of the weight vector this objective accepts."""

    @abstractmethod
    def evaluate_batch(self, weights: npt.ArrayLike) -> FloatArray:
        """Evaluate the objective for a population of portfolios.

        Parameters
        ----------
        weights
            Population matrix of shape ``(P, n)``; row ``p`` is one portfolio.

        Returns
        -------
        FloatArray
            Objective values in natural units, shape ``(P,)``.
        """

    def evaluate(self, weights: npt.ArrayLike) -> float:
        """Evaluate the objective for a single portfolio, in natural units."""
        x = _as_1d(weights, "weights")
        return float(self.evaluate_batch(x.reshape(1, -1))[0])

    def __call__(self, weights: npt.ArrayLike) -> float:
        """Alias for :meth:`evaluate`."""
        return self.evaluate(weights)

    def minimisation_batch(self, weights: npt.ArrayLike) -> FloatArray:
        """Sense-normalised values: ``+f`` if minimising, ``-f`` if maximising."""
        return self.sense.sign * self.evaluate_batch(weights)

    def to_natural(self, minimisation_values: npt.ArrayLike) -> FloatArray:
        """Invert :meth:`minimisation_batch` back to natural units."""
        return self.sense.sign * np.asarray(minimisation_values, dtype=np.float64)

    @property
    def supports_lp(self) -> bool:
        """Whether :meth:`lp_epigraph` is available for this objective."""
        try:
            self.lp_epigraph()
        except NotImplementedError:
            return False
        return True

    def lp_epigraph(self) -> LinearEpigraph:
        """Return the LP epigraph of the **minimisation-sense** objective.

        Raises
        ------
        NotImplementedError
            If the objective is not convex piecewise linear.  Callers must
            treat this as routine control flow, not as an error: it is the
            signal to fall back on the population-based solver.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no linear-programming representation."
        )

    def _validate_population(self, weights: npt.ArrayLike) -> FloatArray:
        """Coerce and shape-check a population matrix."""
        matrix = _as_2d(weights, "weights")
        if matrix.shape[1] != self.n_assets:
            raise ValueError(
                f"{type(self).__name__} expects {self.n_assets} assets, "
                f"got weight vectors of length {matrix.shape[1]}."
            )
        return matrix

    def __repr__(self) -> str:  # pragma: no cover - presentation only
        return f"{type(self).__name__}(name={self.name!r}, sense={self.sense.value})"


# ==============================================================================
# Concrete objectives
# ==============================================================================


class ExpectedReturnObjective(Objective):
    """Expected portfolio return :math:`\\bar{y}^{T}x` - to be **maximised**.

    Linear in ``x``, so its epigraph needs no auxiliary variables.

    Parameters
    ----------
    expected_returns
        Mean return vector, shape ``(n,)``.  Supply the forward-looking view
        the desk actually trades on; the scenario sample mean is only a
        default of convenience, and is a notoriously noisy estimator.
    name
        Label override.
    """

    sense = ObjectiveSense.MAXIMISE

    def __init__(
        self, expected_returns: npt.ArrayLike, name: str = "expected_return"
    ) -> None:
        self._mean = _as_1d(expected_returns, "expected_returns")
        self.name = name

    @classmethod
    def from_scenarios(
        cls, returns: npt.ArrayLike, name: str = "expected_return"
    ) -> "ExpectedReturnObjective":
        """Build from the sample mean of a scenario return matrix ``(q, n)``."""
        return cls(_as_2d(returns, "returns").mean(axis=0), name=name)

    @property
    def n_assets(self) -> int:
        """Number of instruments."""
        return self._mean.size

    @property
    def expected_returns(self) -> FloatArray:
        """The mean return vector."""
        return self._mean

    def evaluate_batch(self, weights: npt.ArrayLike) -> FloatArray:
        """Expected returns of every portfolio in the population."""
        return self._validate_population(weights) @ self._mean

    def lp_epigraph(self) -> LinearEpigraph:
        """Minimisation form ``-m^T x``: linear, no auxiliaries, no constraints."""
        n = self.n_assets
        return LinearEpigraph(
            n_aux=0,
            c_x=-self._mean,  # sense.sign == -1 for a maximisation objective
            c_aux=np.zeros(0),
            A_x=np.zeros((0, n)),
            A_aux=np.zeros((0, 0)),
            b=np.zeros(0),
            aux_bounds=(),
        )


class CVaRObjective(Objective):
    """Scenario-based :math:`\\beta`-CVaR of the portfolio - to be **minimised**.

    Evaluation reuses the exact discrete-atom estimator of
    :mod:`src.models.risk_metrics`, so the value the genetic algorithm sees is
    numerically identical to the value the linear programme optimises.  Any
    discrepancy between the two would make the comparative analysis
    meaningless, which is why both routes share one kernel.

    Parameters
    ----------
    loss_scenarios
        Loss-coefficient matrix ``L`` of shape ``(q, n)``, so that scenario
        losses are ``L @ x``.  For the standard return-based case this is
        ``-Y``; use :meth:`from_returns`.
    beta
        Confidence level in ``(0, 1)``.
    name
        Label override.
    """

    sense = ObjectiveSense.MINIMISE

    def __init__(
        self, loss_scenarios: npt.ArrayLike, beta: float, name: str | None = None
    ) -> None:
        self._losses = _as_2d(loss_scenarios, "loss_scenarios")
        self._beta = _validate_beta(beta)
        self.name = name if name is not None else f"cvar_{self._beta:g}"

    @classmethod
    def from_returns(
        cls, returns: npt.ArrayLike, beta: float, name: str | None = None
    ) -> "CVaRObjective":
        """Build from a scenario return matrix via eq. (12), ``f(x,y) = -x^T y``."""
        return cls(-_as_2d(returns, "returns"), beta, name=name)

    @property
    def n_assets(self) -> int:
        """Number of instruments."""
        return self._losses.shape[1]

    @property
    def n_scenarios(self) -> int:
        """Number of scenarios ``q``."""
        return self._losses.shape[0]

    @property
    def beta(self) -> float:
        """Confidence level."""
        return self._beta

    @property
    def loss_scenarios(self) -> FloatArray:
        """The loss-coefficient matrix ``L``."""
        return self._losses

    def evaluate_batch(self, weights: npt.ArrayLike) -> FloatArray:
        """CVaR of every portfolio, computed in a single vectorised pass."""
        population = self._validate_population(weights)
        # (q, n) @ (n, P) -> (q, P): column p holds portfolio p's scenario losses.
        loss_matrix = self._losses @ population.T
        _, cvar = historical_var_cvar_batch(loss_matrix, self._beta)
        return cvar

    def value_at_risk_batch(self, weights: npt.ArrayLike) -> FloatArray:
        """Companion VaR of every portfolio, free of extra cost.

        Theorem 1 delivers VaR as a by-product of the CVaR calculation, so
        reporting it alongside costs nothing.
        """
        population = self._validate_population(weights)
        var, _ = historical_var_cvar_batch(self._losses @ population.T, self._beta)
        return var

    def lp_epigraph(self) -> LinearEpigraph:
        """The Rockafellar-Uryasev reduction, in epigraph form.

        Auxiliary variables are :math:`(\\alpha, u_1, \\dots, u_q)`:

        .. math::

            \\text{CVaR}_\\beta(x) \\;=\\; \\min_{\\alpha, u \\ge 0}
                \\;\\alpha + \\frac{1}{q(1-\\beta)}\\sum_k u_k
            \\quad\\text{s.t.}\\quad \\ell_k^{T} x - \\alpha - u_k \\le 0 ,

        which is exactly eq. (17) of the paper together with the linearisation
        that follows it.  The inner minimisation over :math:`(\\alpha, u)` is
        the content of Theorem 1, so this epigraph is not an approximation -
        it is an identity.

        Note that :math:`\\alpha` is **free**, not sign-restricted: it is the
        VaR, which is negative whenever the book is profitable at the chosen
        confidence level.
        """
        q, n = self._losses.shape
        n_aux = 1 + q

        c_aux = np.empty(n_aux)
        c_aux[0] = 1.0  # alpha
        c_aux[1:] = 1.0 / (q * (1.0 - self._beta))  # each u_k

        # [-1 | -I_q]: sparse, because the identity block is O(q^2) dense.
        a_aux = sparse.hstack(
            [
                sparse.csr_matrix(-np.ones((q, 1))),  # -alpha
                -sparse.identity(q, format="csr"),  # -u_k
            ],
            format="csr",
        )

        bounds: tuple[tuple[float | None, float | None], ...] = (
            (None, None),  # alpha is free - see docstring
            *(((0.0, None),) * q),  # u_k >= 0
        )
        return LinearEpigraph(
            n_aux=n_aux,
            c_x=np.zeros(n),
            c_aux=c_aux,
            A_x=self._losses,
            A_aux=a_aux,
            b=np.zeros(q),
            aux_bounds=bounds,
        )


@dataclass
class HedgingCostModel:
    """Cost of establishing and carrying a derivative hedge overlay.

    Two economically distinct components, both linear in exposure to first
    order:

    **Execution cost** - paid once, on the traded notional.  Charged as
    ``spreads @ |x - baseline_weights|``: half the bid-ask plus commission on
    the distance from the book the desk already holds.  Options are charged
    several multiples of the cash spread because listed option books are
    thinner than their underlyings.

    **Carry cost** - paid every period, on the position held.  Charged as
    ``carry @ |x|``.  For long options this is theta bleed: the premium decays
    whether or not the tail event arrives, which is precisely what makes
    hedging cost a genuine third objective rather than a footnote to CVaR.
    For short and borrowed positions it is the financing and borrow fee.  The
    absolute value is deliberate - a short position costs money to carry too.

    The result is convex and piecewise linear, hence LP-representable.  Adding
    a market-impact term ``impact @ |x - baseline| ** 1.5`` would be more
    realistic for size and would break that property; see the module docstring.

    Attributes
    ----------
    baseline_weights
        The book as currently held, ``z``.  Defaults to all-zero, i.e. costing
        a portfolio built from flat.
    spreads
        Per-unit execution cost, one entry per instrument.  Scalars broadcast.
    carry
        Per-period holding cost per unit of gross exposure.  Scalars broadcast.
    """

    n_assets: int
    baseline_weights: npt.ArrayLike | None = None
    spreads: float | npt.ArrayLike = 0.0010
    carry: float | npt.ArrayLike = 0.0

    def __post_init__(self) -> None:
        """Broadcast, validate and freeze the cost vectors."""
        n = int(self.n_assets)
        if n <= 0:
            raise ValueError(f"'n_assets' must be positive, got {n}.")
        self.baseline_weights = (
            np.zeros(n)
            if self.baseline_weights is None
            else _as_1d(self.baseline_weights, "baseline_weights")
        )
        if self.baseline_weights.size != n:
            raise ValueError(
                f"'baseline_weights' has {self.baseline_weights.size} entries, "
                f"expected {n}."
            )
        self.spreads = self._broadcast(self.spreads, n, "spreads")
        self.carry = self._broadcast(self.carry, n, "carry")

    @staticmethod
    def _broadcast(value: float | npt.ArrayLike, n: int, name: str) -> FloatArray:
        """Broadcast a scalar cost, or validate a per-instrument vector."""
        arr = np.asarray(value, dtype=np.float64)
        arr = np.full(n, float(arr)) if arr.ndim == 0 else arr.ravel()
        if arr.size != n:
            raise ValueError(f"'{name}' has {arr.size} entries, expected {n}.")
        if np.any(arr < 0.0):
            raise ValueError(f"'{name}' must be non-negative; costs are never rebates.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"'{name}' contains NaN or infinite values.")
        return arr

    def cost_batch(self, weights: npt.ArrayLike) -> FloatArray:
        """Total hedging cost for a population of portfolios, shape ``(P,)``."""
        population = _as_2d(weights, "weights")
        if population.shape[1] != self.n_assets:
            raise ValueError(
                f"HedgingCostModel expects {self.n_assets} assets, "
                f"got {population.shape[1]}."
            )
        turnover = np.abs(population - self.baseline_weights) @ self.spreads
        holding = np.abs(population) @ self.carry
        return turnover + holding

    def cost(self, weights: npt.ArrayLike) -> float:
        """Total hedging cost of a single portfolio."""
        return float(self.cost_batch(_as_1d(weights, "weights").reshape(1, -1))[0])


class HedgingCostObjective(Objective):
    """Derivative hedging cost - to be **minimised**.

    Parameters
    ----------
    model
        The :class:`HedgingCostModel` supplying spreads, carry and the
        baseline book.
    name
        Label override.
    """

    sense = ObjectiveSense.MINIMISE

    def __init__(self, model: HedgingCostModel, name: str = "hedging_cost") -> None:
        self._model = model
        self.name = name

    @property
    def n_assets(self) -> int:
        """Number of instruments."""
        return self._model.n_assets

    @property
    def model(self) -> HedgingCostModel:
        """The underlying cost model."""
        return self._model

    def evaluate_batch(self, weights: npt.ArrayLike) -> FloatArray:
        """Hedging cost of every portfolio in the population."""
        return self._model.cost_batch(self._validate_population(weights))

    def lp_epigraph(self) -> LinearEpigraph:
        """Linearise the absolute values with two blocks of auxiliaries.

        Each ``|x_j - z_j|`` is replaced by ``d_j >= 0`` subject to
        ``x_j - z_j <= d_j`` and ``-(x_j - z_j) <= d_j``; each ``|x_j|`` by
        ``e_j`` similarly.  Because ``d`` and ``e`` carry strictly positive
        objective coefficients they are driven down to the absolute values
        exactly - the standard exact linearisation, valid precisely because
        the costs are non-negative (enforced in :class:`HedgingCostModel`).

        Instruments with zero cost still get auxiliaries; the LP presolve
        eliminates them, and special-casing would complicate the index
        arithmetic for no measurable gain.
        """
        n = self.n_assets
        z = np.asarray(self._model.baseline_weights, dtype=np.float64)
        n_aux = 2 * n  # (d, e)

        identity = np.eye(n)
        zeros = np.zeros((n, n))

        # Rows:  x - d <=  z  |  -x - d <= -z  |  x - e <= 0  |  -x - e <= 0
        a_x = np.vstack([identity, -identity, identity, -identity])
        a_aux = np.vstack(
            [
                np.hstack([-identity, zeros]),
                np.hstack([-identity, zeros]),
                np.hstack([zeros, -identity]),
                np.hstack([zeros, -identity]),
            ]
        )
        b = np.concatenate([z, -z, np.zeros(n), np.zeros(n)])

        return LinearEpigraph(
            n_aux=n_aux,
            c_x=np.zeros(n),
            c_aux=np.concatenate([self._model.spreads, self._model.carry]),
            A_x=a_x,
            A_aux=a_aux,
            b=b,
            aux_bounds=tuple([(0.0, None)] * n_aux),
        )


class VarianceObjective(Objective):
    """Portfolio variance :math:`x^{T}Vx` - to be **minimised**.

    Included so the engine can reproduce the Markowitz problem (P3) of the
    paper alongside the CVaR problem (P1) and quantify how far apart their
    Pareto fronts sit once the loss distribution stops being Gaussian.  It is
    *quadratic*, hence deliberately **not** LP-representable: it is the
    resident example of an objective that forces the population-based route.

    Parameters
    ----------
    covariance
        Covariance matrix ``V``, shape ``(n, n)``.
    name
        Label override.
    """

    sense = ObjectiveSense.MINIMISE

    def __init__(self, covariance: npt.ArrayLike, name: str = "variance") -> None:
        cov = _as_2d(covariance, "covariance")
        if cov.shape[0] != cov.shape[1]:
            raise ValueError(f"'covariance' must be square, got {cov.shape}.")
        if not np.allclose(cov, cov.T, atol=1e-10):
            raise ValueError("'covariance' must be symmetric.")
        self._covariance = cov
        self.name = name

    @property
    def n_assets(self) -> int:
        """Number of instruments."""
        return self._covariance.shape[0]

    def evaluate_batch(self, weights: npt.ArrayLike) -> FloatArray:
        """Variance of every portfolio: the row-wise quadratic form ``x'Vx``."""
        population = self._validate_population(weights)
        # einsum avoids materialising the (P, P) product X V X'.
        return np.einsum("ij,jk,ik->i", population, self._covariance, population)


# ==============================================================================
# Objective collection
# ==============================================================================


class ObjectiveSet:
    """An ordered collection of objectives evaluated as one vector criterion.

    Provides the single evaluation entry point used by both optimisers, and
    the ideal/nadir bookkeeping that scalarisation and the quality indicators
    both need.

    Parameters
    ----------
    objectives
        Two or more :class:`Objective` instances, all over the same number of
        assets.

    Raises
    ------
    ValueError
        If fewer than two objectives are supplied, if their asset dimensions
        disagree, or if two share a name (which would make result frames
        ambiguous).
    """

    def __init__(self, objectives: Sequence[Objective]) -> None:
        objectives = list(objectives)
        if len(objectives) < 2:
            raise ValueError(
                f"Multi-objective optimisation needs at least 2 objectives, "
                f"got {len(objectives)}."
            )
        dimensions = {obj.n_assets for obj in objectives}
        if len(dimensions) != 1:
            raise ValueError(
                f"Objectives disagree on the asset dimension: {sorted(dimensions)}."
            )
        names = [obj.name for obj in objectives]
        if len(set(names)) != len(names):
            raise ValueError(f"Objective names must be unique, got {names}.")
        self._objectives = objectives

    def __len__(self) -> int:
        """Number of objectives ``m``."""
        return len(self._objectives)

    def __iter__(self):
        """Iterate over the component objectives in declaration order."""
        return iter(self._objectives)

    def __getitem__(self, index: int) -> Objective:
        """Access an objective by position."""
        return self._objectives[index]

    @property
    def n_objectives(self) -> int:
        """Number of objectives ``m``."""
        return len(self._objectives)

    @property
    def n_assets(self) -> int:
        """Common asset dimension ``n``."""
        return self._objectives[0].n_assets

    @property
    def names(self) -> list[str]:
        """Objective labels, in order."""
        return [obj.name for obj in self._objectives]

    @property
    def senses(self) -> list[ObjectiveSense]:
        """Objective senses, in order."""
        return [obj.sense for obj in self._objectives]

    @property
    def signs(self) -> FloatArray:
        """Sense signs as a vector, for converting whole objective matrices."""
        return np.array([obj.sense.sign for obj in self._objectives])

    @property
    def supports_lp(self) -> bool:
        """Whether *every* objective is LP-representable.

        The exact Tchebycheff route requires all of them; one quadratic or
        non-convex member forces the population-based solver.
        """
        return all(obj.supports_lp for obj in self._objectives)

    def evaluate_batch(self, weights: npt.ArrayLike) -> FloatArray:
        """Natural-unit objective matrix of shape ``(P, m)``."""
        population = _as_2d(weights, "weights")
        columns = [obj.evaluate_batch(population) for obj in self._objectives]
        return np.column_stack(columns)

    def minimisation_batch(self, weights: npt.ArrayLike) -> FloatArray:
        """Sense-normalised objective matrix of shape ``(P, m)``.

        This is the matrix all Pareto machinery operates on: after
        normalisation, "smaller is better" holds uniformly.
        """
        return self.evaluate_batch(weights) * self.signs

    def to_natural(self, minimisation_matrix: npt.ArrayLike) -> FloatArray:
        """Convert a minimisation-sense matrix back to natural units."""
        return np.asarray(minimisation_matrix, dtype=np.float64) * self.signs

    def evaluate(self, weights: npt.ArrayLike) -> FloatArray:
        """Natural-unit objective vector for a single portfolio, shape ``(m,)``."""
        x = _as_1d(weights, "weights")
        return self.evaluate_batch(x.reshape(1, -1))[0]
