"""Multi-objective portfolio optimisation: NSGA-II and Tchebycheff scalarisation.

The baseline of Rockafellar & Uryasev (2000) minimises a *single* criterion -
:math:`\\beta`-CVaR - subject to a hard floor on expected return (eq. 15).  That
is a defensible model only if the desk already knows the return it wants.  In
practice the trade-off itself is the decision: how much tail risk is worth how
much return, and how much it is worth paying in spread and carry to buy that
tail down.  This module replaces the single optimum with the **Pareto front**,
the set of portfolios where no objective can improve without another
deteriorating.

Two independent routes are implemented, and they are deliberately different in
kind so that agreement between them is evidence rather than coincidence.

Approach A - NSGA-II (Deb, Pratap, Agarwal & Meyarivan, 2002)
-------------------------------------------------------------
A population-based evolutionary algorithm that approximates the whole front in
one run.  Three ingredients: fast non-dominated sorting for rank, crowding
distance for diversity, and elitist :math:`(\\mu + \\lambda)` survivor
selection.  Constraints beyond the simplex are handled by Deb's
constrained-domination rule.  It assumes nothing about convexity, so it
survives the features that break linear programming - market impact,
cardinality limits, minimum lot sizes, barrier payoffs.

Its genetic operators are specialised for weights on the budget simplex; see
:func:`project_onto_budget_box` and the operator classes.

Approach B - Tchebycheff scalarisation
--------------------------------------
For a weight vector :math:`\\lambda > 0` and a utopian reference point
:math:`z^{*}`, solve

.. math::

    \\min_{x \\in X} \\;\\; \\max_{i} \\;\\lambda_i \\big(f_i(x) - z^{*}_i\\big)
        \\;+\\; \\rho \\sum_i \\lambda_i \\big(f_i(x) - z^{*}_i\\big),

sweeping :math:`\\lambda` over a simplex-lattice design to trace the front.
Unlike a weighted *sum*, the Tchebycheff metric can reach points on
**non-convex** parts of the front - the reason it is the standard scalarisation
in multi-criteria decision making.  The augmentation term :math:`\\rho > 0`
removes weakly Pareto-optimal solutions.

The decisive property here: for the canonical objective triple - expected
return (linear), CVaR (convex piecewise linear by Theorem 2), hedging cost
(a positively weighted sum of absolute values) - the ``max`` can be lifted into
an epigraph variable and the whole scalarised problem becomes a **linear
programme**.  Each point on the front is then a *global* optimum, delivered by
HiGHS in one shot.  So Approach B is not a second heuristic to be averaged with
the first: it is ground truth, and the correct use of it is to measure how much
of the true front NSGA-II actually recovered.

That comparison is what :func:`compare_fronts` reports.

References
----------
Deb, K., Pratap, A., Agarwal, S. and Meyarivan, T. (2002).  *A fast and elitist
multiobjective genetic algorithm: NSGA-II*.  IEEE Transactions on Evolutionary
Computation, 6(2), 182-197.

Deb, K. and Agrawal, R.B. (1995).  *Simulated binary crossover for continuous
search space*.  Complex Systems, 9(2), 115-148.

Das, I. and Dennis, J.E. (1998).  *Normal-boundary intersection*.  SIAM Journal
on Optimization, 8(3), 631-657.  (Source of the simplex-lattice weight design.)

Miettinen, K. (1999).  *Nonlinear Multiobjective Optimization*.  Kluwer.
(Augmented Tchebycheff theory.)

Zitzler, E. and Thiele, L. (1999).  *Multiobjective evolutionary algorithms: a
comparative case study and the strength Pareto approach*.  IEEE Transactions on
Evolutionary Computation, 3(4), 257-271.  (Hypervolume and coverage.)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Final, Iterator, Sequence

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.optimize import OptimizeResult, linprog, minimize

from src.models.risk_metrics import (
    PortfolioConstraints,
    _as_1d,
    _as_2d,
)
from src.optimization.objectives import (
    ExpectedReturnObjective,
    LinearEpigraph,
    Objective,
    ObjectiveSet,
)

__all__ = [
    "project_onto_budget_box",
    "constraint_violation",
    "non_dominated_mask",
    "fast_non_dominated_sort",
    "constrained_non_dominated_sort",
    "crowding_distance",
    "hypervolume",
    "default_reference_point",
    "inverted_generational_distance",
    "generational_distance",
    "spacing",
    "coverage",
    "ParetoFront",
    "CrossoverOperator",
    "MutationOperator",
    "SimulatedBinaryCrossover",
    "SimplexBlendCrossover",
    "PolynomialMutation",
    "WeightTransferMutation",
    "NSGAIIConfig",
    "NSGAII",
    "simplex_lattice_weights",
    "TchebycheffSolution",
    "TchebycheffLPSolver",
    "TchebycheffNonlinearSolver",
    "compare_fronts",
]

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]

#: Tolerance below which a constraint violation counts as satisfied.
FEASIBILITY_TOLERANCE: Final[float] = 1e-9

#: Floor applied to Tchebycheff weight components.  A weight of exactly zero
#: drops an objective out of the programme entirely, which also destroys the
#: tightness argument for that objective's epigraph.  See
#: :class:`TchebycheffLPSolver`.
MIN_LAMBDA: Final[float] = 1e-6


# ==============================================================================
# 1. Feasibility: projection onto the budget simplex
# ==============================================================================


def _finite_bounds(
    constraints: PortfolioConstraints, n_assets: int, reference_scale: float = 1.0
) -> tuple[FloatArray, FloatArray]:
    """Materialise box constraints as finite vectors.

    Genetic operators need a finite span to scale their perturbations, and the
    bisection in :func:`project_onto_budget_box` needs a finite bracket.  An
    unbounded side is therefore replaced by a surrogate three orders of
    magnitude beyond anything the budget can reach.  Because every feasible
    portfolio satisfies ``sum(x) = budget``, no genuine optimum can sit near
    the surrogate, so the substitution is invisible to the solution while
    keeping every routine numerically well posed.

    Parameters
    ----------
    constraints
        Source of the box.
    n_assets
        Dimension ``n``.
    reference_scale
        Magnitude of the budget, used to size the surrogate.

    Returns
    -------
    (FloatArray, FloatArray)
        Finite ``(lower, upper)`` vectors of shape ``(n,)``.
    """
    pairs = constraints.bounds_for(n_assets)
    surrogate = 1_000.0 * max(abs(reference_scale), 1.0)
    lower = np.array([-surrogate if lo is None else lo for lo, _ in pairs])
    upper = np.array([surrogate if hi is None else hi for _, hi in pairs])
    return lower, upper


def project_onto_budget_box(
    points: npt.ArrayLike,
    lower: npt.ArrayLike,
    upper: npt.ArrayLike,
    budget: float | None,
    tolerance: float = 1e-12,
    max_iterations: int = 100,
) -> FloatArray:
    """Exact Euclidean projection onto ``{x : l <= x <= u, sum(x) = budget}``.

    This is *the* operator that makes a real-coded genetic algorithm work on
    portfolio weights.  Naive alternatives are all worse:

    * ``x / sum(x)`` fails outright with short positions (the denominator can
      vanish or flip sign) and silently violates box constraints;
    * clipping into the box then renormalising leaves the box again;
    * penalising the budget violation in the fitness wastes most of the
      population on infeasible points.

    The exact projection keeps every individual feasible by construction, so
    the population never wastes evaluations, and - being the *nearest* feasible
    point - it preserves the search geometry the crossover and mutation
    operators are designed around.

    Mathematics
    -----------
    Minimising :math:`\\tfrac{1}{2}\\|x - v\\|^{2}` subject to
    :math:`l \\le x \\le u` and :math:`\\mathbf{1}^{T}x = b` has Lagrangian
    stationarity :math:`x - v + \\tau \\mathbf{1} - \\mu_l + \\mu_u = 0`, and
    complementary slackness collapses the solution to the one-parameter family

    .. math::

        x(\\tau) \\;=\\; \\mathrm{clip}(v - \\tau,\\; l,\\; u).

    The scalar :math:`\\tau` is fixed by
    :math:`g(\\tau) = \\mathbf{1}^{T}x(\\tau) = b`.  Since ``g`` is continuous,
    piecewise linear and non-increasing, bisection converges monotonically.
    The bracket is exact: at :math:`\\tau = \\min_j(v_j - u_j)` every coordinate
    saturates at its upper bound so :math:`g = \\mathbf{1}^{T}u \\ge b`, and at
    :math:`\\tau = \\max_j(v_j - l_j)` every coordinate saturates at its lower
    bound so :math:`g = \\mathbf{1}^{T}l \\le b`.

    Bisection runs vectorised across the whole population - one ``tau`` per
    individual - so projecting 200 portfolios costs the same handful of
    array operations as projecting one.

    Parameters
    ----------
    points
        Point(s) to project: shape ``(n,)`` or ``(P, n)``.
    lower, upper
        Finite box bounds, shape ``(n,)``.
    budget
        Target for ``sum(x)``.  ``None`` skips the equality and simply clips
        into the box.
    tolerance
        Absolute tolerance on ``sum(x) - budget``.
    max_iterations
        Bisection cap.  Each step halves the bracket, so the default of 100 is
        far beyond double precision; it exists only as a guard.

    Returns
    -------
    FloatArray
        Projected points, same shape as ``points``.

    Raises
    ------
    ValueError
        If the box cannot meet the budget - ``sum(l) > budget`` or
        ``sum(u) < budget`` - which makes the feasible set empty.
    """
    matrix = np.atleast_2d(np.asarray(points, dtype=np.float64))
    low = _as_1d(lower, "lower")
    high = _as_1d(upper, "upper")
    if matrix.shape[1] != low.size or low.size != high.size:
        raise ValueError(
            f"Dimension mismatch: points have {matrix.shape[1]} columns, "
            f"bounds have {low.size} and {high.size} entries."
        )

    if budget is None:
        projected = np.clip(matrix, low, high)
        return projected[0] if np.ndim(points) == 1 else projected

    budget = float(budget)
    if low.sum() > budget + tolerance or high.sum() < budget - tolerance:
        raise ValueError(
            f"Empty feasible set: sum(lower)={low.sum():.6g} and "
            f"sum(upper)={high.sum():.6g} cannot bracket budget={budget:.6g}."
        )

    # g(tau) is non-increasing, so the bracket is [tau_low, tau_high] with
    # g(tau_low) >= budget >= g(tau_high).
    tau_low = (matrix - high).min(axis=1)
    tau_high = (matrix - low).max(axis=1)

    for _ in range(max_iterations):
        tau = 0.5 * (tau_low + tau_high)
        total = np.clip(matrix - tau[:, None], low, high).sum(axis=1)
        if np.all(np.abs(total - budget) <= tolerance):
            break
        # Overshooting the budget means tau is too small: raise the lower end.
        overshoot = total > budget
        tau_low = np.where(overshoot, tau, tau_low)
        tau_high = np.where(overshoot, tau_high, tau)

    tau = 0.5 * (tau_low + tau_high)
    projected = np.clip(matrix - tau[:, None], low, high)
    return projected[0] if np.ndim(points) == 1 else projected


def constraint_violation(
    weights: npt.ArrayLike,
    constraints: PortfolioConstraints,
    expected_returns: npt.ArrayLike | None = None,
) -> FloatArray:
    """Total violation of the constraints *not* handled by projection.

    The budget equality and the box are enforced exactly by
    :func:`project_onto_budget_box`, so they never appear here.  What remains
    is the expected-return floor of eq. (15) and any user-supplied linear
    rows, aggregated as a sum of positive parts.  NSGA-II uses the aggregate in
    Deb's constrained-domination rule, which needs only an ordering of "how
    infeasible", not a per-constraint breakdown.

    Parameters
    ----------
    weights
        Population matrix of shape ``(P, n)``.
    constraints
        The constraint set.
    expected_returns
        Mean return vector, required when ``min_expected_return`` is set.

    Returns
    -------
    FloatArray
        Non-negative violation per portfolio, shape ``(P,)``.  Zero means
        feasible.
    """
    population = _as_2d(weights, "weights")
    violation = np.zeros(population.shape[0])

    if constraints.min_expected_return is not None:
        if expected_returns is None:
            raise ValueError(
                "'expected_returns' is required when 'min_expected_return' is set."
            )
        mean = _as_1d(expected_returns, "expected_returns")
        shortfall = constraints.min_expected_return - population @ mean
        violation += np.maximum(shortfall, 0.0)

    if constraints.linear_inequalities is not None:
        a_matrix, b_vector = constraints.linear_inequalities
        a_matrix = np.atleast_2d(np.asarray(a_matrix, dtype=np.float64))
        b_vector = np.atleast_1d(np.asarray(b_vector, dtype=np.float64))
        excess = population @ a_matrix.T - b_vector
        violation += np.maximum(excess, 0.0).sum(axis=1)

    if constraints.linear_equalities is not None:
        a_matrix, b_vector = constraints.linear_equalities
        a_matrix = np.atleast_2d(np.asarray(a_matrix, dtype=np.float64))
        b_vector = np.atleast_1d(np.asarray(b_vector, dtype=np.float64))
        violation += np.abs(population @ a_matrix.T - b_vector).sum(axis=1)

    return violation


# ==============================================================================
# 2. Pareto dominance machinery
# ==============================================================================


def _domination_matrix(objectives: FloatArray) -> npt.NDArray[np.bool_]:
    """Boolean matrix ``D`` with ``D[i, j]`` true when ``i`` dominates ``j``.

    Minimisation convention: ``i`` dominates ``j`` when it is no worse on every
    objective and strictly better on at least one.  Computed as two broadcast
    comparisons - ``O(N^2 m)`` work but entirely vectorised, which for the
    population sizes NSGA-II uses (a few hundred) is far faster than the
    textbook double loop.
    """
    no_worse = (objectives[:, None, :] <= objectives[None, :, :]).all(axis=-1)
    strictly_better = (objectives[:, None, :] < objectives[None, :, :]).any(axis=-1)
    return no_worse & strictly_better


def _constrained_domination_matrix(
    objectives: FloatArray, violation: FloatArray
) -> npt.NDArray[np.bool_]:
    """Domination under Deb's constrained-domination rule.

    Solution ``i`` constrained-dominates ``j`` when any of:

    1. ``i`` is feasible and ``j`` is not;
    2. both are infeasible and ``i`` violates less;
    3. both are feasible and ``i`` Pareto-dominates ``j``.

    The rule needs no penalty parameter - the perennial weakness of penalty
    methods, where the coefficient trades feasibility against quality and has
    to be retuned for every problem instance.  Here feasibility is
    lexicographically prior, full stop.
    """
    feasible = violation <= FEASIBILITY_TOLERANCE
    both_feasible = feasible[:, None] & feasible[None, :]
    neither_feasible = ~feasible[:, None] & ~feasible[None, :]

    feasible_beats_infeasible = feasible[:, None] & ~feasible[None, :]
    less_violation = neither_feasible & (violation[:, None] < violation[None, :])
    pareto = both_feasible & _domination_matrix(objectives)
    return feasible_beats_infeasible | less_violation | pareto


def _fronts_from_domination(domination: npt.NDArray[np.bool_]) -> list[IntArray]:
    """Peel successive non-dominated fronts from a domination matrix.

    This is the "fast non-dominated sort" of Deb et al. (2002): maintain a
    domination count per solution, emit everything with count zero as the
    current front, then decrement the counts of whatever that front dominated.
    """
    n = domination.shape[0]
    dominated_count = domination.sum(axis=0)  # how many dominate each solution
    remaining = np.ones(n, dtype=bool)
    fronts: list[IntArray] = []

    while remaining.any():
        current = np.flatnonzero(remaining & (dominated_count == 0))
        if current.size == 0:
            # Only reachable if the relation has a cycle, which the strict
            # partial orders above cannot produce.  Emitting the remainder is a
            # safe termination rather than an infinite loop.
            LOGGER.error("Cyclic domination detected; emitting remainder as one front.")
            fronts.append(np.flatnonzero(remaining))
            break
        fronts.append(current)
        remaining[current] = False
        # Everything the emitted front dominated loses those domination counts.
        dominated_count = dominated_count - domination[current].sum(axis=0)
        dominated_count[current] = -1  # retire, so it never re-qualifies
    return fronts


def non_dominated_mask(objectives: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Boolean mask of the Pareto-optimal rows of a minimisation matrix.

    Parameters
    ----------
    objectives
        Minimisation-sense objective matrix, shape ``(P, m)``.

    Returns
    -------
    ndarray of bool
        ``True`` where the row is not dominated by any other row.
    """
    matrix = _as_2d(objectives, "objectives")
    return ~_domination_matrix(matrix).any(axis=0)


def fast_non_dominated_sort(objectives: npt.ArrayLike) -> list[IntArray]:
    """Partition solutions into successive Pareto fronts.

    Parameters
    ----------
    objectives
        Minimisation-sense objective matrix, shape ``(P, m)``.

    Returns
    -------
    list of IntArray
        Index arrays; element 0 is the non-dominated front, element 1 is what
        becomes non-dominated once front 0 is removed, and so on.
    """
    return _fronts_from_domination(_domination_matrix(_as_2d(objectives, "objectives")))


def constrained_non_dominated_sort(
    objectives: npt.ArrayLike, violation: npt.ArrayLike
) -> list[IntArray]:
    """Non-dominated sort under Deb's constrained-domination rule.

    Parameters
    ----------
    objectives
        Minimisation-sense objective matrix, shape ``(P, m)``.
    violation
        Non-negative constraint violation per solution, shape ``(P,)``.

    Returns
    -------
    list of IntArray
        Fronts, best first.
    """
    matrix = _as_2d(objectives, "objectives")
    viol = _as_1d(violation, "violation")
    if viol.size != matrix.shape[0]:
        raise ValueError(
            f"'violation' has {viol.size} entries but there are {matrix.shape[0]} "
            "solutions."
        )
    return _fronts_from_domination(_constrained_domination_matrix(matrix, viol))


def crowding_distance(objectives: npt.ArrayLike) -> FloatArray:
    """Deb's crowding distance: a density estimate used as a diversity tiebreak.

    For each objective the solutions are sorted and every interior point is
    credited with the *normalised* gap between its two neighbours; the extremes
    receive infinity so they are always retained.  Summing across objectives
    gives the perimeter of the cuboid spanned by a solution's neighbours.
    Larger means more isolated, hence more valuable for spread.

    Normalising by each objective's range is what keeps the measure meaningful
    when the objectives have incomparable units - here a return of order 0.01,
    a CVaR of order 0.1 and a cost of order 0.001.  Without it the widest-range
    objective would dictate the entire selection.

    Parameters
    ----------
    objectives
        Minimisation-sense objective matrix for one front, shape ``(P, m)``.

    Returns
    -------
    FloatArray
        Crowding distance per solution, shape ``(P,)``; ``inf`` at the extremes.
    """
    matrix = _as_2d(objectives, "objectives")
    n_points, n_objectives = matrix.shape
    if n_points <= 2:
        return np.full(n_points, np.inf)

    distance = np.zeros(n_points)
    for j in range(n_objectives):
        order = np.argsort(matrix[:, j], kind="stable")
        values = matrix[order, j]
        span = values[-1] - values[0]
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        if span <= 0.0:
            continue  # degenerate objective contributes nothing to spread
        distance[order[1:-1]] += (values[2:] - values[:-2]) / span
    return distance


# ==============================================================================
# 3. Quality indicators
# ==============================================================================


def default_reference_point(
    objectives: npt.ArrayLike, margin: float = 0.1
) -> FloatArray:
    """Nadir-plus-margin reference point for hypervolume.

    Hypervolume is only comparable across fronts when every front is measured
    against the *same* reference point, so the usual protocol is to derive it
    once from the union of all fronts being compared and reuse it.

    Parameters
    ----------
    objectives
        Minimisation-sense objective matrix, shape ``(P, m)``.
    margin
        Fraction of each objective's range added beyond the worst value, so
        that boundary solutions still contribute volume.

    Returns
    -------
    FloatArray
        Reference point, shape ``(m,)``.
    """
    matrix = _as_2d(objectives, "objectives")
    worst = matrix.max(axis=0)
    span = worst - matrix.min(axis=0)
    # A degenerate objective (zero span) still needs a strictly worse reference.
    span = np.where(span > 0.0, span, np.maximum(np.abs(worst), 1.0))
    return worst + margin * span


def _hypervolume_2d(points: FloatArray, reference: FloatArray) -> float:
    """Exact 2-D hypervolume by a single sweep along the first objective."""
    order = np.argsort(points[:, 0], kind="stable")
    ordered = points[order]
    volume = 0.0
    previous_f2 = reference[1]
    for f1, f2 in ordered:
        if f2 >= previous_f2:
            continue  # dominated within the sweep, contributes nothing
        volume += (reference[0] - f1) * (previous_f2 - f2)
        previous_f2 = f2
    return volume


def _hypervolume_recursive(points: FloatArray, reference: FloatArray) -> float:
    """Exact hypervolume by dimension slicing (the HSO scheme).

    Slices the dominated region along the last objective: between consecutive
    distinct coordinate values, the cross-section is the ``(m-1)``-dimensional
    hypervolume of the points that have already "started".  Exact for any
    dimension; the cost grows quickly with ``m``, which is acceptable because
    fronts of practical interest to a risk desk have two or three objectives.
    """
    if reference.size == 2:
        return _hypervolume_2d(points, reference)

    order = np.argsort(points[:, -1], kind="stable")
    ordered = points[order]
    volume = 0.0
    for i in range(ordered.shape[0]):
        upper = ordered[i + 1, -1] if i + 1 < ordered.shape[0] else reference[-1]
        depth = upper - ordered[i, -1]
        if depth <= 0.0:
            continue
        projected = ordered[: i + 1, :-1]
        keep = non_dominated_mask(projected)
        volume += depth * _hypervolume_recursive(projected[keep], reference[:-1])
    return volume


def hypervolume(objectives: npt.ArrayLike, reference: npt.ArrayLike) -> float:
    """Volume of the region dominated by a front and bounded by a reference point.

    The single most informative unary indicator: it is the only widely used one
    that is strictly monotonic with respect to Pareto dominance, so a front
    that genuinely improves can never score worse.  It rewards convergence and
    spread simultaneously.

    Parameters
    ----------
    objectives
        Minimisation-sense objective matrix, shape ``(P, m)``.
    reference
        Upper-bound point, shape ``(m,)``; must be dominated by (i.e. worse on
        every objective than) the solutions that are to count.

    Returns
    -------
    float
        The dominated hypervolume.  Zero when no solution beats the reference.
    """
    matrix = _as_2d(objectives, "objectives")
    ref = _as_1d(reference, "reference")
    if ref.size != matrix.shape[1]:
        raise ValueError(
            f"'reference' has {ref.size} entries but the front has "
            f"{matrix.shape[1]} objectives."
        )

    # Only points strictly better than the reference on every objective
    # contribute; the rest are clipped away entirely.
    inside = (matrix < ref).all(axis=1)
    if not inside.any():
        return 0.0
    candidates = matrix[inside]
    candidates = candidates[non_dominated_mask(candidates)]

    if ref.size == 1:
        return float(ref[0] - candidates[:, 0].min())
    return float(_hypervolume_recursive(candidates, ref))


def _normalised(matrix: FloatArray, scale: FloatArray | None) -> FloatArray:
    """Divide by a per-objective scale, guarding against zero ranges."""
    if scale is None:
        return matrix
    safe = np.where(np.abs(scale) > 0.0, scale, 1.0)
    return matrix / safe


def _pairwise_min_distance(
    approximation: FloatArray, reference: FloatArray, scale: FloatArray | None
) -> FloatArray:
    """Distance from each reference point to its nearest approximation point."""
    a = _normalised(approximation, scale)
    r = _normalised(reference, scale)
    differences = r[:, None, :] - a[None, :, :]
    return np.sqrt((differences**2).sum(axis=-1)).min(axis=1)


def inverted_generational_distance(
    approximation: npt.ArrayLike,
    reference: npt.ArrayLike,
    scale: npt.ArrayLike | None = None,
) -> float:
    """Mean distance from each *true* front point to the nearest approximation.

    IGD penalises both poor convergence and poor coverage: a front that nails
    one region of the true front but misses another scores badly, because the
    missed reference points have no near neighbour.  This makes it the natural
    metric when a genuine reference front exists - which, thanks to the
    Tchebycheff LP, it does here.

    Parameters
    ----------
    approximation
        The candidate front (typically NSGA-II), shape ``(P, m)``.
    reference
        The true or best-known front, shape ``(R, m)``.
    scale
        Per-objective normalisation, shape ``(m,)``.  Strongly recommended:
        without it, whichever objective has the largest numerical range
        dominates the Euclidean distance and the score stops meaning anything.

    Returns
    -------
    float
        Mean nearest-neighbour distance, in normalised units.  Lower is better;
        zero means every reference point is exactly matched.
    """
    a = _as_2d(approximation, "approximation")
    r = _as_2d(reference, "reference")
    s = None if scale is None else _as_1d(scale, "scale")
    return float(_pairwise_min_distance(a, r, s).mean())


def generational_distance(
    approximation: npt.ArrayLike,
    reference: npt.ArrayLike,
    scale: npt.ArrayLike | None = None,
) -> float:
    """Mean distance from each approximation point to the nearest true point.

    The dual of IGD: it measures *convergence only*.  A single perfectly placed
    point scores a perfect GD while covering almost none of the front, which is
    exactly why GD must be read alongside IGD and never on its own.

    Parameters
    ----------
    See :func:`inverted_generational_distance`; the roles are swapped.

    Returns
    -------
    float
        Mean nearest-neighbour distance. Lower is better.
    """
    a = _as_2d(approximation, "approximation")
    r = _as_2d(reference, "reference")
    s = None if scale is None else _as_1d(scale, "scale")
    return float(_pairwise_min_distance(r, a, s).mean())


def spacing(objectives: npt.ArrayLike, scale: npt.ArrayLike | None = None) -> float:
    """Schott's spacing metric: uniformity of the gaps along a front.

    The standard deviation of each solution's distance to its nearest
    neighbour.  Zero means perfectly even spacing.  It says nothing about
    whether the front is in the right place - it is a diversity diagnostic
    only, to be read next to hypervolume or IGD.

    Parameters
    ----------
    objectives
        Minimisation-sense objective matrix, shape ``(P, m)``.
    scale
        Per-objective normalisation, shape ``(m,)``.

    Returns
    -------
    float
        Standard deviation of nearest-neighbour distances; ``0.0`` for fronts
        with fewer than two points.
    """
    matrix = _as_2d(objectives, "objectives")
    if matrix.shape[0] < 2:
        return 0.0
    normalised = _normalised(matrix, None if scale is None else _as_1d(scale, "scale"))
    differences = normalised[:, None, :] - normalised[None, :, :]
    distances = np.sqrt((differences**2).sum(axis=-1))
    np.fill_diagonal(distances, np.inf)
    return float(distances.min(axis=1).std())


def coverage(front_a: npt.ArrayLike, front_b: npt.ArrayLike) -> float:
    """Zitzler's C-metric: fraction of ``front_b`` weakly dominated by ``front_a``.

    ``coverage(A, B) == 1`` means every member of ``B`` is matched or beaten by
    something in ``A``.  The metric is asymmetric by design, so both directions
    should always be reported: ``C(A,B)`` high together with ``C(B,A)`` low is
    the only combination that licenses the claim that ``A`` is the better front.

    Parameters
    ----------
    front_a, front_b
        Minimisation-sense objective matrices.

    Returns
    -------
    float
        Fraction in ``[0, 1]``.
    """
    a = _as_2d(front_a, "front_a")
    b = _as_2d(front_b, "front_b")
    no_worse = (a[:, None, :] <= b[None, :, :]).all(axis=-1)
    strictly_better = (a[:, None, :] < b[None, :, :]).any(axis=-1)
    weakly_dominated = (no_worse & strictly_better).any(axis=0)
    return float(weakly_dominated.mean())


# ==============================================================================
# 4. Result container
# ==============================================================================


@dataclass
class ParetoFront:
    """A set of mutually non-dominated portfolios with their objective values.

    Attributes
    ----------
    weights
        Portfolio weights, shape ``(P, n)``.
    objectives
        Objective values in **natural units** (return as a return, CVaR as a
        loss, cost as a cost), shape ``(P, m)``.  This is what a human reads.
    minimisation_objectives
        The same values sense-normalised so smaller is better, shape
        ``(P, m)``.  This is what every indicator consumes; mixing the two up
        is the single easiest way to get a silently wrong comparison, so both
        are stored explicitly rather than reconstructed at each call site.
    names
        Objective labels, in column order.
    method
        Provenance label, e.g. ``"NSGA-II"`` or ``"Tchebycheff-LP"``.
    metadata
        Free-form run diagnostics - generation count, solver messages, timings.
    history
        Per-generation convergence record for population methods; empty for the
        scalarisation route.
    """

    weights: FloatArray
    objectives: FloatArray
    minimisation_objectives: FloatArray
    names: list[str]
    method: str = ""
    metadata: dict[str, object] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)

    def __len__(self) -> int:
        """Number of portfolios on the front."""
        return int(self.weights.shape[0])

    def __iter__(self) -> Iterator[tuple[FloatArray, FloatArray]]:
        """Iterate over ``(weights, natural objectives)`` pairs."""
        return iter(zip(self.weights, self.objectives))

    @property
    def n_objectives(self) -> int:
        """Number of objectives ``m``."""
        return int(self.objectives.shape[1])

    @property
    def n_assets(self) -> int:
        """Number of instruments ``n``."""
        return int(self.weights.shape[1])

    def filter_non_dominated(self) -> "ParetoFront":
        """Drop any dominated or duplicated rows, preserving order.

        Scalarisation runs can return the same optimum for neighbouring weight
        vectors, and a truncated NSGA-II population can retain dominated
        members.  Indicators assume a clean front, so this is applied before
        every comparison.
        """
        keep = non_dominated_mask(self.minimisation_objectives)
        indices = np.flatnonzero(keep)
        # Deduplicate on the objective vector; two weight vectors mapping to the
        # same point add nothing to any indicator but inflate cardinality.
        _, unique = np.unique(
            self.minimisation_objectives[indices].round(12), axis=0, return_index=True
        )
        indices = indices[np.sort(unique)]
        return ParetoFront(
            weights=self.weights[indices],
            objectives=self.objectives[indices],
            minimisation_objectives=self.minimisation_objectives[indices],
            names=list(self.names),
            method=self.method,
            metadata=dict(self.metadata),
            history=list(self.history),
        )

    def hypervolume(self, reference: npt.ArrayLike) -> float:
        """Hypervolume of this front against a shared reference point."""
        return hypervolume(self.minimisation_objectives, reference)

    def to_frame(self):
        """Return a ``pandas.DataFrame`` of objectives and weights.

        Imported lazily so that the optimisation core carries no hard pandas
        dependency - handy when the engine is embedded in a service that only
        needs arrays.
        """
        import pandas as pd

        frame = pd.DataFrame(self.objectives, columns=self.names)
        for j in range(self.n_assets):
            frame[f"w{j}"] = self.weights[:, j]
        return frame

    def __str__(self) -> str:  # pragma: no cover - presentation only
        if len(self) == 0:
            return f"ParetoFront({self.method}, empty)"
        lines = [f"ParetoFront({self.method}, {len(self)} portfolios)"]
        for name, column in zip(self.names, self.objectives.T):
            lines.append(f"  {name:<18} [{column.min():.6f}, {column.max():.6f}]")
        return "\n".join(lines)


# ==============================================================================
# 5. Genetic operators tailored to budget-constrained weights
# ==============================================================================


class CrossoverOperator(ABC):
    """Abstract recombination operator over portfolio weight vectors."""

    @abstractmethod
    def recombine(
        self,
        parents_a: FloatArray,
        parents_b: FloatArray,
        lower: FloatArray,
        upper: FloatArray,
        rng: np.random.Generator,
    ) -> tuple[FloatArray, FloatArray]:
        """Produce two child populations from two parent populations.

        Parameters
        ----------
        parents_a, parents_b
            Parent matrices of shape ``(K, n)``, paired row-wise.
        lower, upper
            Finite box bounds, shape ``(n,)``.
        rng
            The algorithm's random generator, threaded through so that a run is
            reproducible from a single seed.

        Returns
        -------
        (FloatArray, FloatArray)
            Two child matrices of shape ``(K, n)``.  Children need not satisfy
            the budget constraint; the driver projects them.
        """


class MutationOperator(ABC):
    """Abstract mutation operator over portfolio weight vectors."""

    @abstractmethod
    def mutate(
        self,
        population: FloatArray,
        lower: FloatArray,
        upper: FloatArray,
        rng: np.random.Generator,
    ) -> FloatArray:
        """Return a mutated copy of ``population``, shape ``(P, n)``."""


class SimulatedBinaryCrossover(CrossoverOperator):
    """SBX (Deb & Agrawal, 1995), the canonical real-coded crossover.

    Children are placed around the parents with a spread controlled by the
    distribution index ``eta``: large ``eta`` concentrates them near the
    parents (exploitation), small ``eta`` scatters them (exploration).  SBX is
    the default in NSGA-II because it mimics the behaviour of single-point
    binary crossover on real variables - the search step self-scales with the
    parents' separation, so it contracts automatically as the population
    converges.

    **Budget behaviour.**  The SBX step itself gives ``c1 + c2 = p1 + p2``
    gene-wise, so the *pair* conserves total weight even though neither child
    does individually.  That identity survives only until the children are
    clipped into the box, which this operator must do because SBX knows
    nothing about bounds - so in general **neither** invariant reaches the
    driver intact.  Feasibility is restored wholesale by
    :func:`project_onto_budget_box`.  That repair is what tailors a
    general-purpose operator to the simplex, and it is why the projection is
    exact rather than a normalisation: an inexact repair biases the search
    toward whatever direction the repair happens to push.  Where a
    repair-free operator is preferred, use :class:`SimplexBlendCrossover`.

    Parameters
    ----------
    probability
        Chance that a given parent pair is recombined at all.
    eta
        Distribution index; 15 is the NSGA-II default.
    gene_probability
        Per-gene chance of swapping, conventionally 0.5.
    """

    def __init__(
        self,
        probability: float = 0.9,
        eta: float = 15.0,
        gene_probability: float = 0.5,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"'probability' must be in [0, 1], got {probability!r}.")
        if eta <= 0.0:
            raise ValueError(f"'eta' must be positive, got {eta!r}.")
        self.probability = float(probability)
        self.eta = float(eta)
        self.gene_probability = float(gene_probability)

    def recombine(
        self,
        parents_a: FloatArray,
        parents_b: FloatArray,
        lower: FloatArray,
        upper: FloatArray,
        rng: np.random.Generator,
    ) -> tuple[FloatArray, FloatArray]:
        """Apply SBX gene-wise to the paired parents."""
        n_pairs, n_genes = parents_a.shape
        child_a = parents_a.copy()
        child_b = parents_b.copy()

        active = (rng.random(n_pairs) <= self.probability)[:, None] & (
            rng.random((n_pairs, n_genes)) <= self.gene_probability
        )

        # Spread factor from the SBX density; u is in [0, 1) so 1 - u > 0.
        u = rng.random((n_pairs, n_genes))
        exponent = 1.0 / (self.eta + 1.0)
        beta = np.where(
            u <= 0.5,
            (2.0 * u) ** exponent,
            (1.0 / (2.0 * (1.0 - u))) ** exponent,
        )

        mean = 0.5 * (parents_a + parents_b)
        half_difference = 0.5 * beta * (parents_a - parents_b)
        child_a = np.where(active, mean + half_difference, child_a)
        child_b = np.where(active, mean - half_difference, child_b)

        return np.clip(child_a, lower, upper), np.clip(child_b, lower, upper)


class SimplexBlendCrossover(CrossoverOperator):
    """Affine blend crossover that preserves the budget **exactly**.

    A child is an affine combination ``c = lam*a + (1-lam)*b`` whose
    coefficients sum to one, so ``sum(c) = lam*sum(a) + (1-lam)*sum(b) =
    budget`` whenever both parents satisfy the budget - no repair needed, for
    any ``lam``, including values outside ``[0, 1]``.

    Drawing ``lam`` from ``[-alpha, 1+alpha]`` (BLX-style) permits
    extrapolation beyond the parents, which matters because pure interpolation
    with ``lam in [0,1]`` can only ever shrink the population's convex hull -
    a line-search operator that collapses diversity within a few dozen
    generations.

    ``lam`` is drawn **per individual**, not per gene: a per-gene draw would
    break the coefficient-sum identity and forfeit the whole advantage.

    Only the box can be violated (by extrapolation), and clipping restores it
    at the cost of a small budget error that the driver's projection then
    absorbs.  Useful as a low-repair alternative to SBX, especially with tight
    box constraints where SBX children are frequently clipped.

    Parameters
    ----------
    probability
        Chance that a given parent pair is recombined.
    alpha
        Extrapolation reach beyond the parent segment.
    """

    def __init__(self, probability: float = 0.9, alpha: float = 0.3) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"'probability' must be in [0, 1], got {probability!r}.")
        if alpha < 0.0:
            raise ValueError(f"'alpha' must be non-negative, got {alpha!r}.")
        self.probability = float(probability)
        self.alpha = float(alpha)

    def recombine(
        self,
        parents_a: FloatArray,
        parents_b: FloatArray,
        lower: FloatArray,
        upper: FloatArray,
        rng: np.random.Generator,
    ) -> tuple[FloatArray, FloatArray]:
        """Blend each parent pair with a single shared coefficient."""
        n_pairs = parents_a.shape[0]
        lam = rng.uniform(-self.alpha, 1.0 + self.alpha, size=(n_pairs, 1))
        active = (rng.random(n_pairs) <= self.probability)[:, None]

        child_a = np.where(active, lam * parents_a + (1.0 - lam) * parents_b, parents_a)
        child_b = np.where(active, (1.0 - lam) * parents_a + lam * parents_b, parents_b)
        return np.clip(child_a, lower, upper), np.clip(child_b, lower, upper)


class PolynomialMutation(MutationOperator):
    """Polynomial mutation (Deb), the NSGA-II companion to SBX.

    Perturbs a gene by a polynomial-distributed step whose reach is set by the
    distribution index ``eta``: the perturbation is concentrated near zero and
    respects the variable's own bounds, so no clipping bias is introduced.

    Like SBX it does not preserve the budget, and like SBX it is repaired by
    projection.

    Parameters
    ----------
    probability
        Per-gene mutation chance.  ``None`` selects the standard ``1/n``, which
        mutates about one gene per individual regardless of dimension.
    eta
        Distribution index; 20 is the NSGA-II default.
    """

    def __init__(self, probability: float | None = None, eta: float = 20.0) -> None:
        if probability is not None and not 0.0 <= probability <= 1.0:
            raise ValueError(f"'probability' must be in [0, 1], got {probability!r}.")
        if eta <= 0.0:
            raise ValueError(f"'eta' must be positive, got {eta!r}.")
        self.probability = probability
        self.eta = float(eta)

    def mutate(
        self,
        population: FloatArray,
        lower: FloatArray,
        upper: FloatArray,
        rng: np.random.Generator,
    ) -> FloatArray:
        """Apply polynomial mutation gene-wise."""
        n_individuals, n_genes = population.shape
        rate = 1.0 / n_genes if self.probability is None else self.probability
        mutated = population.copy()

        span = upper - lower
        # A pinned weight (zero span) cannot be perturbed without leaving the box.
        active = (rng.random((n_individuals, n_genes)) <= rate) & (span > 0.0)
        if not active.any():
            return mutated

        u = rng.random((n_individuals, n_genes))
        exponent = 1.0 / (self.eta + 1.0)
        # Distance to each bound, normalised, so the step never overshoots.
        to_lower = np.divide(population - lower, span, out=np.zeros_like(population), where=span > 0)
        to_upper = np.divide(upper - population, span, out=np.zeros_like(population), where=span > 0)

        delta_lower = (
            2.0 * u + (1.0 - 2.0 * u) * (1.0 - to_lower) ** (self.eta + 1.0)
        ) ** exponent - 1.0
        delta_upper = 1.0 - (
            2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - to_upper) ** (self.eta + 1.0)
        ) ** exponent
        delta = np.where(u <= 0.5, delta_lower, delta_upper)

        mutated = np.where(active, population + delta * span, population)
        return np.clip(mutated, lower, upper)


class WeightTransferMutation(MutationOperator):
    """Budget-preserving mutation modelled on an actual trade.

    Picks a source and a sink instrument and moves weight from one to the
    other.  Because the two changes are equal and opposite, ``sum(x)`` is
    conserved **exactly**, and because the transferred amount is capped by the
    available headroom, the box is conserved exactly too.  No repair, no
    projection, no drift.

    Beyond the numerics this is the economically meaningful move: every
    mutation is a rebalancing trade the desk could actually execute, sized
    within the position limits.  It also composes well with a turnover-based
    hedging cost, since the size of the perturbation is exactly the turnover it
    generates.

    Its weakness is the mirror of its strength - it only ever moves along
    simplex edges, so it explores the interior slowly.  Best used *alongside* a
    polynomial mutation rather than instead of it.

    Parameters
    ----------
    probability
        Per-individual chance of being mutated.
    n_trades
        Number of transfers applied to a selected individual.
    max_fraction
        Largest share of the available headroom that a single transfer may
        move.  Small values give a fine-grained local search.
    """

    def __init__(
        self,
        probability: float = 0.2,
        n_trades: int = 1,
        max_fraction: float = 0.25,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"'probability' must be in [0, 1], got {probability!r}.")
        if n_trades < 1:
            raise ValueError(f"'n_trades' must be at least 1, got {n_trades}.")
        if not 0.0 < max_fraction <= 1.0:
            raise ValueError(
                f"'max_fraction' must be in (0, 1], got {max_fraction!r}."
            )
        self.probability = float(probability)
        self.n_trades = int(n_trades)
        self.max_fraction = float(max_fraction)

    def mutate(
        self,
        population: FloatArray,
        lower: FloatArray,
        upper: FloatArray,
        rng: np.random.Generator,
    ) -> FloatArray:
        """Apply weight transfers to a random subset of the population."""
        n_individuals, n_genes = population.shape
        if n_genes < 2:
            return population.copy()

        mutated = population.copy()
        selected = np.flatnonzero(rng.random(n_individuals) <= self.probability)
        if selected.size == 0:
            return mutated

        for _ in range(self.n_trades):
            source = rng.integers(0, n_genes, size=selected.size)
            # Offsetting guarantees sink != source without a rejection loop.
            sink = (source + 1 + rng.integers(0, n_genes - 1, size=selected.size)) % n_genes

            # Headroom: what the source can give and the sink can take.
            can_give = mutated[selected, source] - lower[source]
            can_take = upper[sink] - mutated[selected, sink]
            headroom = np.maximum(np.minimum(can_give, can_take), 0.0)

            amount = rng.random(selected.size) * self.max_fraction * headroom
            mutated[selected, source] -= amount
            mutated[selected, sink] += amount

        return mutated


# ==============================================================================
# 6. Approach A - NSGA-II
# ==============================================================================


@dataclass
class NSGAIIConfig:
    """Hyper-parameters for :class:`NSGAII`.

    Attributes
    ----------
    population_size
        Number of portfolios carried between generations.  Must be even so the
        mating pool pairs exactly.  For three objectives, 100-200 is the usual
        range; too small and the crowding distance cannot maintain spread.
    n_generations
        Number of generations.
    seed
        Seed for the run's generator.  A run is fully reproducible from it.
    initial_concentration
        Dirichlet concentration for the initial population.  ``1.0`` is uniform
        on the simplex; below 1 favours concentrated portfolios near the
        vertices, above 1 favours diversified ones near the centre.
    track_every
        Record convergence diagnostics every this many generations.  Computing
        hypervolume each generation is wasteful for a 3-D front, so it is
        sampled instead.
    verbose
        Log progress at INFO level.
    """

    population_size: int = 120
    n_generations: int = 200
    seed: int | None = None
    initial_concentration: float = 1.0
    track_every: int = 10
    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.population_size < 4 or self.population_size % 2 != 0:
            raise ValueError(
                f"'population_size' must be even and at least 4, "
                f"got {self.population_size}."
            )
        if self.n_generations < 1:
            raise ValueError(
                f"'n_generations' must be at least 1, got {self.n_generations}."
            )
        if self.initial_concentration <= 0.0:
            raise ValueError(
                f"'initial_concentration' must be positive, "
                f"got {self.initial_concentration}."
            )
        if self.track_every < 1:
            raise ValueError(f"'track_every' must be at least 1, got {self.track_every}.")


class NSGAII:
    """Elitist non-dominated sorting genetic algorithm, specialised for portfolios.

    Generation loop
    ---------------
    1. **Binary tournament** on ``(constraint violation, rank, crowding)``.
    2. **Crossover** and **mutation** with the configured operators.
    3. **Projection** of every child onto the budget simplex, so the entire
       offspring population is feasible with respect to budget and box.
    4. **Evaluation** of the objective vector and the residual violation.
    5. **Elitist survival**: parents and offspring are pooled, sorted by
       constrained non-domination, and truncated to the population size using
       crowding distance to break ties inside the boundary front.

    Step 5 is what makes the algorithm elitist: a good solution can never be
    lost, because it competes directly against its own offspring.

    Parameters
    ----------
    objectives
        The criteria to optimise.
    constraints
        Feasible set.  Budget and box are enforced by projection; anything else
        enters through constrained domination.
    config
        Hyper-parameters.
    crossover
        Recombination operator.  Defaults to :class:`SimulatedBinaryCrossover`.
    mutation
        Mutation operator.  Defaults to :class:`PolynomialMutation`.
    expected_returns
        Mean return vector, needed only when ``min_expected_return`` is set.
        Auto-detected from an :class:`ExpectedReturnObjective` in the set when
        not supplied.
    """

    def __init__(
        self,
        objectives: ObjectiveSet,
        constraints: PortfolioConstraints | None = None,
        config: NSGAIIConfig | None = None,
        crossover: CrossoverOperator | None = None,
        mutation: MutationOperator | None = None,
        expected_returns: npt.ArrayLike | None = None,
    ) -> None:
        self.objectives = objectives
        self.constraints = (
            constraints if constraints is not None else PortfolioConstraints()
        )
        self.config = config if config is not None else NSGAIIConfig()
        self.crossover = crossover if crossover is not None else SimulatedBinaryCrossover()
        self.mutation = mutation if mutation is not None else PolynomialMutation()
        self._expected_returns = _resolve_expected_returns(
            objectives, self.constraints, expected_returns
        )

        self._n_assets = objectives.n_assets
        budget = self.constraints.budget
        self._lower, self._upper = _finite_bounds(
            self.constraints, self._n_assets, reference_scale=budget or 1.0
        )

    # -- population initialisation -------------------------------------------

    def _initial_population(
        self, rng: np.random.Generator, seeds: FloatArray | None
    ) -> FloatArray:
        """Sample a feasible starting population.

        Dirichlet sampling gives points spread over the simplex rather than
        clustered near its centre, which is what independent uniform draws
        followed by normalisation would produce.  Optional seed portfolios -
        typically the single-objective optima - are injected verbatim so the
        run starts with the extremes of the front already anchored.
        """
        size = self.config.population_size
        n_seeds = 0 if seeds is None else min(seeds.shape[0], size)

        raw = rng.dirichlet(
            np.full(self._n_assets, self.config.initial_concentration),
            size=size - n_seeds,
        )
        budget = self.constraints.budget
        if budget is not None:
            raw = raw * budget

        population = raw
        if n_seeds:
            population = np.vstack([seeds[:n_seeds], raw])
        return project_onto_budget_box(
            population, self._lower, self._upper, self.constraints.budget
        )

    # -- evaluation -----------------------------------------------------------

    def _evaluate(self, population: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return the minimisation-sense objective matrix and violation vector."""
        values = self.objectives.minimisation_batch(population)
        violation = constraint_violation(
            population, self.constraints, self._expected_returns
        )
        return values, violation

    # -- selection ------------------------------------------------------------

    @staticmethod
    def _ranks_from_fronts(fronts: list[IntArray], size: int) -> IntArray:
        """Map each individual to the index of the front containing it."""
        rank = np.empty(size, dtype=np.int_)
        for level, front in enumerate(fronts):
            rank[front] = level
        return rank

    @staticmethod
    def _crowding_from_fronts(
        fronts: list[IntArray], values: FloatArray, size: int
    ) -> FloatArray:
        """Crowding distance computed within each front, assembled per individual."""
        crowding = np.zeros(size)
        for front in fronts:
            crowding[front] = crowding_distance(values[front])
        return crowding

    def _tournament(
        self,
        rank: IntArray,
        crowding: FloatArray,
        violation: FloatArray,
        n_winners: int,
        rng: np.random.Generator,
    ) -> IntArray:
        """Binary tournament under the constrained-domination preference order.

        Preference, in strict priority: feasible over infeasible; among
        infeasible, less violation; among feasible, lower rank; ties broken by
        larger crowding distance.
        """
        size = rank.size
        left = rng.integers(0, size, size=n_winners)
        right = rng.integers(0, size, size=n_winners)

        feasible = violation <= FEASIBILITY_TOLERANCE
        left_feasible, right_feasible = feasible[left], feasible[right]

        left_wins = np.where(
            left_feasible != right_feasible,
            left_feasible,  # exactly one is feasible: it wins
            np.where(
                ~left_feasible,  # both infeasible: less violation wins
                violation[left] < violation[right],
                np.where(  # both feasible: rank, then crowding
                    rank[left] != rank[right],
                    rank[left] < rank[right],
                    crowding[left] > crowding[right],
                ),
            ),
        )
        return np.where(left_wins, left, right)

    # -- driver ---------------------------------------------------------------

    def run(self, seed_solutions: npt.ArrayLike | None = None) -> ParetoFront:
        """Execute the genetic algorithm and return the final Pareto front.

        Parameters
        ----------
        seed_solutions
            Optional starting portfolios of shape ``(S, n)`` injected into the
            initial population.  Seeding with the single-objective optima
            speeds convergence considerably, but note that it also *flatters*
            the algorithm in any benchmark against those same optima - a fair
            comparison against the Tchebycheff front should leave this unset.

        Returns
        -------
        ParetoFront
            The feasible non-dominated members of the final population, with a
            per-generation convergence history in ``history``.
        """
        rng = np.random.default_rng(self.config.seed)
        size = self.config.population_size

        seeds = None if seed_solutions is None else _as_2d(seed_solutions, "seed_solutions")
        population = self._initial_population(rng, seeds)
        values, violation = self._evaluate(population)

        fronts = constrained_non_dominated_sort(values, violation)
        rank = self._ranks_from_fronts(fronts, size)
        crowding = self._crowding_from_fronts(fronts, values, size)

        history: list[dict[str, float]] = []
        reference = default_reference_point(values)

        for generation in range(self.config.n_generations):
            # --- variation ---------------------------------------------------
            winners = self._tournament(rank, crowding, violation, size, rng)
            mating_pool = population[winners]
            parents_a = mating_pool[0::2]
            parents_b = mating_pool[1::2]

            child_a, child_b = self.crossover.recombine(
                parents_a, parents_b, self._lower, self._upper, rng
            )
            offspring = np.vstack([child_a, child_b])
            offspring = self.mutation.mutate(offspring, self._lower, self._upper, rng)
            # The single point where feasibility is restored for the whole
            # offspring population; every operator above may leave the simplex.
            offspring = project_onto_budget_box(
                offspring, self._lower, self._upper, self.constraints.budget
            )

            offspring_values, offspring_violation = self._evaluate(offspring)

            # --- elitist survival (mu + lambda) ------------------------------
            pooled = np.vstack([population, offspring])
            pooled_values = np.vstack([values, offspring_values])
            pooled_violation = np.concatenate([violation, offspring_violation])

            survivors = self._select_survivors(pooled_values, pooled_violation, size)
            population = pooled[survivors]
            values = pooled_values[survivors]
            violation = pooled_violation[survivors]

            fronts = constrained_non_dominated_sort(values, violation)
            rank = self._ranks_from_fronts(fronts, size)
            crowding = self._crowding_from_fronts(fronts, values, size)

            if (generation + 1) % self.config.track_every == 0 or generation == 0:
                record = self._diagnostics(generation, values, violation, reference)
                history.append(record)
                if self.config.verbose:
                    LOGGER.info(
                        "gen %4d | front %3d | hypervolume %.6e | feasible %3d",
                        generation + 1,
                        int(record["front_size"]),
                        record["hypervolume"],
                        int(record["n_feasible"]),
                    )

        return self._build_front(population, values, violation, history)

    @staticmethod
    def _select_survivors(
        values: FloatArray, violation: FloatArray, n_survivors: int
    ) -> IntArray:
        """Truncate a pooled population to ``n_survivors`` by rank then crowding.

        Fronts are accepted whole until one would overflow; that boundary front
        is then thinned by keeping the most isolated members, which is exactly
        how NSGA-II maintains spread without an explicit niching parameter.
        """
        fronts = constrained_non_dominated_sort(values, violation)
        chosen: list[IntArray] = []
        filled = 0

        for front in fronts:
            if filled + front.size <= n_survivors:
                chosen.append(front)
                filled += front.size
                if filled == n_survivors:
                    break
            else:
                distances = crowding_distance(values[front])
                order = np.argsort(-distances, kind="stable")
                chosen.append(front[order[: n_survivors - filled]])
                filled = n_survivors
                break
        return np.concatenate(chosen)

    @staticmethod
    def _diagnostics(
        generation: int,
        values: FloatArray,
        violation: FloatArray,
        reference: FloatArray,
    ) -> dict[str, float]:
        """Convergence record for one generation."""
        feasible = violation <= FEASIBILITY_TOLERANCE
        if feasible.any():
            feasible_values = values[feasible]
            front = feasible_values[non_dominated_mask(feasible_values)]
            volume = hypervolume(front, reference)
            front_size = float(front.shape[0])
        else:
            volume, front_size = 0.0, 0.0
        return {
            "generation": float(generation + 1),
            "hypervolume": float(volume),
            "front_size": front_size,
            "n_feasible": float(feasible.sum()),
            "mean_violation": float(violation.mean()),
        }

    def _build_front(
        self,
        population: FloatArray,
        values: FloatArray,
        violation: FloatArray,
        history: list[dict[str, float]],
    ) -> ParetoFront:
        """Extract the feasible non-dominated members as a :class:`ParetoFront`."""
        feasible = violation <= FEASIBILITY_TOLERANCE
        if not feasible.any():
            LOGGER.warning(
                "No feasible solution found after %d generations; returning the "
                "least-infeasible members (minimum violation %.3e). Check that "
                "'min_expected_return' is attainable.",
                self.config.n_generations,
                float(violation.min()),
            )
            feasible = violation <= violation.min() + FEASIBILITY_TOLERANCE

        indices = np.flatnonzero(feasible)
        keep = indices[non_dominated_mask(values[indices])]

        front = ParetoFront(
            weights=population[keep],
            objectives=self.objectives.to_natural(values[keep]),
            minimisation_objectives=values[keep],
            names=self.objectives.names,
            method="NSGA-II",
            metadata={
                "population_size": self.config.population_size,
                "n_generations": self.config.n_generations,
                "seed": self.config.seed,
                "crossover": type(self.crossover).__name__,
                "mutation": type(self.mutation).__name__,
                "n_evaluations": self.config.population_size
                * (self.config.n_generations + 1),
            },
            history=history,
        )
        return front.filter_non_dominated()


# ==============================================================================
# 7. Approach B - Tchebycheff scalarisation
# ==============================================================================


def simplex_lattice_weights(n_objectives: int, n_partitions: int) -> FloatArray:
    """Das-Dennis simplex-lattice weight vectors.

    Enumerates every vector whose components are multiples of
    ``1 / n_partitions`` and sum to one - a structured, uniformly spaced design
    on the weight simplex.  The count is
    ``C(n_partitions + m - 1, m - 1)``: for three objectives, 12 partitions
    gives 91 vectors.

    A structured design is preferable to random weights because it guarantees
    coverage of the boundary (the single-objective extremes) as well as the
    interior, which is precisely where a randomly sampled set tends to be thin.

    Parameters
    ----------
    n_objectives
        Number of objectives ``m``, at least 2.
    n_partitions
        Divisions per axis ``H``, at least 1.

    Returns
    -------
    FloatArray
        Weight vectors of shape ``(C(H+m-1, m-1), m)``; each row sums to 1.
    """
    if n_objectives < 2:
        raise ValueError(f"'n_objectives' must be at least 2, got {n_objectives}.")
    if n_partitions < 1:
        raise ValueError(f"'n_partitions' must be at least 1, got {n_partitions}.")

    def _compose(remaining: int, depth: int) -> list[list[int]]:
        """All non-negative integer compositions of ``remaining`` into ``depth`` parts."""
        if depth == 1:
            return [[remaining]]
        return [
            [head] + tail
            for head in range(remaining + 1)
            for tail in _compose(remaining - head, depth - 1)
        ]

    lattice = np.array(_compose(n_partitions, n_objectives), dtype=np.float64)
    return lattice / n_partitions


@dataclass(frozen=True)
class TchebycheffSolution:
    """A single scalarised optimum.

    Attributes
    ----------
    weights
        Optimal portfolio.
    objectives
        Objective values in natural units, shape ``(m,)``.
    minimisation_objectives
        The same, sense-normalised.
    lambda_vector
        The weight vector that produced this point.
    scalar_value
        Optimal value of the augmented Tchebycheff function.
    success, message
        Solver diagnostics.
    """

    weights: FloatArray
    objectives: FloatArray
    minimisation_objectives: FloatArray
    lambda_vector: FloatArray
    scalar_value: float
    success: bool
    message: str


def _resolve_expected_returns(
    objectives: ObjectiveSet,
    constraints: PortfolioConstraints,
    supplied: npt.ArrayLike | None,
) -> FloatArray | None:
    """Find the mean return vector needed by an expected-return floor.

    Prefers an explicit argument, then an :class:`ExpectedReturnObjective`
    inside the objective set.  Only raises when the constraint is actually in
    use, so problems without a return floor never have to supply one.
    """
    if supplied is not None:
        return _as_1d(supplied, "expected_returns")
    for objective in objectives:
        if isinstance(objective, ExpectedReturnObjective):
            return objective.expected_returns
    if constraints.min_expected_return is not None:
        raise ValueError(
            "'min_expected_return' is set but no expected-return vector is "
            "available. Pass 'expected_returns' explicitly or include an "
            "ExpectedReturnObjective in the objective set."
        )
    return None


def _normalise_lambda(weights: npt.ArrayLike, n_objectives: int) -> FloatArray:
    """Validate a weight vector and floor its components away from zero.

    A component of exactly zero removes an objective from the programme.  That
    is legitimate in the abstract, but it also removes the downward pressure
    that keeps that objective's epigraph tight, so the LP would be free to
    return a point whose reported value for it is meaningless.  Flooring at
    :data:`MIN_LAMBDA` keeps every epigraph tight while changing the trade-off
    by less than a part per million.
    """
    lam = _as_1d(weights, "lambda_vector")
    if lam.size != n_objectives:
        raise ValueError(
            f"'lambda_vector' has {lam.size} entries, expected {n_objectives}."
        )
    if np.any(lam < 0.0):
        raise ValueError("Tchebycheff weights must be non-negative.")
    if lam.sum() <= 0.0:
        raise ValueError("Tchebycheff weights must not be all zero.")
    return np.maximum(lam / lam.sum(), MIN_LAMBDA)


class TchebycheffLPSolver:
    """Exact augmented Tchebycheff scalarisation by linear programming.

    Formulation
    -----------
    With an epigraph variable ``t`` standing in for the ``max``:

    .. math::

        \\min_{x, w, t} \\;\\; t + \\rho \\sum_i \\tilde{\\lambda}_i
            \\big(f_i(x) - z^{*}_i\\big)
        \\quad \\text{s.t.} \\quad
        \\tilde{\\lambda}_i \\big(f_i(x) - z^{*}_i\\big) \\le t \\;\\; \\forall i,
        \\;\\; x \\in X,

    where each :math:`f_i` is replaced by its :class:`LinearEpigraph` -
    contributing that objective's own auxiliary variables and constraints - and
    :math:`\\tilde{\\lambda}_i = \\lambda_i / (\\text{nadir}_i -
    \\text{ideal}_i)` when normalisation is enabled.

    Variable layout: ``[x (n) | aux_0 | aux_1 | ... | aux_{m-1} | t (1)]``.

    Why the epigraphs are tight
    ---------------------------
    Replacing :math:`f_i(x)` by an inner minimisation is only valid if the
    outer problem pushes that minimisation down.  Here each :math:`f_i`
    appears with coefficient :math:`\\rho \\tilde{\\lambda}_i > 0` in the
    objective *and* under a ``<= t`` constraint whose right-hand side is being
    minimised.  Both act downward, so every auxiliary block settles at exactly
    the value that reproduces :math:`f_i(x)`.  This is why :math:`\\rho` must
    be strictly positive and why :data:`MIN_LAMBDA` floors the weights - with
    either at zero the corresponding epigraph goes slack and the reported value
    for that objective becomes an upper bound rather than the value.

    Why normalisation matters
    -------------------------
    Expected return is of order 0.01, CVaR of order 0.1, hedging cost of order
    0.001.  Applied to raw values, a uniform :math:`\\lambda` is not uniform at
    all - it silently weights CVaR a hundred times more than cost, and the
    resulting "front" bunches into one corner.  Scaling each objective by its
    ideal-to-nadir range fixes this, and since it is only a positive rescaling
    of :math:`\\lambda` the problem stays linear.

    Parameters
    ----------
    objectives
        All members must be LP-representable; :class:`VarianceObjective` and
        anything else non-linear must go through
        :class:`TchebycheffNonlinearSolver`.
    constraints
        Feasible set :math:`X`.
    expected_returns
        Mean return vector for an expected-return floor; auto-detected when
        omitted.
    rho
        Augmentation coefficient.  Too small and weakly Pareto-optimal points
        survive; too large and the solution drifts toward the weighted-sum
        optimum, losing access to non-convex regions.  ``1e-4`` is a
        conventional compromise.
    normalise
        Scale objectives by their ideal-to-nadir range.  Leave enabled.
    solver_options
        Passed through to :func:`scipy.optimize.linprog`.

    Raises
    ------
    ValueError
        If any objective lacks an LP representation.
    """

    def __init__(
        self,
        objectives: ObjectiveSet,
        constraints: PortfolioConstraints | None = None,
        expected_returns: npt.ArrayLike | None = None,
        rho: float = 1e-4,
        normalise: bool = True,
        solver_options: dict[str, object] | None = None,
    ) -> None:
        if not objectives.supports_lp:
            offenders = [obj.name for obj in objectives if not obj.supports_lp]
            raise ValueError(
                f"TchebycheffLPSolver requires every objective to be "
                f"LP-representable; {offenders} are not. Use "
                f"TchebycheffNonlinearSolver or NSGAII instead."
            )
        if rho <= 0.0:
            raise ValueError(
                f"'rho' must be strictly positive for the epigraph "
                f"representation to be tight, got {rho!r}."
            )

        self.objectives = objectives
        self.constraints = (
            constraints if constraints is not None else PortfolioConstraints()
        )
        self.rho = float(rho)
        self.normalise = bool(normalise)
        self.solver_options = solver_options
        self._expected_returns = _resolve_expected_returns(
            objectives, self.constraints, expected_returns
        )

        self._n_assets = objectives.n_assets
        self._epigraphs: list[LinearEpigraph] = [obj.lp_epigraph() for obj in objectives]
        self._aux_offsets: list[int] = []
        offset = self._n_assets
        for epigraph in self._epigraphs:
            self._aux_offsets.append(offset)
            offset += epigraph.n_aux
        self._n_variables = offset + 1  # + t
        self._t_index = offset

        self._ideal: FloatArray | None = None
        self._nadir: FloatArray | None = None
        self._payoff: FloatArray | None = None

    # -- reference points -----------------------------------------------------

    def payoff_table(self) -> FloatArray:
        """Compute the payoff table, and with it the ideal and nadir points.

        Row ``i`` holds every objective evaluated at the portfolio that
        minimises objective ``i`` alone.  The diagonal is therefore the **ideal
        point** (each objective at its unconstrained-by-others best), and the
        column-wise maxima give the standard **nadir estimate**.

        The single-objective solves keep only the relevant objective's
        epigraph, so the other auxiliary blocks are absent and cannot go slack.
        Every objective is then evaluated *directly* from the resulting ``x``
        rather than read out of the LP, which sidesteps the tightness question
        entirely.

        Returns
        -------
        FloatArray
            Payoff table of shape ``(m, m)`` in minimisation-sense units.
        """
        if self._payoff is not None:
            return self._payoff

        n_objectives = self.objectives.n_objectives
        table = np.empty((n_objectives, n_objectives))
        for i in range(n_objectives):
            x = self._solve_single_objective(i)
            table[i] = self.objectives.minimisation_batch(x.reshape(1, -1))[0]

        self._payoff = table
        self._ideal = np.diag(table).copy()
        self._nadir = table.max(axis=0)
        return table

    @property
    def ideal(self) -> FloatArray:
        """Ideal point :math:`z^{*}` in minimisation units."""
        if self._ideal is None:
            self.payoff_table()
        assert self._ideal is not None
        return self._ideal

    @property
    def nadir(self) -> FloatArray:
        """Nadir estimate from the payoff table, in minimisation units."""
        if self._nadir is None:
            self.payoff_table()
        assert self._nadir is not None
        return self._nadir

    @property
    def objective_scale(self) -> FloatArray:
        """Ideal-to-nadir range per objective, floored away from zero."""
        if not self.normalise:
            return np.ones(self.objectives.n_objectives)
        span = self.nadir - self.ideal
        return np.where(np.abs(span) > 1e-12, span, 1.0)

    # -- LP assembly ----------------------------------------------------------

    def _pad(
        self,
        block_x: FloatArray | sparse.spmatrix,
        block_aux: FloatArray | sparse.spmatrix,
        aux_offset: int,
        n_aux: int,
        n_rows: int,
    ) -> sparse.csr_matrix:
        """Widen an epigraph's constraint block to the full variable vector."""
        parts: list[sparse.spmatrix] = [sparse.csr_matrix(block_x)]
        left = aux_offset - self._n_assets
        if left:
            parts.append(sparse.csr_matrix((n_rows, left)))
        if n_aux:
            parts.append(sparse.csr_matrix(block_aux))
        right = self._n_variables - (aux_offset + n_aux)
        if right:
            parts.append(sparse.csr_matrix((n_rows, right)))
        return sparse.hstack(parts, format="csr")

    def _portfolio_constraint_rows(
        self,
    ) -> tuple[list[sparse.spmatrix], list[FloatArray], list[sparse.spmatrix], list[FloatArray]]:
        """Rows encoding the feasible set :math:`X` over the ``x`` variables."""
        n = self._n_assets
        pad_width = self._n_variables - n
        ub_blocks: list[sparse.spmatrix] = []
        ub_rhs: list[FloatArray] = []
        eq_blocks: list[sparse.spmatrix] = []
        eq_rhs: list[FloatArray] = []

        def widen(matrix: FloatArray) -> sparse.csr_matrix:
            """Pad an x-only constraint block with zeros for aux and t."""
            return sparse.hstack(
                [sparse.csr_matrix(matrix), sparse.csr_matrix((matrix.shape[0], pad_width))],
                format="csr",
            )

        if self.constraints.budget is not None:
            ub_or_eq = np.ones((1, n))
            eq_blocks.append(widen(ub_or_eq))
            eq_rhs.append(np.array([float(self.constraints.budget)]))

        if self.constraints.min_expected_return is not None:
            assert self._expected_returns is not None  # guaranteed by _resolve
            ub_blocks.append(widen(-self._expected_returns.reshape(1, -1)))
            ub_rhs.append(np.array([-float(self.constraints.min_expected_return)]))

        if self.constraints.linear_inequalities is not None:
            matrix, vector = self.constraints.linear_inequalities
            matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
            ub_blocks.append(widen(matrix))
            ub_rhs.append(np.atleast_1d(np.asarray(vector, dtype=np.float64)))

        if self.constraints.linear_equalities is not None:
            matrix, vector = self.constraints.linear_equalities
            matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
            eq_blocks.append(widen(matrix))
            eq_rhs.append(np.atleast_1d(np.asarray(vector, dtype=np.float64)))

        return ub_blocks, ub_rhs, eq_blocks, eq_rhs

    def _bounds(self) -> list[tuple[float | None, float | None]]:
        """Variable bounds for ``[x | aux | t]``."""
        bounds = self.constraints.bounds_for(self._n_assets)
        for epigraph in self._epigraphs:
            bounds.extend(epigraph.aux_bounds)
        bounds.append((None, None))  # t is free
        return bounds

    def _assemble(
        self, lam: FloatArray, ideal: FloatArray, scale: FloatArray, use_epigraph: bool
    ) -> dict[str, object]:
        """Build the LP data for one weight vector.

        ``use_epigraph=False`` drops the ``max`` rows and the ``t`` cost,
        leaving a pure weighted sum - used internally to solve for one
        objective at a time when building the payoff table.
        """
        cost = np.zeros(self._n_variables)
        if use_epigraph:
            cost[self._t_index] = 1.0

        ub_blocks, ub_rhs, eq_blocks, eq_rhs = self._portfolio_constraint_rows()

        scaled_lambda = lam / scale
        for i, epigraph in enumerate(self._epigraphs):
            offset = self._aux_offsets[i]
            weight = scaled_lambda[i]
            # Augmentation term: rho * lambda_i * f_i, also what keeps the
            # epigraph tight when the max-row for i happens to be slack.
            coefficient = self.rho * weight if use_epigraph else weight
            cost[: self._n_assets] += coefficient * epigraph.c_x
            if epigraph.n_aux:
                cost[offset : offset + epigraph.n_aux] += coefficient * epigraph.c_aux

            if epigraph.b.size:
                ub_blocks.append(
                    self._pad(
                        epigraph.A_x,
                        epigraph.A_aux,
                        offset,
                        epigraph.n_aux,
                        epigraph.b.size,
                    )
                )
                ub_rhs.append(epigraph.b)

            if use_epigraph:
                # lambda_i * (f_i(x) - z*_i) <= t
                row = np.zeros(self._n_variables)
                row[: self._n_assets] = weight * epigraph.c_x
                if epigraph.n_aux:
                    row[offset : offset + epigraph.n_aux] = weight * epigraph.c_aux
                row[self._t_index] = -1.0
                ub_blocks.append(sparse.csr_matrix(row.reshape(1, -1)))
                ub_rhs.append(np.array([weight * ideal[i]]))

        return {
            "c": cost,
            "A_ub": sparse.vstack(ub_blocks, format="csr") if ub_blocks else None,
            "b_ub": np.concatenate(ub_rhs) if ub_rhs else None,
            "A_eq": sparse.vstack(eq_blocks, format="csr") if eq_blocks else None,
            "b_eq": np.concatenate(eq_rhs) if eq_rhs else None,
            "bounds": self._bounds(),
        }

    def _run_linprog(self, problem: dict[str, object]) -> OptimizeResult:
        """Solve an assembled programme with HiGHS."""
        return linprog(
            c=problem["c"],
            A_ub=problem["A_ub"],
            b_ub=problem["b_ub"],
            A_eq=problem["A_eq"],
            b_eq=problem["b_eq"],
            bounds=problem["bounds"],
            method="highs",
            options=self.solver_options,
        )

    def _solve_single_objective(self, index: int) -> FloatArray:
        """Minimise objective ``index`` alone over :math:`X`; return the weights."""
        lam = np.zeros(self.objectives.n_objectives)
        lam[index] = 1.0
        problem = self._assemble(
            lam,
            ideal=np.zeros(self.objectives.n_objectives),
            scale=np.ones(self.objectives.n_objectives),
            use_epigraph=False,
        )
        result = self._run_linprog(problem)
        if not result.success:
            raise RuntimeError(
                f"Single-objective LP for '{self.objectives[index].name}' failed "
                f"(status={result.status}): {result.message}"
            )
        return np.asarray(result.x[: self._n_assets], dtype=np.float64)

    # -- public API -----------------------------------------------------------

    def solve_one(self, lambda_vector: npt.ArrayLike) -> TchebycheffSolution:
        """Solve the scalarised problem for a single weight vector.

        Parameters
        ----------
        lambda_vector
            Non-negative weights, shape ``(m,)``; normalised internally to sum
            to one and floored at :data:`MIN_LAMBDA`.

        Returns
        -------
        TchebycheffSolution
            A globally optimal, Pareto-optimal portfolio.

        Raises
        ------
        RuntimeError
            If the programme is infeasible or unbounded.
        """
        lam = _normalise_lambda(lambda_vector, self.objectives.n_objectives)
        scale = self.objective_scale
        problem = self._assemble(lam, self.ideal, scale, use_epigraph=True)
        result = self._run_linprog(problem)
        if not result.success:
            raise RuntimeError(
                f"Tchebycheff LP failed for lambda={np.round(lam, 4)} "
                f"(status={result.status}): {result.message}"
            )

        weights = np.asarray(result.x[: self._n_assets], dtype=np.float64)
        # Read the objectives back from the portfolio, never from the LP's
        # auxiliary variables: independent of any tightness assumption.
        minimisation = self.objectives.minimisation_batch(weights.reshape(1, -1))[0]

        # linprog carries no constant term, so its optimal value is
        # t + rho * sum_i lambda_i * f_i(x) - the reference-point offset
        # -rho * sum_i lambda_i * z*_i is missing.  Adding it back makes
        # 'scalar_value' the augmented Tchebycheff value itself, which is what
        # makes values comparable across weight vectors.
        offset = -self.rho * float((lam / scale) @ self.ideal)
        return TchebycheffSolution(
            weights=weights,
            objectives=self.objectives.to_natural(minimisation),
            minimisation_objectives=minimisation,
            lambda_vector=lam,
            scalar_value=float(result.fun) + offset,
            success=True,
            message=str(result.message),
        )

    def solve_front(
        self,
        weight_vectors: npt.ArrayLike | None = None,
        n_partitions: int = 12,
        skip_failures: bool = True,
    ) -> ParetoFront:
        """Trace the Pareto front by sweeping a set of weight vectors.

        Parameters
        ----------
        weight_vectors
            Explicit weights of shape ``(R, m)``.  Defaults to a Das-Dennis
            lattice with ``n_partitions`` divisions.
        n_partitions
            Lattice resolution used when ``weight_vectors`` is omitted.
        skip_failures
            Log and skip infeasible sub-problems instead of raising.  A weight
            vector can be infeasible only through the *portfolio* constraints,
            which are shared by every sub-problem, so a failure here almost
            always means the whole feasible set is empty.

        Returns
        -------
        ParetoFront
            Deduplicated and filtered to the non-dominated members.
        """
        lattice = (
            simplex_lattice_weights(self.objectives.n_objectives, n_partitions)
            if weight_vectors is None
            else _as_2d(weight_vectors, "weight_vectors")
        )

        solutions: list[TchebycheffSolution] = []
        for lam in lattice:
            try:
                solutions.append(self.solve_one(lam))
            except RuntimeError as exc:
                if not skip_failures:
                    raise
                LOGGER.warning("Skipping lambda=%s: %s", np.round(lam, 4), exc)

        if not solutions:
            raise RuntimeError(
                "Every Tchebycheff sub-problem failed; the feasible set is "
                "probably empty. Check the budget, box and return floor."
            )

        front = ParetoFront(
            weights=np.vstack([s.weights for s in solutions]),
            objectives=np.vstack([s.objectives for s in solutions]),
            minimisation_objectives=np.vstack(
                [s.minimisation_objectives for s in solutions]
            ),
            names=self.objectives.names,
            method="Tchebycheff-LP",
            metadata={
                "n_weight_vectors": int(lattice.shape[0]),
                "n_solved": len(solutions),
                "rho": self.rho,
                "normalise": self.normalise,
                "ideal": self.ideal.tolist(),
                "nadir": self.nadir.tolist(),
            },
        )
        return front.filter_non_dominated()


class TchebycheffNonlinearSolver:
    """Augmented Tchebycheff scalarisation for objectives without an LP form.

    The general-purpose counterpart to :class:`TchebycheffLPSolver`, used when
    an objective is quadratic (:class:`~src.optimization.objectives.VarianceObjective`)
    or otherwise non-linear.  The same epigraph reformulation is applied - minimise
    ``t`` subject to ``t >= lambda_i * (f_i(x) - z*_i)`` - which converts the
    non-smooth ``max`` into smooth constraints that SLSQP can handle.

    Two honest caveats, which are exactly why the LP route is preferred
    whenever it is available:

    * SLSQP finds a **local** optimum.  Multi-start mitigates but does not
      remove this; on a non-convex problem there is no certificate.
    * Gradients are estimated by finite differences, so each iteration costs
      ``O(n)`` objective evaluations - expensive when the objective is a CVaR
      over tens of thousands of scenarios.

    Parameters
    ----------
    objectives
        Any objectives, LP-representable or not.
    constraints
        Feasible set.  Budget and box go to the solver directly; extra linear
        rows are added as constraint functions.
    expected_returns
        Mean return vector for an expected-return floor.
    rho
        Augmentation coefficient.
    normalise
        Scale objectives by their ideal-to-nadir range, estimated by
        single-objective solves.
    n_restarts
        Random restarts per weight vector; the best local optimum is kept.
    seed
        Seed for the restart sampler.
    """

    def __init__(
        self,
        objectives: ObjectiveSet,
        constraints: PortfolioConstraints | None = None,
        expected_returns: npt.ArrayLike | None = None,
        rho: float = 1e-4,
        normalise: bool = True,
        n_restarts: int = 3,
        seed: int | None = None,
    ) -> None:
        self.objectives = objectives
        self.constraints = (
            constraints if constraints is not None else PortfolioConstraints()
        )
        self.rho = float(rho)
        self.normalise = bool(normalise)
        self.n_restarts = max(1, int(n_restarts))
        self.seed = seed
        self._expected_returns = _resolve_expected_returns(
            objectives, self.constraints, expected_returns
        )
        self._n_assets = objectives.n_assets
        self._lower, self._upper = _finite_bounds(
            self.constraints, self._n_assets, reference_scale=self.constraints.budget or 1.0
        )
        self._ideal: FloatArray | None = None
        self._scale: FloatArray | None = None

    def _evaluate(self, x: FloatArray) -> FloatArray:
        """Minimisation-sense objective vector at a single point."""
        return self.objectives.minimisation_batch(np.asarray(x).reshape(1, -1))[0]

    def _scipy_constraints(self) -> list[dict[str, object]]:
        """Assemble the portfolio constraints in SLSQP's dictionary form."""
        entries: list[dict[str, object]] = []
        if self.constraints.budget is not None:
            budget = float(self.constraints.budget)
            entries.append(
                {"type": "eq", "fun": lambda z: float(z[:-1].sum() - budget)}
            )
        if self.constraints.min_expected_return is not None:
            mean = self._expected_returns
            floor = float(self.constraints.min_expected_return)
            entries.append(
                {"type": "ineq", "fun": lambda z, m=mean, r=floor: float(z[:-1] @ m - r)}
            )
        if self.constraints.linear_inequalities is not None:
            matrix, vector = self.constraints.linear_inequalities
            matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
            vector = np.atleast_1d(np.asarray(vector, dtype=np.float64))
            entries.append(
                {"type": "ineq", "fun": lambda z, a=matrix, b=vector: b - a @ z[:-1]}
            )
        if self.constraints.linear_equalities is not None:
            matrix, vector = self.constraints.linear_equalities
            matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
            vector = np.atleast_1d(np.asarray(vector, dtype=np.float64))
            entries.append(
                {"type": "eq", "fun": lambda z, a=matrix, b=vector: a @ z[:-1] - b}
            )
        return entries

    def _reference_points(self) -> tuple[FloatArray, FloatArray]:
        """Estimate the ideal point and objective scales by single-objective runs."""
        if self._ideal is not None and self._scale is not None:
            return self._ideal, self._scale

        n_objectives = self.objectives.n_objectives
        table = np.empty((n_objectives, n_objectives))
        base_constraints = self._scipy_constraints()
        bounds = list(zip(self._lower, self._upper))
        rng = np.random.default_rng(self.seed)

        for i in range(n_objectives):
            best_value, best_x = np.inf, None
            for _ in range(self.n_restarts):
                start = project_onto_budget_box(
                    rng.dirichlet(np.ones(self._n_assets)) * (self.constraints.budget or 1.0),
                    self._lower,
                    self._upper,
                    self.constraints.budget,
                )
                result = minimize(
                    lambda x, k=i: float(self._evaluate(x)[k]),
                    start,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=[
                        {**entry, "fun": _strip_epigraph(entry["fun"])}
                        for entry in base_constraints
                    ],
                    options={"maxiter": 200, "ftol": 1e-10},
                )
                if result.fun < best_value:
                    best_value, best_x = float(result.fun), np.asarray(result.x)
            table[i] = self._evaluate(best_x)

        ideal = np.diag(table).copy()
        nadir = table.max(axis=0)
        span = nadir - ideal
        scale = np.where(np.abs(span) > 1e-12, span, 1.0) if self.normalise else np.ones(n_objectives)
        self._ideal, self._scale = ideal, scale
        return ideal, scale

    def solve_one(self, lambda_vector: npt.ArrayLike) -> TchebycheffSolution:
        """Solve the scalarised problem for one weight vector by SLSQP.

        Parameters
        ----------
        lambda_vector
            Non-negative weights, shape ``(m,)``.

        Returns
        -------
        TchebycheffSolution
            The best local optimum found across restarts.  ``success`` reflects
            SLSQP's own convergence flag, which callers should check: unlike
            the LP route there is no global guarantee.
        """
        lam = _normalise_lambda(lambda_vector, self.objectives.n_objectives)
        ideal, scale = self._reference_points()
        weighted = lam / scale
        rng = np.random.default_rng(self.seed)

        # Decision vector is [x | t]; t is the epigraph variable for the max.
        bounds = list(zip(self._lower, self._upper)) + [(None, None)]
        constraints = self._scipy_constraints()
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda z: z[-1] - weighted * (self._evaluate(z[:-1]) - ideal),
            }
        )

        def scalarised(z: FloatArray) -> float:
            """Augmented Tchebycheff value at ``z = [x | t]``."""
            return float(z[-1] + self.rho * (weighted * (self._evaluate(z[:-1]) - ideal)).sum())

        best: OptimizeResult | None = None
        for restart in range(self.n_restarts):
            start_x = project_onto_budget_box(
                rng.dirichlet(np.ones(self._n_assets)) * (self.constraints.budget or 1.0),
                self._lower,
                self._upper,
                self.constraints.budget,
            )
            start_t = float((weighted * (self._evaluate(start_x) - ideal)).max())
            result = minimize(
                scalarised,
                np.append(start_x, start_t),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 300, "ftol": 1e-10},
            )
            if best is None or result.fun < best.fun:
                best = result

        assert best is not None
        weights = project_onto_budget_box(
            np.asarray(best.x[:-1]), self._lower, self._upper, self.constraints.budget
        )
        minimisation = self._evaluate(weights)
        return TchebycheffSolution(
            weights=weights,
            objectives=self.objectives.to_natural(minimisation),
            minimisation_objectives=minimisation,
            lambda_vector=lam,
            scalar_value=float(best.fun),
            success=bool(best.success),
            message=str(best.message),
        )

    def solve_front(
        self,
        weight_vectors: npt.ArrayLike | None = None,
        n_partitions: int = 8,
    ) -> ParetoFront:
        """Trace the front by sweeping weight vectors; see :meth:`solve_one`."""
        lattice = (
            simplex_lattice_weights(self.objectives.n_objectives, n_partitions)
            if weight_vectors is None
            else _as_2d(weight_vectors, "weight_vectors")
        )
        solutions = [self.solve_one(lam) for lam in lattice]
        front = ParetoFront(
            weights=np.vstack([s.weights for s in solutions]),
            objectives=np.vstack([s.objectives for s in solutions]),
            minimisation_objectives=np.vstack(
                [s.minimisation_objectives for s in solutions]
            ),
            names=self.objectives.names,
            method="Tchebycheff-SLSQP",
            metadata={
                "n_weight_vectors": int(lattice.shape[0]),
                "n_converged": sum(s.success for s in solutions),
                "rho": self.rho,
                "n_restarts": self.n_restarts,
            },
        )
        return front.filter_non_dominated()


def _strip_epigraph(function):
    """Adapt a ``[x | t]`` constraint function back to a plain ``x`` signature."""

    def wrapped(x):
        return function(np.append(np.asarray(x), 0.0))

    return wrapped


# ==============================================================================
# 8. Comparative analysis
# ==============================================================================


def compare_fronts(
    candidate: ParetoFront,
    reference: ParetoFront,
    reference_point: npt.ArrayLike | None = None,
) -> dict[str, float]:
    """Score an approximate front against a reference front.

    Every indicator is computed on **normalised** objectives - scaled by the
    range of the union of both fronts - because the raw objectives here differ
    by two orders of magnitude and an unnormalised Euclidean distance would be
    a report on the largest-range objective alone.

    Hypervolume for both fronts is measured against one shared reference point,
    derived from the union unless supplied; comparing hypervolumes taken
    against different reference points is meaningless.

    Parameters
    ----------
    candidate
        The approximate front, typically from NSGA-II.
    reference
        The true or best-known front, typically from the Tchebycheff LP.
    reference_point
        Shared hypervolume reference in minimisation units.  Defaults to the
        nadir of the union plus a 10% margin.

    Returns
    -------
    dict
        ``igd`` and ``gd`` (lower is better), ``hypervolume_candidate`` /
        ``hypervolume_reference`` and their ``hypervolume_ratio`` (higher is
        better; a ratio near 1 means the candidate captured essentially all the
        dominated volume), ``coverage_candidate_over_reference`` and its
        converse, ``spacing_*``, and the two front sizes.

    Raises
    ------
    ValueError
        If the fronts have different numbers of objectives, or either is empty.
    """
    if len(candidate) == 0 or len(reference) == 0:
        raise ValueError("Both fronts must be non-empty to be compared.")
    a = candidate.minimisation_objectives
    b = reference.minimisation_objectives
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Objective count mismatch: candidate has {a.shape[1]}, "
            f"reference has {b.shape[1]}."
        )

    union = np.vstack([a, b])
    span = union.max(axis=0) - union.min(axis=0)
    scale = np.where(span > 0.0, span, 1.0)

    shared_reference = (
        default_reference_point(union)
        if reference_point is None
        else _as_1d(reference_point, "reference_point")
    )
    hv_candidate = hypervolume(a, shared_reference)
    hv_reference = hypervolume(b, shared_reference)

    return {
        "igd": inverted_generational_distance(a, b, scale),
        "gd": generational_distance(a, b, scale),
        "hypervolume_candidate": hv_candidate,
        "hypervolume_reference": hv_reference,
        "hypervolume_ratio": hv_candidate / hv_reference if hv_reference > 0 else np.nan,
        "coverage_candidate_over_reference": coverage(a, b),
        "coverage_reference_over_candidate": coverage(b, a),
        "spacing_candidate": spacing(a, scale),
        "spacing_reference": spacing(b, scale),
        "n_candidate": float(len(candidate)),
        "n_reference": float(len(reference)),
    }
