"""Tests for the multi-objective engine: objectives, NSGA-II and Tchebycheff.

The suite is layered:

1.  **Simplex feasibility** - the projection operator that keeps every genetic
    individual on the budget simplex.
2.  **Pareto machinery** - dominance, sorting, crowding, and the quality
    indicators, each checked against hand-computed values.
3.  **Objectives** - agreement with :mod:`src.models.risk_metrics`, and
    tightness of every LP epigraph.
4.  **Genetic operators** - the invariants each one is supposed to preserve.
5.  **Cross-validation** - the decisive layer.  The Tchebycheff LP produces
    *globally* optimal points, so NSGA-II must never dominate one; and on a
    two-objective sub-problem the exact front must coincide with the
    :func:`~src.models.risk_metrics.cvar_efficient_frontier` computed
    independently in Step 1.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from src.data.paper_datasets import COVARIANCE, MEAN_RETURNS, sample_normal_returns
from src.models.risk_metrics import (
    CVaRLinearProgram,
    PortfolioConstraints,
    historical_var_cvar,
    historical_var_cvar_batch,
)
from src.optimization.multiobjective import (
    NSGAII,
    NSGAIIConfig,
    ParetoFront,
    PolynomialMutation,
    SimplexBlendCrossover,
    SimulatedBinaryCrossover,
    TchebycheffLPSolver,
    TchebycheffNonlinearSolver,
    WeightTransferMutation,
    compare_fronts,
    constrained_non_dominated_sort,
    constraint_violation,
    coverage,
    crowding_distance,
    default_reference_point,
    fast_non_dominated_sort,
    generational_distance,
    hypervolume,
    inverted_generational_distance,
    non_dominated_mask,
    project_onto_budget_box,
    simplex_lattice_weights,
    spacing,
)
from src.optimization.objectives import (
    CVaRObjective,
    ExpectedReturnObjective,
    HedgingCostModel,
    HedgingCostObjective,
    ObjectiveSense,
    ObjectiveSet,
    VarianceObjective,
)

BETA = 0.95
N_SCENARIOS = 2_048


@pytest.fixture(scope="module")
def returns() -> np.ndarray:
    """A modest Sobol scenario set - large enough to be realistic, fast to solve."""
    return sample_normal_returns(N_SCENARIOS, engine="sobol", seed=7)


@pytest.fixture(scope="module")
def hedging_model() -> HedgingCostModel:
    """A book currently 100% in the S&P sleeve.

    Small caps carry the widest spread and the only carry charge, standing in
    for a protective option overlay: it is the cheapest way to cut CVaR and the
    most expensive to hold.
    """
    return HedgingCostModel(
        n_assets=3,
        baseline_weights=np.array([1.0, 0.0, 0.0]),
        spreads=np.array([0.0010, 0.0005, 0.0040]),
        carry=np.array([0.0, 0.0, 0.0020]),
    )


@pytest.fixture(scope="module")
def objectives(returns: np.ndarray, hedging_model: HedgingCostModel) -> ObjectiveSet:
    """The canonical conflicting triple: return, tail risk, hedging cost."""
    return ObjectiveSet(
        [
            ExpectedReturnObjective(MEAN_RETURNS),
            CVaRObjective.from_returns(returns, BETA),
            HedgingCostObjective(hedging_model),
        ]
    )


@pytest.fixture(scope="module")
def constraints() -> PortfolioConstraints:
    """Long-only, fully invested."""
    return PortfolioConstraints(budget=1.0, lower_bounds=0.0, upper_bounds=1.0)


@pytest.fixture(scope="module")
def reference_front(
    objectives: ObjectiveSet, constraints: PortfolioConstraints
) -> ParetoFront:
    """The exact Pareto front from the Tchebycheff linear programme."""
    return TchebycheffLPSolver(objectives, constraints).solve_front(n_partitions=12)


# ==============================================================================
# 1. Projection onto the budget simplex
# ==============================================================================


def test_projection_lands_on_the_simplex() -> None:
    """Projected points satisfy the budget exactly and lie inside the box."""
    rng = np.random.default_rng(0)
    points = rng.normal(scale=3.0, size=(50, 6))
    lower, upper = np.zeros(6), np.ones(6)

    projected = project_onto_budget_box(points, lower, upper, budget=1.0)
    assert projected.sum(axis=1) == pytest.approx(1.0, abs=1e-10)
    assert np.all(projected >= -1e-12)
    assert np.all(projected <= 1.0 + 1e-12)


def test_projection_is_idempotent() -> None:
    """A point already feasible is returned unchanged - the projection is exact."""
    rng = np.random.default_rng(1)
    feasible = rng.dirichlet(np.ones(5), size=20)
    lower, upper = np.zeros(5), np.ones(5)

    projected = project_onto_budget_box(feasible, lower, upper, budget=1.0)
    assert projected == pytest.approx(feasible, abs=1e-10)


def test_projection_is_the_nearest_feasible_point() -> None:
    """Verify Euclidean optimality against a general-purpose QP solve.

    The bisection derivation is only worth having if it really produces the
    *nearest* point, so it is checked against SLSQP rather than assumed.
    """
    rng = np.random.default_rng(2)
    lower, upper = np.zeros(4), np.full(4, 0.6)

    for _ in range(5):
        point = rng.normal(scale=2.0, size=4)
        projected = project_onto_budget_box(point, lower, upper, budget=1.0)
        reference = minimize(
            lambda x, p=point: float(((x - p) ** 2).sum()),
            np.full(4, 0.25),
            method="SLSQP",
            bounds=list(zip(lower, upper)),
            constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0}],
            options={"ftol": 1e-14, "maxiter": 500},
        )
        assert projected == pytest.approx(reference.x, abs=1e-6)


def test_projection_supports_short_positions() -> None:
    """Negative lower bounds (a long-short book) project correctly."""
    rng = np.random.default_rng(3)
    points = rng.normal(size=(20, 4))
    lower, upper = np.full(4, -0.5), np.full(4, 1.5)

    projected = project_onto_budget_box(points, lower, upper, budget=1.0)
    assert projected.sum(axis=1) == pytest.approx(1.0, abs=1e-10)
    assert np.all(projected >= -0.5 - 1e-12)


def test_projection_rejects_an_empty_feasible_set() -> None:
    """Bounds that cannot reach the budget make the problem infeasible."""
    with pytest.raises(ValueError, match="Empty feasible set"):
        project_onto_budget_box(np.zeros((2, 3)), np.zeros(3), np.full(3, 0.2), budget=1.0)


def test_projection_without_a_budget_only_clips() -> None:
    """``budget=None`` degenerates to a box projection."""
    points = np.array([[-1.0, 0.5, 2.0]])
    projected = project_onto_budget_box(points, np.zeros(3), np.ones(3), budget=None)
    assert projected == pytest.approx(np.array([[0.0, 0.5, 1.0]]))


# ==============================================================================
# 2. Pareto machinery and quality indicators
# ==============================================================================


def test_domination_and_fronts_on_a_hand_worked_example() -> None:
    """Minimisation: (1,5), (2,3) and (3,1) are mutually non-dominating."""
    values = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0], [5.0, 5.0]])
    assert non_dominated_mask(values).tolist() == [True, True, True, False, False]

    fronts = fast_non_dominated_sort(values)
    assert [front.tolist() for front in fronts] == [[0, 1, 2], [3], [4]]


def test_every_front_is_internally_non_dominated() -> None:
    """A structural invariant of the sort, checked on random data."""
    rng = np.random.default_rng(4)
    values = rng.normal(size=(60, 3))
    for front in fast_non_dominated_sort(values):
        assert non_dominated_mask(values[front]).all()


def test_crowding_distance_marks_the_extremes_as_infinite() -> None:
    """Boundary solutions must always survive truncation, so they score ``inf``."""
    values = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
    distances = crowding_distance(values)
    assert np.isinf(distances[0]) and np.isinf(distances[-1])
    assert np.all(np.isfinite(distances[1:-1]))


def test_constrained_domination_puts_feasibility_first() -> None:
    """An infeasible solution loses to a feasible one however good its objectives."""
    values = np.array([[0.0, 0.0], [5.0, 5.0], [9.0, 9.0]])
    violation = np.array([1.0, 0.0, 2.0])  # only index 1 is feasible

    fronts = constrained_non_dominated_sort(values, violation)
    assert fronts[0].tolist() == [1]  # feasible, despite mediocre objectives
    assert fronts[1].tolist() == [0]  # less infeasible than index 2
    assert fronts[2].tolist() == [2]


def test_constraint_violation_aggregates_the_return_floor() -> None:
    """A portfolio short of the expected-return floor reports the shortfall."""
    weights = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    limits = PortfolioConstraints(min_expected_return=0.010)
    violation = constraint_violation(weights, limits, MEAN_RETURNS)
    assert violation[0] == pytest.approx(0.010 - MEAN_RETURNS[1])
    assert violation[1] == pytest.approx(0.0)  # small cap clears the floor


@pytest.mark.parametrize(
    "points, reference, expected",
    [
        # One point: a plain box.
        ([[1.0, 1.0]], [3.0, 3.0], 4.0),
        # Two points, overlap removed by inclusion-exclusion: 3 + 6 - 2.
        ([[1.0, 3.0], [2.0, 1.0]], [4.0, 4.0], 7.0),
        # Three dimensions, single point.
        ([[1.0, 1.0, 1.0]], [3.0, 3.0, 3.0], 8.0),
        # Three dimensions, two points: 2*2*1 + 1*2.5*2 - 1*2*1.
        ([[1.0, 1.0, 2.0], [2.0, 0.5, 1.0]], [3.0, 3.0, 3.0], 7.0),
    ],
)
def test_hypervolume_matches_hand_computation(points, reference, expected) -> None:
    """Exact hypervolume in 2-D and 3-D against inclusion-exclusion by hand."""
    assert hypervolume(np.array(points), np.array(reference)) == pytest.approx(expected)


def test_hypervolume_is_monotonic_under_dominance() -> None:
    """Adding a dominating point can only increase hypervolume.

    This is the property that makes hypervolume the indicator of choice: it
    cannot reward a front that has genuinely got worse.
    """
    reference = np.array([5.0, 5.0])
    base = np.array([[2.0, 3.0], [3.0, 2.0]])
    improved = np.vstack([base, [1.5, 1.5]])  # dominates both
    assert hypervolume(improved, reference) > hypervolume(base, reference)


def test_hypervolume_ignores_points_worse_than_the_reference() -> None:
    """Solutions outside the reference box contribute nothing."""
    reference = np.array([2.0, 2.0])
    assert hypervolume(np.array([[3.0, 3.0]]), reference) == 0.0


def test_igd_and_gd_vanish_for_identical_fronts() -> None:
    """A perfect approximation scores zero on both distance indicators."""
    front = np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]])
    assert inverted_generational_distance(front, front) == pytest.approx(0.0)
    assert generational_distance(front, front) == pytest.approx(0.0)


def test_igd_penalises_a_missing_region() -> None:
    """Covering only half the true front costs IGD, though GD stays small."""
    reference = np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]])
    partial = np.array([[1.0, 4.0]])  # exactly on the front, but only one point
    assert generational_distance(partial, reference) == pytest.approx(0.0)
    assert inverted_generational_distance(partial, reference) > 0.5


def test_coverage_is_asymmetric_and_bounded() -> None:
    """A dominating front covers its rival completely; the converse is zero."""
    better = np.array([[1.0, 1.0]])
    worse = np.array([[2.0, 2.0], [3.0, 3.0]])
    assert coverage(better, worse) == pytest.approx(1.0)
    assert coverage(worse, better) == pytest.approx(0.0)
    assert coverage(better, better) == pytest.approx(0.0)  # nothing dominates itself


def test_spacing_is_zero_for_a_uniform_front() -> None:
    """Evenly spaced solutions have identical nearest-neighbour gaps."""
    front = np.array([[float(i), 10.0 - i] for i in range(6)])
    assert spacing(front) == pytest.approx(0.0, abs=1e-12)


def test_simplex_lattice_has_the_binomial_cardinality() -> None:
    """Das-Dennis produces C(H+m-1, m-1) weight vectors, each summing to one."""
    from math import comb

    for m, h in [(2, 10), (3, 8), (4, 4)]:
        lattice = simplex_lattice_weights(m, h)
        assert lattice.shape == (comb(h + m - 1, m - 1), m)
        assert lattice.sum(axis=1) == pytest.approx(1.0)
        assert np.all(lattice >= 0.0)


# ==============================================================================
# 3. Objectives and epigraph tightness
# ==============================================================================


def test_expected_return_objective_is_the_inner_product() -> None:
    """Natural units are returns; minimisation units are their negation."""
    objective = ExpectedReturnObjective(MEAN_RETURNS)
    weights = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert objective.evaluate_batch(weights) == pytest.approx(
        [MEAN_RETURNS[0], MEAN_RETURNS[2]]
    )
    assert objective.sense is ObjectiveSense.MAXIMISE
    assert objective.minimisation_batch(weights) == pytest.approx(
        [-MEAN_RETURNS[0], -MEAN_RETURNS[2]]
    )


def test_cvar_objective_matches_the_risk_metrics_estimator(returns: np.ndarray) -> None:
    """The GA and the LP must see numerically identical CVaR, not merely similar."""
    objective = CVaRObjective.from_returns(returns, BETA)
    population = np.random.default_rng(5).dirichlet(np.ones(3), size=15)

    batch = objective.evaluate_batch(population)
    for p, weights in enumerate(population):
        expected = historical_var_cvar(-(returns @ weights), BETA).cvar
        assert batch[p] == pytest.approx(expected, rel=1e-12)


def test_batch_kernel_matches_the_scalar_estimator(returns: np.ndarray) -> None:
    """The vectorised kernel is an optimisation, never a different formula."""
    population = np.random.default_rng(6).dirichlet(np.ones(3), size=20)
    loss_matrix = (-returns) @ population.T

    var, cvar = historical_var_cvar_batch(loss_matrix, BETA)
    for p in range(population.shape[0]):
        scalar = historical_var_cvar(loss_matrix[:, p], BETA)
        assert var[p] == pytest.approx(scalar.var, rel=1e-12)
        assert cvar[p] == pytest.approx(scalar.cvar, rel=1e-12)


def test_hedging_cost_charges_turnover_and_carry(hedging_model: HedgingCostModel) -> None:
    """Cost is spread on the distance from the book, plus carry on gross exposure."""
    # Staying put costs nothing: no turnover, and the S&P sleeve has no carry.
    assert hedging_model.cost(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)

    # Rotating fully into small caps: turnover on both legs, plus small-cap carry.
    weights = np.array([0.0, 0.0, 1.0])
    expected = 0.0010 * 1.0 + 0.0040 * 1.0 + 0.0020 * 1.0
    assert hedging_model.cost(weights) == pytest.approx(expected)


def test_hedging_cost_rejects_negative_costs() -> None:
    """A negative spread would be a rebate and would break the LP linearisation."""
    with pytest.raises(ValueError, match="never rebates"):
        HedgingCostModel(n_assets=3, spreads=np.array([0.001, -0.002, 0.001]))


def test_variance_objective_is_not_lp_representable() -> None:
    """A quadratic objective must decline the LP route rather than fake one."""
    objective = VarianceObjective(COVARIANCE)
    assert not objective.supports_lp
    with pytest.raises(NotImplementedError, match="no linear-programming"):
        objective.lp_epigraph()


def test_variance_objective_evaluates_the_quadratic_form() -> None:
    """Row-wise ``x'Vx``, computed without materialising the full product."""
    objective = VarianceObjective(COVARIANCE)
    population = np.random.default_rng(8).dirichlet(np.ones(3), size=10)
    expected = np.array([w @ COVARIANCE @ w for w in population])
    assert objective.evaluate_batch(population) == pytest.approx(expected)


def test_cvar_epigraph_reproduces_the_step_one_linear_programme(
    returns: np.ndarray, constraints: PortfolioConstraints
) -> None:
    """The epigraph is the same reduction Step 1 solves, so the optima must agree.

    Minimising the CVaR objective alone through the Tchebycheff machinery must
    land on exactly the value :class:`CVaRLinearProgram` reports.
    """
    objective_set = ObjectiveSet(
        [ExpectedReturnObjective(MEAN_RETURNS), CVaRObjective.from_returns(returns, BETA)]
    )
    solver = TchebycheffLPSolver(objective_set, constraints)
    payoff = solver.payoff_table()
    # Row 1 is the portfolio that minimises CVaR alone; column 1 is its CVaR.
    cvar_from_scalarisation = payoff[1, 1]

    step_one = CVaRLinearProgram.from_returns(returns, BETA, constraints).solve()
    assert cvar_from_scalarisation == pytest.approx(step_one.cvar, rel=1e-8)


def test_objective_set_validates_its_members(returns: np.ndarray) -> None:
    """Dimension mismatches, duplicate names and degenerate sets are rejected."""
    cvar = CVaRObjective.from_returns(returns, BETA)
    with pytest.raises(ValueError, match="at least 2 objectives"):
        ObjectiveSet([cvar])
    with pytest.raises(ValueError, match="disagree on the asset dimension"):
        ObjectiveSet([cvar, ExpectedReturnObjective(np.ones(5))])
    with pytest.raises(ValueError, match="names must be unique"):
        ObjectiveSet(
            [
                ExpectedReturnObjective(MEAN_RETURNS, name="same"),
                CVaRObjective.from_returns(returns, BETA, name="same"),
            ]
        )


# ==============================================================================
# 4. Genetic operators and their invariants
# ==============================================================================


def test_sbx_conserves_the_pair_sum_before_clipping() -> None:
    """SBX itself satisfies ``c1 + c2 = p1 + p2`` gene-wise.

    Bounds are set wide so the operator's own clip never fires; with a real
    box the identity is broken by clipping, which is precisely why the driver
    projects afterwards rather than relying on it.
    """
    rng = np.random.default_rng(9)
    parents_a = rng.dirichlet(np.ones(5), size=30)
    parents_b = rng.dirichlet(np.ones(5), size=30)
    lower, upper = np.full(5, -50.0), np.full(5, 50.0)

    child_a, child_b = SimulatedBinaryCrossover(probability=1.0).recombine(
        parents_a, parents_b, lower, upper, rng
    )
    assert child_a + child_b == pytest.approx(parents_a + parents_b, abs=1e-10)


def test_sbx_children_are_repaired_onto_the_simplex() -> None:
    """Under a real box, feasibility comes from the projection, not from SBX."""
    rng = np.random.default_rng(19)
    parents_a = rng.dirichlet(np.ones(5), size=30)
    parents_b = rng.dirichlet(np.ones(5), size=30)
    lower, upper = np.zeros(5), np.ones(5)

    child_a, _ = SimulatedBinaryCrossover(probability=1.0).recombine(
        parents_a, parents_b, lower, upper, rng
    )
    repaired = project_onto_budget_box(child_a, lower, upper, budget=1.0)
    assert repaired.sum(axis=1) == pytest.approx(1.0, abs=1e-10)
    assert np.all(repaired >= -1e-12)


def test_simplex_blend_conserves_each_child_budget() -> None:
    """The affine blend keeps every individual child on the budget hyperplane.

    Bounds are set wide so that clipping never fires and the invariant can be
    observed on its own.
    """
    rng = np.random.default_rng(10)
    parents_a = rng.dirichlet(np.ones(4), size=25)
    parents_b = rng.dirichlet(np.ones(4), size=25)
    lower, upper = np.full(4, -10.0), np.full(4, 10.0)

    child_a, child_b = SimplexBlendCrossover(probability=1.0, alpha=0.5).recombine(
        parents_a, parents_b, lower, upper, rng
    )
    assert child_a.sum(axis=1) == pytest.approx(1.0, abs=1e-12)
    assert child_b.sum(axis=1) == pytest.approx(1.0, abs=1e-12)


def test_weight_transfer_mutation_conserves_budget_and_box_exactly() -> None:
    """The tailored mutation needs no repair: both invariants hold identically."""
    rng = np.random.default_rng(11)
    population = rng.dirichlet(np.ones(6), size=40)
    lower, upper = np.zeros(6), np.full(6, 0.5)
    population = project_onto_budget_box(population, lower, upper, 1.0)

    mutated = WeightTransferMutation(probability=1.0, n_trades=3).mutate(
        population, lower, upper, rng
    )
    assert mutated.sum(axis=1) == pytest.approx(1.0, abs=1e-12)
    assert np.all(mutated >= lower - 1e-12)
    assert np.all(mutated <= upper + 1e-12)
    assert not np.allclose(mutated, population)  # it did something


def test_polynomial_mutation_stays_inside_the_box() -> None:
    """Polynomial mutation respects bounds by construction, before any clipping."""
    rng = np.random.default_rng(12)
    population = rng.dirichlet(np.ones(5), size=50)
    lower, upper = np.zeros(5), np.ones(5)

    mutated = PolynomialMutation(probability=1.0).mutate(population, lower, upper, rng)
    assert np.all(mutated >= lower - 1e-12)
    assert np.all(mutated <= upper + 1e-12)


def test_operators_are_reproducible_from_a_seed() -> None:
    """Identical generators give identical offspring - runs must be auditable."""
    parents_a = np.random.default_rng(13).dirichlet(np.ones(4), size=10)
    parents_b = np.random.default_rng(14).dirichlet(np.ones(4), size=10)
    lower, upper = np.zeros(4), np.ones(4)
    operator = SimulatedBinaryCrossover()

    first = operator.recombine(parents_a, parents_b, lower, upper, np.random.default_rng(99))
    second = operator.recombine(parents_a, parents_b, lower, upper, np.random.default_rng(99))
    assert first[0] == pytest.approx(second[0])


# ==============================================================================
# 5. Tchebycheff scalarisation
# ==============================================================================


def test_payoff_table_diagonal_is_the_ideal_point(
    objectives: ObjectiveSet, constraints: PortfolioConstraints
) -> None:
    """Each diagonal entry is that objective's unconstrained best, hence minimal."""
    solver = TchebycheffLPSolver(objectives, constraints)
    table = solver.payoff_table()
    assert np.diag(table) == pytest.approx(solver.ideal)
    # No off-diagonal entry can beat the diagonal: the diagonal is the optimum.
    for column in range(table.shape[1]):
        assert table[:, column].min() == pytest.approx(solver.ideal[column], abs=1e-9)
    assert np.all(solver.nadir >= solver.ideal - 1e-12)


def test_the_three_objectives_genuinely_conflict(
    objectives: ObjectiveSet, constraints: PortfolioConstraints
) -> None:
    """If any two agreed, the Pareto front would collapse and the exercise is moot.

    The payoff table makes the conflict explicit: the portfolio that is best
    for one objective is strictly worse for the others.
    """
    solver = TchebycheffLPSolver(objectives, constraints)
    table = solver.payoff_table()
    for i in range(3):
        for j in range(3):
            if i != j:
                assert table[i, j] > solver.ideal[j] + 1e-9


def test_tchebycheff_epigraphs_are_tight(
    objectives: ObjectiveSet, constraints: PortfolioConstraints
) -> None:
    """The LP's optimal value must equal the scalarisation recomputed from ``x``.

    If any epigraph went slack, the linear programme's objective would exceed
    the value implied by directly evaluating the returned portfolio.  This is
    the assertion that guards the ``rho > 0`` and ``MIN_LAMBDA`` design.
    """
    solver = TchebycheffLPSolver(objectives, constraints)
    scale = solver.objective_scale
    ideal = solver.ideal

    for lam in simplex_lattice_weights(3, 4):
        solution = solver.solve_one(lam)
        weighted = solution.lambda_vector / scale
        deviations = weighted * (solution.minimisation_objectives - ideal)
        recomputed = deviations.max() + solver.rho * deviations.sum()
        assert solution.scalar_value == pytest.approx(recomputed, abs=1e-9)


def test_tchebycheff_front_is_feasible_and_non_dominated(
    reference_front: ParetoFront,
) -> None:
    """Every point is fully invested, long-only, and mutually non-dominating."""
    assert len(reference_front) > 10
    assert reference_front.weights.sum(axis=1) == pytest.approx(1.0, abs=1e-8)
    assert np.all(reference_front.weights >= -1e-9)
    assert non_dominated_mask(reference_front.minimisation_objectives).all()


def test_tchebycheff_reproduces_the_step_one_efficient_frontier(
    returns: np.ndarray, constraints: PortfolioConstraints
) -> None:
    """Cross-module agreement with the convex frontier computed in Step 1.

    On the two-objective sub-problem the Tchebycheff front and the
    return-constrained CVaR frontier are the same mathematical object reached
    by two different routes - a weight sweep and a target sweep.  Every
    Tchebycheff point must therefore attain the CVaR that Step 1's linear
    programme achieves at that same expected return.

    Both routes are given the *same* expected-return vector.  Leaving the
    frontier on its scenario-sample-mean default while the objective uses the
    population mean would compare two subtly different constraints and produce
    discrepancies that look like solver noise but are not.
    """
    from src.models.risk_metrics import cvar_efficient_frontier

    objective_set = ObjectiveSet(
        [ExpectedReturnObjective(MEAN_RETURNS), CVaRObjective.from_returns(returns, BETA)]
    )
    front = TchebycheffLPSolver(objective_set, constraints).solve_front(n_partitions=25)

    targets = front.objectives[:, 0]  # the expected returns actually attained
    step_one = cvar_efficient_frontier(
        returns,
        BETA,
        targets,
        PortfolioConstraints(budget=1.0, lower_bounds=0.0, upper_bounds=1.0),
        expected_returns=MEAN_RETURNS,
    )
    assert len(step_one) == len(targets)
    # Tolerance is set by HiGHS's default primal feasibility tolerance (~1e-7),
    # not by the mathematics: both routes solve the same LP to that accuracy.
    solver_tolerance = 1e-6
    for point, tchebycheff_cvar in zip(step_one, front.objectives[:, 1]):
        assert point.cvar == pytest.approx(tchebycheff_cvar, abs=solver_tolerance)
        # Directional check: Step 1 minimises CVaR at exactly this return, while
        # the Tchebycheff point is nudged off that optimum by the rho
        # augmentation, so Step 1 can never come out worse.
        assert point.cvar <= tchebycheff_cvar + solver_tolerance


def test_tchebycheff_honours_extra_constraints(
    objectives: ObjectiveSet, returns: np.ndarray
) -> None:
    """A return floor and a sector cap both bind through the LP."""
    limited = PortfolioConstraints(
        budget=1.0,
        lower_bounds=0.0,
        upper_bounds=1.0,
        min_expected_return=0.011,
        linear_inequalities=(np.array([[0.0, 0.0, 1.0]]), np.array([0.6])),
    )
    front = TchebycheffLPSolver(objectives, limited).solve_front(n_partitions=6)
    assert front.objectives[:, 0].min() >= 0.011 - 1e-9  # return floor
    assert front.weights[:, 2].max() <= 0.6 + 1e-9  # small-cap cap


def test_tchebycheff_lp_rejects_a_non_linear_objective(returns: np.ndarray) -> None:
    """A quadratic objective must be refused with a pointer to the alternatives."""
    mixed = ObjectiveSet(
        [ExpectedReturnObjective(MEAN_RETURNS), VarianceObjective(COVARIANCE)]
    )
    with pytest.raises(ValueError, match="LP-representable"):
        TchebycheffLPSolver(mixed)


def test_tchebycheff_requires_positive_augmentation(objectives: ObjectiveSet) -> None:
    """``rho = 0`` breaks epigraph tightness and is rejected up front."""
    with pytest.raises(ValueError, match="strictly positive"):
        TchebycheffLPSolver(objectives, rho=0.0)


def test_tchebycheff_weights_are_validated(
    objectives: ObjectiveSet, constraints: PortfolioConstraints
) -> None:
    """Negative or all-zero weight vectors are rejected."""
    solver = TchebycheffLPSolver(objectives, constraints)
    with pytest.raises(ValueError, match="non-negative"):
        solver.solve_one(np.array([-0.5, 0.75, 0.75]))
    with pytest.raises(ValueError, match="not be all zero"):
        solver.solve_one(np.zeros(3))


def test_nonlinear_solver_handles_a_quadratic_objective(returns: np.ndarray) -> None:
    """The SLSQP route covers objectives the LP cannot express.

    Variance is deliberately used here: it is the objective that closes the LP
    route, so it exercises the fallback for real.
    """
    mixed = ObjectiveSet(
        [ExpectedReturnObjective(MEAN_RETURNS), VarianceObjective(COVARIANCE)]
    )
    limits = PortfolioConstraints(budget=1.0, lower_bounds=0.0, upper_bounds=1.0)
    front = TchebycheffNonlinearSolver(mixed, limits, n_restarts=2, seed=1).solve_front(
        n_partitions=6
    )
    assert len(front) >= 2
    assert front.weights.sum(axis=1) == pytest.approx(1.0, abs=1e-6)
    assert np.all(front.weights >= -1e-6)
    # The minimum-variance end must beat any single-asset portfolio.
    assert front.objectives[:, 1].min() < min(np.diag(COVARIANCE))


# ==============================================================================
# 6. NSGA-II
# ==============================================================================


def test_nsga2_returns_a_feasible_non_dominated_front(
    objectives: ObjectiveSet, constraints: PortfolioConstraints
) -> None:
    """Every individual is on the budget simplex and inside the box."""
    front = NSGAII(
        objectives, constraints, NSGAIIConfig(population_size=60, n_generations=40, seed=1)
    ).run()

    assert len(front) > 5
    assert front.weights.sum(axis=1) == pytest.approx(1.0, abs=1e-9)
    assert np.all(front.weights >= -1e-9)
    assert np.all(front.weights <= 1.0 + 1e-9)
    assert non_dominated_mask(front.minimisation_objectives).all()


def test_nsga2_is_reproducible(
    objectives: ObjectiveSet, constraints: PortfolioConstraints
) -> None:
    """The same seed gives the same front - a hard requirement for model audit."""
    config = NSGAIIConfig(population_size=40, n_generations=20, seed=123)
    first = NSGAII(objectives, constraints, config).run()
    second = NSGAII(objectives, constraints, config).run()
    assert first.objectives == pytest.approx(second.objectives)


def test_nsga2_never_dominates_the_exact_front(
    objectives: ObjectiveSet,
    constraints: PortfolioConstraints,
    reference_front: ParetoFront,
) -> None:
    """The decisive correctness check on the whole Step 2 stack.

    Tchebycheff-LP points are *globally* optimal, so no portfolio whatsoever
    can dominate one.  If NSGA-II ever did, either the genetic algorithm is
    violating a constraint or the two routes disagree about the objectives -
    and both are serious bugs.  This test is the reason to build the exact
    solver at all.
    """
    front = NSGAII(
        objectives, constraints, NSGAIIConfig(population_size=80, n_generations=60, seed=2)
    ).run()
    assert coverage(front.minimisation_objectives, reference_front.minimisation_objectives) == 0.0


def test_nsga2_converges_on_the_two_objective_problem(
    returns: np.ndarray, constraints: PortfolioConstraints
) -> None:
    """With two objectives the front is recovered almost exactly.

    Crowding distance is a reliable diversity operator in two dimensions, so
    this is where the genetic algorithm should be held to a tight standard:
    essentially all of the dominated hypervolume, and a small IGD.
    """
    objective_set = ObjectiveSet(
        [ExpectedReturnObjective(MEAN_RETURNS), CVaRObjective.from_returns(returns, BETA)]
    )
    exact = TchebycheffLPSolver(objective_set, constraints).solve_front(n_partitions=40)
    evolved = NSGAII(
        objective_set,
        constraints,
        NSGAIIConfig(population_size=100, n_generations=80, seed=3),
    ).run()

    metrics = compare_fronts(evolved, exact)
    assert metrics["hypervolume_ratio"] > 0.98
    assert metrics["igd"] < 0.02
    assert metrics["coverage_candidate_over_reference"] == 0.0


def test_nsga2_recovers_most_of_the_three_objective_front(
    objectives: ObjectiveSet,
    constraints: PortfolioConstraints,
    reference_front: ParetoFront,
) -> None:
    """At three objectives the algorithm should still capture most of the volume.

    The bar is deliberately lower than the two-objective case: crowding
    distance is a known-weak diversity mechanism beyond two dimensions, which
    is exactly why the exact front is worth having as a yardstick.
    """
    front = NSGAII(
        objectives, constraints, NSGAIIConfig(population_size=100, n_generations=100, seed=4)
    ).run()
    metrics = compare_fronts(front, reference_front)
    assert metrics["hypervolume_ratio"] > 0.90


def test_nsga2_respects_a_return_floor(
    objectives: ObjectiveSet, returns: np.ndarray
) -> None:
    """Constrained domination drives the population into the feasible region."""
    floor = 0.011
    limited = PortfolioConstraints(
        budget=1.0, lower_bounds=0.0, upper_bounds=1.0, min_expected_return=floor
    )
    front = NSGAII(
        objectives, limited, NSGAIIConfig(population_size=80, n_generations=60, seed=5)
    ).run()
    assert front.objectives[:, 0].min() >= floor - 1e-8


def test_nsga2_records_convergence_history(
    objectives: ObjectiveSet, constraints: PortfolioConstraints
) -> None:
    """Diagnostics are captured so a run can be inspected after the fact."""
    front = NSGAII(
        objectives,
        constraints,
        NSGAIIConfig(population_size=40, n_generations=30, seed=6, track_every=10),
    ).run()
    assert len(front.history) >= 3
    for record in front.history:
        assert set(record) >= {"generation", "hypervolume", "front_size", "n_feasible"}
        assert record["hypervolume"] >= 0.0


def test_nsga2_seeding_injects_known_solutions(
    objectives: ObjectiveSet, constraints: PortfolioConstraints
) -> None:
    """Seeded portfolios survive into the run when they are Pareto-optimal."""
    seeds = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    front = NSGAII(
        objectives, constraints, NSGAIIConfig(population_size=40, n_generations=5, seed=7)
    ).run(seed_solutions=seeds)
    # The all-small-cap seed is the unique maximum-return portfolio, so it must
    # be on any correct front.
    assert front.objectives[:, 0].max() == pytest.approx(MEAN_RETURNS[2], abs=1e-9)


def test_nsga2_alternative_operators_run_and_stay_feasible(
    objectives: ObjectiveSet, constraints: PortfolioConstraints
) -> None:
    """The natively feasible operator pair is a drop-in replacement."""
    front = NSGAII(
        objectives,
        constraints,
        NSGAIIConfig(population_size=40, n_generations=30, seed=8),
        crossover=SimplexBlendCrossover(),
        mutation=WeightTransferMutation(probability=0.3, n_trades=2),
    ).run()
    assert front.weights.sum(axis=1) == pytest.approx(1.0, abs=1e-9)
    assert front.metadata["crossover"] == "SimplexBlendCrossover"
    assert front.metadata["mutation"] == "WeightTransferMutation"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"population_size": 41}, "even and at least 4"),
        ({"population_size": 2}, "even and at least 4"),
        ({"n_generations": 0}, "at least 1"),
        ({"initial_concentration": 0.0}, "must be positive"),
        ({"track_every": 0}, "at least 1"),
    ],
)
def test_nsga2_config_is_validated(kwargs: dict, message: str) -> None:
    """Bad hyper-parameters fail at construction, not mid-run."""
    with pytest.raises(ValueError, match=message):
        NSGAIIConfig(**kwargs)


# ==============================================================================
# 7. Comparison harness
# ==============================================================================


def test_compare_fronts_scores_a_perfect_match(reference_front: ParetoFront) -> None:
    """A front compared with itself scores perfectly on every indicator."""
    metrics = compare_fronts(reference_front, reference_front)
    assert metrics["igd"] == pytest.approx(0.0)
    assert metrics["gd"] == pytest.approx(0.0)
    assert metrics["hypervolume_ratio"] == pytest.approx(1.0)
    assert metrics["coverage_candidate_over_reference"] == pytest.approx(0.0)


def test_compare_fronts_validates_its_inputs(reference_front: ParetoFront) -> None:
    """Empty fronts and objective-count mismatches are rejected."""
    empty = ParetoFront(
        weights=np.zeros((0, 3)),
        objectives=np.zeros((0, 3)),
        minimisation_objectives=np.zeros((0, 3)),
        names=reference_front.names,
    )
    with pytest.raises(ValueError, match="non-empty"):
        compare_fronts(empty, reference_front)

    mismatched = ParetoFront(
        weights=np.zeros((2, 3)),
        objectives=np.zeros((2, 2)),
        minimisation_objectives=np.zeros((2, 2)),
        names=["a", "b"],
    )
    with pytest.raises(ValueError, match="Objective count mismatch"):
        compare_fronts(mismatched, reference_front)


def test_pareto_front_filters_and_exports(reference_front: ParetoFront) -> None:
    """Filtering is idempotent, and the frame carries objectives plus weights."""
    filtered = reference_front.filter_non_dominated()
    assert len(filtered) == len(reference_front)

    frame = reference_front.to_frame()
    assert list(frame.columns[:3]) == reference_front.names
    assert frame.shape == (len(reference_front), 3 + reference_front.n_assets)


def test_default_reference_point_is_dominated_by_the_front(
    reference_front: ParetoFront,
) -> None:
    """The reference must be worse than every solution or hypervolume is zero."""
    point = default_reference_point(reference_front.minimisation_objectives)
    assert np.all(point >= reference_front.minimisation_objectives.max(axis=0))
    assert reference_front.hypervolume(point) > 0.0
