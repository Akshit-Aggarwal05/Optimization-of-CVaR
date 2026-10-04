"""Regression and property tests for :mod:`src.models.risk_metrics`.

The suite is organised in four layers:

1.  **Paper regression** - reproduce Tables 3 and 4 of Rockafellar & Uryasev
    (2000) exactly, and reproduce Table 6 (Sobol-sampled LP) to within the
    tolerance the paper itself reports.
2.  **Theorem verification** - assert numerically that CVaR is the minimum of
    the auxiliary function and that VaR is the left endpoint of its argmin
    (Theorem 1), and that the LP optimum satisfies Theorem 2.
3.  **Coherence properties** - monotonicity, translation equivariance, positive
    homogeneity and subadditivity of CVaR, none of which VaR satisfies.
4.  **Contract tests** - constraint handling and input validation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import erfinv
from scipy.stats import norm

from src.data.paper_datasets import (
    COVARIANCE,
    MEAN_RETURNS,
    MIN_VARIANCE_VARIANCE,
    MIN_VARIANCE_WEIGHTS,
    REFERENCE_CVAR,
    REFERENCE_VAR,
    TARGET_RETURN,
    sample_normal_returns,
)
from src.models.risk_metrics import (
    CVaRLinearProgram,
    HistoricalRiskEstimator,
    LossDistribution,
    ParametricRiskEstimator,
    PortfolioConstraints,
    QuantileConvention,
    cvar_efficient_frontier,
    historical_var,
    historical_var_cvar,
    parametric_cvar,
    parametric_var,
    portfolio_loss_scenarios,
    ru_auxiliary_function,
)

BETAS = (0.90, 0.95, 0.99)


@pytest.fixture(scope="module")
def sobol_returns() -> np.ndarray:
    """16,384 quasi-random scenarios from N(m, V) - the Table 6 setting."""
    return sample_normal_returns(n_scenarios=16_384, engine="sobol", seed=20_000)


@pytest.fixture(scope="module")
def parametric_engine() -> ParametricRiskEstimator:
    """Analytic engine on the paper's Table 1 / Table 2 inputs."""
    return ParametricRiskEstimator(MEAN_RETURNS, COVARIANCE)


# ==============================================================================
# 1. Regression against the published tables
# ==============================================================================


def test_table_3_portfolio_variance(parametric_engine: ParametricRiskEstimator) -> None:
    """Section 3 quotes sigma^2(x*) = 0.00378529 for the Table 3 portfolio.

    Table 3 prints the weights to six decimals, so the reconstructed variance
    can only be trusted to about 1e-8; that rounding, not the estimator, sets
    the tolerance here.
    """
    variance = parametric_engine.volatility(MIN_VARIANCE_WEIGHTS) ** 2
    assert variance == pytest.approx(MIN_VARIANCE_VARIANCE, abs=1e-8)


def test_table_3_return_constraint_is_active(
    parametric_engine: ParametricRiskEstimator,
) -> None:
    """mu(x*) = -0.011: eq. (15) binds, which is what activates the Proposition."""
    assert parametric_engine.mean_loss(MIN_VARIANCE_WEIGHTS) == pytest.approx(
        -TARGET_RETURN, abs=1e-6
    )


@pytest.mark.parametrize("beta", BETAS)
def test_table_4_var_and_cvar(
    parametric_engine: ParametricRiskEstimator, beta: float
) -> None:
    """Reproduce Table 4 from eqs. (18)-(19) applied to the Table 3 portfolio."""
    estimate = parametric_engine.estimate(MIN_VARIANCE_WEIGHTS, beta)
    assert estimate.var == pytest.approx(REFERENCE_VAR[beta], abs=1e-6)
    assert estimate.cvar == pytest.approx(REFERENCE_CVAR[beta], abs=1e-6)


@pytest.mark.parametrize("beta", BETAS)
def test_erf_forms_of_c1_and_c2(beta: float) -> None:
    """The erf-based coefficients of eqs. (18)-(19) match the scipy normal forms.

    eq. (18):  c1(beta) = sqrt(2) * erfinv(2*beta - 1)
    eq. (19):  c2(beta) = 1 / (sqrt(2*pi) * exp(erfinv(2*beta-1)^2) * (1-beta))
    """
    c1_paper = np.sqrt(2.0) * erfinv(2.0 * beta - 1.0)
    c2_paper = 1.0 / (
        np.sqrt(2.0 * np.pi) * np.exp(erfinv(2.0 * beta - 1.0) ** 2) * (1.0 - beta)
    )
    # Unit-scale loss: VaR = c1, CVaR = c2.
    assert parametric_var(0.0, 1.0, beta) == pytest.approx(c1_paper, rel=1e-12)
    assert parametric_cvar(0.0, 1.0, beta) == pytest.approx(c2_paper, rel=1e-12)


@pytest.mark.parametrize("beta", BETAS)
def test_lp_reproduces_table_6(sobol_returns: np.ndarray, beta: float) -> None:
    """The Sobol-sampled LP converges to the minimum-variance benchmark.

    Table 6 of the paper reports that with a quasi-random sequence and more
    than 10,000 scenarios the VaR and CVaR from the Min-CVaR LP differ from the
    Min-Variance analytic values by less than 1%.  The Proposition guarantees
    the two problems share an optimum here, so this is a genuine end-to-end
    check of the constraint matrix, not a self-consistency check.
    """
    constraints = PortfolioConstraints(
        budget=1.0, lower_bounds=0.0, min_expected_return=TARGET_RETURN
    )
    program = CVaRLinearProgram.from_returns(
        sobol_returns, beta, constraints, expected_returns=MEAN_RETURNS
    )
    solution = program.solve()

    assert solution.success
    assert solution.cvar == pytest.approx(REFERENCE_CVAR[beta], rel=0.01)
    assert solution.reported_var == pytest.approx(REFERENCE_VAR[beta], rel=0.02)
    # Weights land near Table 3; the paper's own Table 6 rows wander by a few
    # points at this sample size because the optimum is very flat.
    assert solution.weights == pytest.approx(MIN_VARIANCE_WEIGHTS, abs=0.05)


def test_lp_return_constraint_binds(sobol_returns: np.ndarray) -> None:
    """eq. (15) is active at the optimum, as the paper reports for this data."""
    constraints = PortfolioConstraints(
        budget=1.0, lower_bounds=0.0, min_expected_return=TARGET_RETURN
    )
    solution = CVaRLinearProgram.from_returns(
        sobol_returns, 0.95, constraints, expected_returns=MEAN_RETURNS
    ).solve()
    assert float(MEAN_RETURNS @ solution.weights) == pytest.approx(
        TARGET_RETURN, abs=1e-8
    )


def test_lp_beats_min_variance_in_sample(sobol_returns: np.ndarray) -> None:
    """In sample the LP optimum must dominate any feasible comparator.

    Optimality is only meaningful relative to the *same* scenario set, so the
    benchmark portfolio is scored on the identical sample.
    """
    beta = 0.95
    constraints = PortfolioConstraints(
        budget=1.0, lower_bounds=0.0, min_expected_return=TARGET_RETURN
    )
    solution = CVaRLinearProgram.from_returns(
        sobol_returns, beta, constraints, expected_returns=MEAN_RETURNS
    ).solve()

    engine = HistoricalRiskEstimator(sobol_returns)
    benchmark_cvar = engine.cvar(MIN_VARIANCE_WEIGHTS, beta)
    assert solution.cvar <= benchmark_cvar + 1e-9


# ==============================================================================
# 2. Theorem 1 and Theorem 2 verification
# ==============================================================================


@pytest.mark.parametrize("beta", BETAS)
def test_cvar_is_the_minimum_of_the_auxiliary_function(beta: float) -> None:
    """Theorem 1: phi_beta(x) = min_alpha F_beta(x, alpha), eq. (5).

    The discrete-atom estimator in ``historical_var_cvar`` is compared with a
    brute-force minimisation of eq. (17) over alpha.
    """
    rng = np.random.default_rng(7)
    losses = rng.standard_t(df=5, size=4_000) * 0.02 - 0.001

    estimate = historical_var_cvar(losses, beta)
    brute = minimize_scalar(
        lambda a: ru_auxiliary_function(losses, a, beta),
        bounds=(losses.min() - 1.0, losses.max() + 1.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    assert estimate.cvar == pytest.approx(float(brute.fun), rel=1e-9)


@pytest.mark.parametrize("beta", BETAS)
def test_var_is_the_left_endpoint_of_the_argmin(beta: float) -> None:
    """Theorem 1, eq. (7): VaR attains the minimum, and nothing to its left does.

    F is convex and piecewise linear, so it suffices to check that F(VaR) is the
    minimum value and that F is strictly larger just below VaR.
    """
    rng = np.random.default_rng(11)
    losses = rng.normal(loc=-0.005, scale=0.03, size=5_000)

    estimate = historical_var_cvar(losses, beta)
    at_var = ru_auxiliary_function(losses, estimate.var, beta)
    assert at_var == pytest.approx(estimate.cvar, rel=1e-12)

    step = 1e-4
    assert ru_auxiliary_function(losses, estimate.var - step, beta) > estimate.cvar


@pytest.mark.parametrize("beta", BETAS)
def test_cvar_decomposition_into_var_and_cvar_plus(beta: float) -> None:
    """CVaR = lambda*VaR + (1-lambda)*CVaR+ (Rockafellar & Uryasev 2002).

    This is the identity that makes the discrete estimator exact; a naive
    "mean of losses above VaR" would fail it whenever beta*q is not an integer.
    """
    rng = np.random.default_rng(3)
    losses = rng.normal(size=997)  # deliberately not a round number

    est = historical_var_cvar(losses, beta)
    reconstructed = est.var_weight * est.var + (1.0 - est.var_weight) * est.cvar_plus
    assert reconstructed == pytest.approx(est.cvar, rel=1e-12)
    assert 0.0 <= est.var_weight <= 1.0


def test_lp_cvar_agrees_with_empirical_recomputation(sobol_returns: np.ndarray) -> None:
    """Theorem 2: the LP optimal value equals the CVaR of the optimal portfolio."""
    for beta in BETAS:
        solution = CVaRLinearProgram.from_returns(
            sobol_returns,
            beta,
            PortfolioConstraints(min_expected_return=TARGET_RETURN),
            expected_returns=MEAN_RETURNS,
        ).solve()
        assert solution.cvar == pytest.approx(solution.empirical.cvar, rel=1e-7)
        assert solution.var_is_left_endpoint


def test_lp_auxiliary_variables_equal_the_positive_parts(
    sobol_returns: np.ndarray,
) -> None:
    """At the optimum u_k = [f(x,y_k) - alpha]^+ exactly, as the reduction requires."""
    beta = 0.95
    program = CVaRLinearProgram.from_returns(
        sobol_returns, beta, PortfolioConstraints(min_expected_return=TARGET_RETURN),
        expected_returns=MEAN_RETURNS,
    )
    solution = program.solve()
    losses = portfolio_loss_scenarios(solution.weights, sobol_returns)
    implied = np.maximum(losses - solution.var, 0.0)
    # Rebuild F from the implied slacks; equality with the LP objective proves
    # the linearisation is tight.
    rebuilt = solution.var + implied.sum() / (losses.size * (1.0 - beta))
    assert rebuilt == pytest.approx(solution.cvar, rel=1e-9)


def test_objective_rescaling_is_equivalent(sobol_returns: np.ndarray) -> None:
    """The (1-beta) rescaling suggested after Theorem 1 leaves the optimum alone."""
    constraints = PortfolioConstraints(min_expected_return=TARGET_RETURN)
    kwargs = dict(returns=sobol_returns, beta=0.99, constraints=constraints,
                  expected_returns=MEAN_RETURNS)
    scaled = CVaRLinearProgram.from_returns(**kwargs, rescale_objective=True).solve()
    plain = CVaRLinearProgram.from_returns(**kwargs, rescale_objective=False).solve()
    assert scaled.cvar == pytest.approx(plain.cvar, rel=1e-9)


# ==============================================================================
# 3. Coherence properties of CVaR
# ==============================================================================


@pytest.mark.parametrize("beta", BETAS)
def test_cvar_dominates_var(beta: float) -> None:
    """The definitions ensure beta-VaR <= beta-CVaR, for both estimator families."""
    rng = np.random.default_rng(5)
    losses = rng.standard_t(df=4, size=2_500)
    empirical = historical_var_cvar(losses, beta)
    assert empirical.cvar >= empirical.var

    analytic = ParametricRiskEstimator(MEAN_RETURNS, COVARIANCE).estimate(
        MIN_VARIANCE_WEIGHTS, beta
    )
    assert analytic.cvar >= analytic.var


def test_cvar_is_subadditive_where_var_need_not_be() -> None:
    """CVaR(w1 + w2) <= CVaR(w1) + CVaR(w2): the coherence property VaR lacks."""
    rng = np.random.default_rng(13)
    returns = rng.standard_t(df=3, size=(6_000, 2)) * 0.01
    beta = 0.95
    engine = HistoricalRiskEstimator(returns)

    combined = engine.cvar([0.5, 0.5], beta)
    standalone = 0.5 * engine.cvar([1.0, 0.0], beta) + 0.5 * engine.cvar([0.0, 1.0], beta)
    assert combined <= standalone + 1e-12


def test_cvar_is_translation_equivariant_and_homogeneous() -> None:
    """CVaR(L + c) = CVaR(L) + c and CVaR(t*L) = t*CVaR(L) for t > 0."""
    rng = np.random.default_rng(17)
    losses = rng.normal(size=3_000)
    beta, shift, scale = 0.95, 0.037, 2.5

    base = historical_var_cvar(losses, beta).cvar
    assert historical_var_cvar(losses + shift, beta).cvar == pytest.approx(
        base + shift, rel=1e-12
    )
    assert historical_var_cvar(losses * scale, beta).cvar == pytest.approx(
        base * scale, rel=1e-12
    )


def test_historical_converges_to_parametric_under_normality() -> None:
    """With enough Gaussian scenarios the empirical estimator matches eqs. (18)-(19)."""
    beta = 0.95
    returns = sample_normal_returns(n_scenarios=65_536, engine="sobol", seed=1)
    empirical = HistoricalRiskEstimator(returns).estimate(MIN_VARIANCE_WEIGHTS, beta)
    analytic = ParametricRiskEstimator(MEAN_RETURNS, COVARIANCE).estimate(
        MIN_VARIANCE_WEIGHTS, beta
    )
    assert empirical.var == pytest.approx(analytic.var, rel=0.01)
    assert empirical.cvar == pytest.approx(analytic.cvar, rel=0.01)


def test_student_t_has_fatter_tails_and_converges_to_normal() -> None:
    """Student-t CVaR exceeds the Gaussian one, and collapses onto it as dof grows."""
    beta, mean_loss, vol = 0.99, -0.01, 0.05
    gaussian = parametric_cvar(mean_loss, vol, beta)
    fat = parametric_cvar(mean_loss, vol, beta, LossDistribution.STUDENT_T, dof=4)
    nearly_normal = parametric_cvar(
        mean_loss, vol, beta, LossDistribution.STUDENT_T, dof=5_000
    )
    assert fat > gaussian
    assert nearly_normal == pytest.approx(gaussian, rel=1e-2)


@pytest.mark.parametrize("q", [999, 1_000, 1_001])
@pytest.mark.parametrize("beta", BETAS)
def test_quantile_conventions_agree_to_one_order_statistic(q: int, beta: float) -> None:
    """The two conventions differ by at most one order statistic.

    Neither dominates the other: ``numpy``'s linear method interpolates at
    index ``beta*(q-1)`` while the order-statistic convention sits at
    ``ceil(beta*q) - 1``, and which is larger depends on where ``beta*q`` falls
    between integers.  What *is* guaranteed is that both land inside the same
    adjacent pair of sample points - the practical statement of "same quantile,
    different tie-breaking".  Only the order statistic minimises eq. (17),
    which is why it is the default for anything feeding the optimiser.
    """
    rng = np.random.default_rng(23)
    losses = rng.normal(size=q)
    ordered = np.sort(losses)

    order_stat = historical_var(losses, beta, QuantileConvention.ORDER_STATISTIC)
    interpolated = historical_var(losses, beta, QuantileConvention.INTERPOLATED)

    assert interpolated == pytest.approx(np.quantile(losses, beta), rel=1e-12)
    k = int(np.ceil(beta * q))  # 1-based index of the order-statistic VaR
    assert order_stat == ordered[k - 1]
    lower = ordered[max(k - 2, 0)]
    upper = ordered[min(k, q - 1)]
    assert lower - 1e-12 <= interpolated <= upper + 1e-12


# ==============================================================================
# 4. Constraints, general loss matrices, and input contracts
# ==============================================================================


def test_box_constraints_are_respected(sobol_returns: np.ndarray) -> None:
    """Per-asset caps bind and the budget still holds."""
    constraints = PortfolioConstraints(
        budget=1.0, lower_bounds=0.05, upper_bounds=0.40, min_expected_return=None
    )
    solution = CVaRLinearProgram.from_returns(sobol_returns, 0.95, constraints).solve()
    assert np.all(solution.weights >= 0.05 - 1e-9)
    assert np.all(solution.weights <= 0.40 + 1e-9)
    assert solution.weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_extra_linear_inequality_is_applied(sobol_returns: np.ndarray) -> None:
    """A sector cap x1 + x3 <= 0.6 supplied through linear_inequalities binds."""
    cap = np.array([[1.0, 0.0, 1.0]])
    constraints = PortfolioConstraints(
        budget=1.0, lower_bounds=0.0, linear_inequalities=(cap, np.array([0.6]))
    )
    solution = CVaRLinearProgram.from_returns(sobol_returns, 0.95, constraints).solve()
    assert float((cap @ solution.weights)[0]) <= 0.6 + 1e-9


def test_unconstrained_return_target_lowers_cvar(sobol_returns: np.ndarray) -> None:
    """Dropping eq. (15) can only relax the problem, so CVaR must not increase."""
    beta = 0.95
    with_target = CVaRLinearProgram.from_returns(
        sobol_returns, beta,
        PortfolioConstraints(min_expected_return=TARGET_RETURN),
        expected_returns=MEAN_RETURNS,
    ).solve()
    without_target = CVaRLinearProgram.from_returns(
        sobol_returns, beta, PortfolioConstraints(), expected_returns=MEAN_RETURNS
    ).solve()
    assert without_target.cvar <= with_target.cvar + 1e-9


def test_infeasible_return_target_raises(sobol_returns: np.ndarray) -> None:
    """A target above the best attainable expected return is infeasible."""
    impossible = float(MEAN_RETURNS.max()) + 0.05
    program = CVaRLinearProgram.from_returns(
        sobol_returns,
        0.95,
        PortfolioConstraints(min_expected_return=impossible),
        expected_returns=MEAN_RETURNS,
    )
    with pytest.raises(RuntimeError, match="linear programme failed"):
        program.solve()


def test_general_loss_matrix_matches_hedging_formulation() -> None:
    """eq. (23): f(x, y) = x^T (m - y) is handled by the loss-matrix constructor.

    Passing ``L[k] = m - y^k`` must give the same optimum as passing the
    equivalent return matrix ``Y' = y - m`` through ``from_returns``, because
    ``-x^T (y - m) = x^T (m - y)``.
    """
    rng = np.random.default_rng(29)
    prices_now = np.array([100.0, 50.0, 25.0])
    prices_next = prices_now * (1.0 + rng.normal(scale=0.02, size=(2_000, 3)))
    loss_matrix = prices_now - prices_next

    constraints = PortfolioConstraints(budget=1.0, lower_bounds=0.0)
    direct = CVaRLinearProgram(loss_matrix, 0.95, constraints).solve()
    via_returns = CVaRLinearProgram.from_returns(
        -loss_matrix, 0.95, constraints
    ).solve()
    assert direct.cvar == pytest.approx(via_returns.cvar, rel=1e-9)
    assert direct.weights == pytest.approx(via_returns.weights, abs=1e-8)


def test_negative_var_is_allowed() -> None:
    """A profitable book has negative VaR; alpha must not be clamped at zero."""
    rng = np.random.default_rng(31)
    returns = rng.normal(loc=0.20, scale=0.01, size=(2_000, 2))
    solution = CVaRLinearProgram.from_returns(
        returns, 0.90, PortfolioConstraints(budget=1.0, lower_bounds=0.0)
    ).solve()
    assert solution.reported_var < 0.0
    assert solution.cvar < 0.0


def test_efficient_frontier_is_monotone(sobol_returns: np.ndarray) -> None:
    """CVaR is non-decreasing in the return target along the efficient frontier."""
    targets = np.linspace(0.006, 0.013, 6)
    frontier = cvar_efficient_frontier(
        sobol_returns, 0.95, targets, PortfolioConstraints(budget=1.0, lower_bounds=0.0)
    )
    assert len(frontier) == len(targets)
    cvars = [point.cvar for point in frontier]
    assert np.all(np.diff(cvars) >= -1e-9)


def test_lp_uses_a_sparse_constraint_matrix(sobol_returns: np.ndarray) -> None:
    """The scenario block must stay sparse: dense storage is O(q^2) memory."""
    from scipy import sparse

    problem = CVaRLinearProgram.from_returns(sobol_returns, 0.95).build()
    a_ub = problem["A_ub"]
    assert sparse.issparse(a_ub)
    q, n = sobol_returns.shape
    assert a_ub.shape == (q, n + 1 + q)
    # Each scenario row has n loss coefficients, one alpha and one slack entry.
    assert a_ub.nnz == q * (n + 2)


@pytest.mark.parametrize("bad_beta", [0.0, 1.0, -0.1, 1.5, np.nan])
def test_beta_must_be_interior(bad_beta: float) -> None:
    """beta = 0 or 1 breaks the (1-beta)^-1 factor and is rejected."""
    with pytest.raises(ValueError, match="beta"):
        historical_var_cvar(np.array([0.1, 0.2, 0.3]), bad_beta)


def test_non_finite_inputs_are_rejected() -> None:
    """NaN losses would silently poison the sort order; reject them up front."""
    with pytest.raises(ValueError, match="NaN or infinite"):
        historical_var_cvar(np.array([0.1, np.nan, 0.3]), 0.95)


def test_weight_dimension_mismatch_is_rejected(sobol_returns: np.ndarray) -> None:
    """Weight/scenario dimension mismatches must fail loudly, not broadcast."""
    engine = HistoricalRiskEstimator(sobol_returns)
    with pytest.raises(ValueError, match="defined over 3 assets"):
        engine.estimate([0.5, 0.5], 0.95)


def test_asymmetric_covariance_is_rejected() -> None:
    """A non-symmetric covariance matrix is a data error, not a rounding issue."""
    bad = np.array([[1.0, 0.5], [0.2, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        ParametricRiskEstimator(np.zeros(2), bad)


def test_non_psd_covariance_is_rejected() -> None:
    """Negative eigenvalues would give an imaginary volatility."""
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1
    with pytest.raises(ValueError, match="positive semi-definite"):
        ParametricRiskEstimator(np.zeros(2), bad)


def test_infeasible_box_is_rejected() -> None:
    """lower > upper describes an empty feasible set."""
    with pytest.raises(ValueError, match="Infeasible box"):
        PortfolioConstraints(lower_bounds=0.5, upper_bounds=0.2).bounds_for(3)


def test_thin_tail_emits_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """q*(1-beta) below the threshold means the tail average is noise-dominated."""
    rng = np.random.default_rng(37)
    with caplog.at_level("WARNING", logger="src.models.risk_metrics"):
        historical_var_cvar(rng.normal(size=200), 0.99)
    assert "Thin tail" in caplog.text


def test_student_t_requires_dof_above_two_when_standardised() -> None:
    """Variance is infinite for dof <= 2, so a standard deviation is meaningless."""
    with pytest.raises(ValueError, match="dof <= 2"):
        parametric_var(0.0, 1.0, 0.95, LossDistribution.STUDENT_T, dof=1.5)
