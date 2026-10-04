"""Primal feasibility of the Rockafellar-Uryasev linear programme.

The paper states the reduction in words immediately after eq. (17): minimise

    alpha + 1/(q(1-beta)) * sum_k u_k

subject to eq. (11), eq. (15), and

    u_k >= 0        and        x'y_k + alpha + u_k >= 0        for k = 1 ... q

These tests assert those constraints against the **solver's own output** - the
auxiliary variables are read back through
:attr:`~src.models.risk_metrics.CVaRSolution.tail_slacks` rather than
recomputed - so a mis-signed row, a dropped block or a transposed index in the
constraint matrix is caught directly rather than inferred from a plausible
objective value.

The second half of the module gates the published tables: these tests fail if
the solver drifts away from Rockafellar & Uryasev (2000), which is what makes
the CI badge a statement about reproducing the paper rather than merely about
the code importing.
"""

from __future__ import annotations

import numpy as np
import pytest

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
    ParametricRiskEstimator,
    PortfolioConstraints,
)

BETAS = (0.90, 0.95, 0.99)

#: Scenario count for the feasibility checks.  Large enough to exercise the
#: sparse block, small enough to keep CI inside a minute.
N_SCENARIOS = 8_192

#: Tolerance for the LP's primal feasibility.  HiGHS defaults to about 1e-7;
#: anything looser would let a genuine sign error through.
FEASIBILITY_ATOL = 1e-9


@pytest.fixture(scope="module")
def returns() -> np.ndarray:
    """Quasi-random scenarios from the paper's N(m, V)."""
    return sample_normal_returns(N_SCENARIOS, engine="sobol", seed=20_000)


@pytest.fixture(scope="module")
def constraints() -> PortfolioConstraints:
    """The paper's feasible set X: eq. (11) plus eq. (15)."""
    return PortfolioConstraints(
        budget=1.0, lower_bounds=0.0, min_expected_return=TARGET_RETURN
    )


def _solve(returns: np.ndarray, beta: float, constraints: PortfolioConstraints):
    """Solve the programme with the paper's expected-return vector."""
    return CVaRLinearProgram.from_returns(
        returns, beta, constraints, expected_returns=MEAN_RETURNS
    ).solve()


# ==============================================================================
# 1. The two constraints printed after eq. (17)
# ==============================================================================


@pytest.mark.parametrize("beta", BETAS)
def test_tail_slacks_are_non_negative(
    returns: np.ndarray, constraints: PortfolioConstraints, beta: float
) -> None:
    """``u_k >= 0`` for every scenario.

    The first of the two constraints the paper states.  A negative slack would
    let the objective buy a spurious CVaR reduction, so this is asserted on the
    solver's actual variables rather than assumed from the bound declaration.
    """
    solution = _solve(returns, beta, constraints)
    slacks = solution.tail_slacks

    assert slacks.shape == (returns.shape[0],)
    assert np.all(slacks >= -FEASIBILITY_ATOL), (
        f"minimum slack {slacks.min():.3e} violates u_k >= 0"
    )


@pytest.mark.parametrize("beta", BETAS)
def test_scenario_constraint_holds_for_every_scenario(
    returns: np.ndarray, constraints: PortfolioConstraints, beta: float
) -> None:
    """``x'y_k + alpha + u_k >= 0`` for every scenario.

    The second constraint, and the one that encodes the linearisation of the
    positive part.  It is checked on all ``q`` rows, not a sample: a single
    violated scenario would mean the reported CVaR understates the true tail.
    """
    solution = _solve(returns, beta, constraints)
    residual = returns @ solution.weights + solution.var + solution.tail_slacks

    assert residual.shape == (returns.shape[0],)
    worst = float(residual.min())
    assert worst >= -FEASIBILITY_ATOL, (
        f"scenario constraint violated by {worst:.3e} at beta={beta}"
    )


@pytest.mark.parametrize("beta", BETAS)
def test_slacks_equal_the_positive_part_at_the_optimum(
    returns: np.ndarray, constraints: PortfolioConstraints, beta: float
) -> None:
    """At the optimum ``u_k = max(f(x, y_k) - alpha, 0)`` exactly.

    The constraint only bounds ``u_k`` from below; it is the *minimisation*
    that drives it down to the positive part.  Verifying the equality confirms
    the linearisation is tight, which is what makes the LP's optimal value
    equal to CVaR rather than merely an upper bound on it.
    """
    solution = _solve(returns, beta, constraints)
    losses = -(returns @ solution.weights)  # eq. (12)
    expected = np.maximum(losses - solution.var, 0.0)

    assert solution.tail_slacks == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("beta", BETAS)
def test_objective_rebuilt_from_the_auxiliary_variables(
    returns: np.ndarray, constraints: PortfolioConstraints, beta: float
) -> None:
    """Reassembling eq. (17) from ``(alpha, u)`` reproduces the reported CVaR.

    An end-to-end check that the objective coefficients carry the
    ``1/(q(1-beta))`` normalisation the paper specifies, and that the optional
    ``(1-beta)`` rescaling is divided back out correctly.
    """
    solution = _solve(returns, beta, constraints)
    q = returns.shape[0]
    rebuilt = solution.var + solution.tail_slacks.sum() / (q * (1.0 - beta))

    assert rebuilt == pytest.approx(solution.cvar, rel=1e-9)


@pytest.mark.parametrize("beta", BETAS)
def test_portfolio_constraints_eleven_and_fifteen(
    returns: np.ndarray, constraints: PortfolioConstraints, beta: float
) -> None:
    """eq. (11) budget and non-negativity, and eq. (15) expected-return floor."""
    solution = _solve(returns, beta, constraints)

    assert solution.weights.sum() == pytest.approx(1.0, abs=1e-9)  # eq. (11)
    assert np.all(solution.weights >= -FEASIBILITY_ATOL)  # eq. (11)
    # eq. (15): mu(x) = -x'm <= -R, i.e. x'm >= R.
    assert float(MEAN_RETURNS @ solution.weights) >= TARGET_RETURN - 1e-9


def test_feasibility_survives_the_objective_rescaling(
    returns: np.ndarray, constraints: PortfolioConstraints
) -> None:
    """The ``(1-beta)`` rescaling must not disturb primal feasibility.

    The paper suggests minimising ``(1-beta) * F_beta`` for conditioning when
    ``1-beta`` is small.  That is a positive scaling of the objective, so the
    feasible set is untouched - and this asserts it rather than trusting it.
    """
    for rescale in (True, False):
        solution = CVaRLinearProgram.from_returns(
            returns, 0.99, constraints, expected_returns=MEAN_RETURNS,
            rescale_objective=rescale,
        ).solve()
        residual = returns @ solution.weights + solution.var + solution.tail_slacks
        assert np.all(solution.tail_slacks >= -FEASIBILITY_ATOL)
        assert float(residual.min()) >= -FEASIBILITY_ATOL


def test_slack_count_matches_the_scenario_count(
    returns: np.ndarray, constraints: PortfolioConstraints
) -> None:
    """One auxiliary variable per scenario, as the reduction requires.

    The paper's text after eq. (17) prints the index range as ``k = 1 ... r``
    while the objective sums ``k = 1 ... q``; ``r`` is a typo for ``q``. This
    test pins the correct reading.
    """
    solution = _solve(returns, 0.95, constraints)
    assert solution.tail_slacks.size == returns.shape[0]
    assert solution.n_scenarios == returns.shape[0]


# ==============================================================================
# 2. Published-table gates - these fail on numerical regression
# ==============================================================================


def test_table_3_minimum_variance_benchmark() -> None:
    """Table 3's portfolio and its variance, to publication precision.

    Table 3 prints weights to six decimals, so the reconstructed variance can
    only be trusted to about 1e-8; that rounding, not the estimator, sets the
    tolerance.
    """
    engine = ParametricRiskEstimator(MEAN_RETURNS, COVARIANCE)

    assert MIN_VARIANCE_WEIGHTS.sum() == pytest.approx(1.0, abs=1e-12)
    assert engine.volatility(MIN_VARIANCE_WEIGHTS) ** 2 == pytest.approx(
        MIN_VARIANCE_VARIANCE, abs=1e-8
    )
    # The paper states mu(x*) = -0.011, i.e. eq. (15) is active at the optimum,
    # which is the precondition for the Proposition to apply.
    assert engine.mean_loss(MIN_VARIANCE_WEIGHTS) == pytest.approx(
        -TARGET_RETURN, abs=1e-8
    )


@pytest.mark.parametrize("beta", BETAS)
def test_table_4_var_and_cvar_within_tolerance(beta: float) -> None:
    """Table 4, reproduced from eqs. (18)-(19). Fails on any real drift."""
    estimate = ParametricRiskEstimator(MEAN_RETURNS, COVARIANCE).estimate(
        MIN_VARIANCE_WEIGHTS, beta
    )
    assert estimate.var == pytest.approx(REFERENCE_VAR[beta], abs=1e-5)
    assert estimate.cvar == pytest.approx(REFERENCE_CVAR[beta], abs=1e-5)


@pytest.mark.parametrize("beta", BETAS)
def test_table_6_lp_stays_within_one_percent(
    returns: np.ndarray, constraints: PortfolioConstraints, beta: float
) -> None:
    """Table 6's claim: Sobol-sampled LP within 1% above 10,000 scenarios.

    This is the gate that makes the badge mean something. The LP is scored
    against the *analytic* minimum-variance benchmark, which the Proposition
    guarantees is the same optimum - so the comparison is independent of the
    solver being tested, and a regression cannot hide behind it.
    """
    solution = _solve(returns, beta, constraints)

    var_deviation = abs(solution.reported_var - REFERENCE_VAR[beta]) / REFERENCE_VAR[beta]
    cvar_deviation = abs(solution.cvar - REFERENCE_CVAR[beta]) / REFERENCE_CVAR[beta]

    assert cvar_deviation < 0.01, (
        f"CVaR deviates {cvar_deviation:.3%} from Table 4/6 at beta={beta}"
    )
    assert var_deviation < 0.02, (
        f"VaR deviates {var_deviation:.3%} from Table 4/6 at beta={beta}"
    )


def test_lp_converges_to_the_table_3_portfolio(
    returns: np.ndarray, constraints: PortfolioConstraints
) -> None:
    """Weights converge on Table 3, as the Proposition requires under normality."""
    solution = _solve(returns, 0.95, constraints)
    assert solution.weights == pytest.approx(MIN_VARIANCE_WEIGHTS, abs=0.05)
