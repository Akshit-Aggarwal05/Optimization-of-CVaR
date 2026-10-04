"""Tests for the NIKKEI option book of the paper's Section 4.

The division of labour here matters, because it is what separates reproduction
from reconstruction:

* **Exact** - Table 7's instruments, strikes, maturities, positions and marks;
  the spots derived from the equity rows; the implied volatilities inverted
  from the published values; and the base valuation, which must reproduce
  Table 7 to machine precision. These are asserted tightly.
* **Reconstructed** - the 1,000 Monte Carlo scenarios, which are not published.
  These are asserted only on *qualitative* properties the paper states, chiefly
  that a normal distribution fits the loss distribution poorly. Asserting their
  magnitudes against the paper would be asserting a coincidence.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import jarque_bera

from src.data.nikkei_portfolio import (
    CONTRACT_MULTIPLIER,
    PAPER_BETA,
    PAPER_CVAR_95,
    PAPER_VAR_95,
    RISK_FREE_RATE,
    SPOTS,
    TABLE_7,
    base_market_state,
    build_portfolio,
    calibrate_implied_volatilities,
    simulate_loss_scenarios,
)
from src.data.nikkei_portfolio import _diffusion_volatilities, _price_book
from src.models.risk_metrics import historical_var_cvar
from src.models.stress_tester import EuropeanOption, OptionType, black_scholes_price

#: Sum of Table 7's Value column, in 1,000 JPY.
TABLE_7_TOTAL = 12_493_368.0


@pytest.fixture(scope="module")
def implied() -> dict[str, float]:
    """Implied volatilities inverted from Table 7's published prices."""
    return calibrate_implied_volatilities()


# ==============================================================================
# 1. Table 7, reproduced exactly
# ==============================================================================


def test_table_7_has_eleven_instruments() -> None:
    """Two equities and nine options, as published."""
    assert len(TABLE_7) == 11
    assert sum(1 for row in TABLE_7 if row.kind == "Equity") == 2
    assert sum(1 for row in TABLE_7 if row.is_option) == 9


def test_table_7_total_value() -> None:
    """The Value column sums to the published book value."""
    assert sum(row.value for row in TABLE_7) == pytest.approx(TABLE_7_TOTAL)


def test_spots_are_derived_exactly_from_the_equity_rows() -> None:
    """Equity value equals position times multiplier times spot, with no modelling.

    This is the one inference in the reconstruction that is pure arithmetic,
    and it is self-checking: the recovered spots must sit inside the surrounding
    strike ladder, which runs 670-900.
    """
    assert SPOTS["MITSUBISHI"] == pytest.approx(860.0)
    assert SPOTS["KOMATSU"] == pytest.approx(840.0)

    strikes = [row.strike for row in TABLE_7 if row.strike is not None]
    for spot in SPOTS.values():
        assert min(strikes) <= spot <= max(strikes)


def test_contract_multiplier_reconciles_every_row() -> None:
    """A single multiplier of 1,000 makes all eleven rows coherent.

    The multiplier is inferred, not published, so the evidence for it is
    asserted: with it, every option's price lies between its intrinsic value
    and its underlying's spot - the elementary no-arbitrage band. Any other
    multiplier breaks that for at least one row.
    """
    assert CONTRACT_MULTIPLIER == 1_000.0

    for row in TABLE_7:
        if not row.is_option:
            continue
        spot = SPOTS[row.underlying]
        assert row.strike is not None
        intrinsic = (
            max(spot - row.strike, 0.0)
            if row.kind == "Call"
            else max(row.strike - spot, 0.0)
        )
        assert intrinsic - 1e-9 <= row.unit_price <= spot + 1e-9, (
            f"{row.name} prices outside the no-arbitrage band"
        )


def test_implied_volatilities_invert_and_reprice_exactly(
    implied: dict[str, float]
) -> None:
    """Every option inverts to a plausible volatility and reprices to its mark.

    Repricing to machine precision is the property that matters: it guarantees
    any divergence in the loss distribution comes from the scenarios, never
    from the starting marks.
    """
    assert len(implied) == 9

    for row in TABLE_7:
        if not row.is_option:
            continue
        volatility = implied[row.name]
        assert 0.05 < volatility < 1.5, f"{row.name} implied vol {volatility} implausible"

        assert row.strike is not None and row.maturity_years is not None
        option_type = row.option_type
        assert option_type is not None
        reprice = black_scholes_price(
            SPOTS[row.underlying], row.strike, row.maturity_years,
            RISK_FREE_RATE, volatility, option_type,
        )
        assert reprice == pytest.approx(row.unit_price, rel=1e-9)


def test_put_skew_is_present(implied: dict[str, float]) -> None:
    """The Mitsubishi 800 put carries the highest implied volatility.

    An ordinary put skew rather than a data artefact - worth pinning, because
    if the multiplier or spot inference were wrong this is the row that would
    produce an absurd number first.
    """
    highest = max(implied, key=lambda name: implied[name])
    assert highest == "Mitsubishi Psep30 800"
    assert implied[highest] < 1.0  # elevated, but not nonsensical


def test_base_valuation_reproduces_table_7(implied: dict[str, float]) -> None:
    """The reconstructed book values to Table 7's total, to machine precision."""
    book, _ = build_portfolio()
    value = _price_book(
        book.instruments, book.units, dict(SPOTS), implied,
        _diffusion_volatilities(implied), RISK_FREE_RATE, 0.0,
    )
    assert value == pytest.approx(TABLE_7_TOTAL, abs=1e-3)


def test_portfolio_structure(implied: dict[str, float]) -> None:
    """The book carries optionality and both short and long positions."""
    book, _ = build_portfolio()

    assert len(book) == 11
    assert book.has_optionality
    assert np.any(book.units < 0.0), "the butterfly spread has short legs"
    assert sum(isinstance(i, EuropeanOption) for i in book.instruments) == 9


def test_market_state_carries_both_underlyings() -> None:
    """The base state prices both names and uses the JPY rate."""
    market = base_market_state()
    assert set(market.spots) == {"MITSUBISHI", "KOMATSU"}
    assert market.rate == pytest.approx(RISK_FREE_RATE)
    for name in market.spots:
        assert market.volatility(name) > 0.0


# ==============================================================================
# 2. The reconstruction - qualitative properties only
# ==============================================================================


def test_simulated_losses_have_the_right_shape_and_sign_convention() -> None:
    """Losses are positive-for-loss, one per scenario, and span both signs."""
    losses, base_value = simulate_loss_scenarios(n_scenarios=500, seed=1997)

    assert losses.shape == (500,)
    assert base_value == pytest.approx(TABLE_7_TOTAL, abs=1e-3)
    # eq. (23) is initial value minus later value, so a rally is a negative loss.
    assert losses.min() < 0.0 < losses.max()


def test_normal_distribution_fits_the_loss_distribution_poorly() -> None:
    """The paper's qualitative claim about Figure 1, asserted statistically.

    Section 4 notes that for this book "the normal distribution fits the data
    poorly", which is why Minimum CVaR and Minimum Variance can diverge here.
    That is the property this reconstruction is required to reproduce - not the
    magnitudes, which depend on scenarios the paper never published.
    """
    losses, _ = simulate_loss_scenarios(n_scenarios=1_000, seed=1997)
    test = jarque_bera(losses)
    assert test.pvalue < 0.01, (
        f"normality not rejected (p = {test.pvalue:.3e}); the reconstruction "
        "fails to reproduce the paper's qualitative finding"
    )


def test_cvar_exceeds_var_on_the_reconstructed_book() -> None:
    """Coherence survives full repricing of a non-linear book."""
    losses, _ = simulate_loss_scenarios(n_scenarios=1_000, seed=1997)
    estimate = historical_var_cvar(losses, PAPER_BETA)
    assert estimate.cvar >= estimate.var


def test_reconstruction_is_reproducible() -> None:
    """The same seed gives the same scenarios - required for an audit trail."""
    first, _ = simulate_loss_scenarios(n_scenarios=400, seed=7)
    second, _ = simulate_loss_scenarios(n_scenarios=400, seed=7)
    assert first == pytest.approx(second)


def test_vega_dominates_this_book_s_tail() -> None:
    """A volatility shock moves VaR far more than the spot driver does.

    The finding that justifies defaulting ``vol_shock_sd`` to zero rather than
    guessing it: the book is heavily long volatility, so its tail risk is
    governed by the volatility model. Quoting a single VaR for it without
    stating that assumption would be misleading.
    """
    fixed, _ = simulate_loss_scenarios(n_scenarios=1_000, seed=1997, vol_shock_sd=0.0)
    shocked, _ = simulate_loss_scenarios(n_scenarios=1_000, seed=1997, vol_shock_sd=0.35)

    fixed_var = historical_var_cvar(fixed, PAPER_BETA).var
    shocked_var = historical_var_cvar(shocked, PAPER_BETA).var
    assert shocked_var > 3.0 * fixed_var


def test_published_paper_figures_are_recorded_not_asserted() -> None:
    """The paper's VaR/CVaR are carried for reporting, never matched.

    Documenting the gap in a test keeps a later contributor from "fixing" the
    reconstruction by tuning parameters until these numbers appear, which would
    be curve-fitting to two scalars with no independent support.
    """
    assert PAPER_VAR_95 == 657_816.0
    assert PAPER_CVAR_95 == 2_022_060.0

    losses, _ = simulate_loss_scenarios(n_scenarios=1_000, seed=1997)
    estimate = historical_var_cvar(losses, PAPER_BETA)
    # Same order of magnitude, deliberately not the same number.
    assert 0.1 < estimate.var / PAPER_VAR_95 < 10.0


# ==============================================================================
# 3. Input validation
# ==============================================================================


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_scenarios": 0}, "must be positive"),
        ({"horizon_days": 0.0}, "must be positive"),
        ({"correlation": 1.5}, r"\[-1, 1\]"),
        ({"distribution": "cauchy"}, "must be 'normal' or 'student_t'"),
        ({"dof": 1.5}, "must exceed 2"),
        ({"vol_shock_sd": -0.1}, "non-negative"),
    ],
)
def test_simulation_rejects_bad_inputs(kwargs: dict, message: str) -> None:
    """Invalid scenario parameters fail loudly rather than producing nonsense."""
    with pytest.raises(ValueError, match=message):
        simulate_loss_scenarios(n_scenarios=kwargs.pop("n_scenarios", 100), **kwargs)


def test_option_rows_map_to_european_options() -> None:
    """Calls and puts carry the right right."""
    calls = [r for r in TABLE_7 if r.kind == "Call"]
    puts = [r for r in TABLE_7 if r.kind == "Put"]
    assert len(calls) == 6 and len(puts) == 3
    assert all(r.option_type is OptionType.CALL for r in calls)
    assert all(r.option_type is OptionType.PUT for r in puts)
    assert all(r.option_type is None for r in TABLE_7 if r.kind == "Equity")
