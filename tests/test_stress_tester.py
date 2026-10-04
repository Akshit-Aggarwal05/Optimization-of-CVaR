"""Tests for the non-linear stress-testing engine.

Layers:

1.  **Black-Scholes-Merton** - every Greek against a finite difference of the
    pricer, plus put-call parity and the structural identities that relate call
    and put sensitivities.
2.  **Scenarios and market state** - the factor-to-instrument plumbing.
3.  **Instruments and portfolio** - bond convexity, Greek aggregation, and the
    weights-to-positions bridge from Steps 1 and 2.
4.  **Taylor convergence** - the decisive test.  Delta error must fall as
    ``O(h^2)`` and delta-gamma error as ``O(h^3)`` as the shock shrinks.  Any
    mistake in a Greek, a sign, or the decomposition destroys those rates, so
    the orders of convergence pin the whole expansion down far more tightly
    than any single-point comparison could.
5.  **The CVaR bridge** - stressed VaR/CVaR for an optimised portfolio.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.paper_datasets import MEAN_RETURNS, sample_normal_returns
from src.models.risk_metrics import historical_var_cvar
from src.models.stress_tester import (
    HISTORICAL_SCENARIOS,
    BondInstrument,
    EquityInstrument,
    EuropeanOption,
    Greeks,
    MarketState,
    OptionType,
    RevaluationMethod,
    RiskFactor,
    ShockScenario,
    StressPortfolio,
    StressTester,
    UnderlyingSpec,
    black_scholes_greeks,
    black_scholes_price,
    default_underlyings,
    get_scenario,
    list_scenarios,
    stress_return_scenarios,
    stressed_risk_estimate,
)

# Base option parameters used across the pricing tests.
SPOT, STRIKE, MATURITY, RATE, VOL, DIVIDEND = 100.0, 95.0, 0.75, 0.03, 0.22, 0.015


@pytest.fixture(scope="module")
def underlyings() -> dict[str, UnderlyingSpec]:
    """Factor specs with the bond's rate response carried by the instrument."""
    return default_underlyings(bond_duration_in_instrument=True)


@pytest.fixture(scope="module")
def base_market() -> MarketState:
    """A plausible starting market for the paper's three sleeves."""
    return MarketState(
        spots={"SPX": 4500.0, "GOVT": 100.0, "SMALL": 2000.0},
        volatilities={"SPX": 0.18, "GOVT": 0.06, "SMALL": 0.24},
        rate=0.04,
        dividend_yields={"SPX": 0.015, "SMALL": 0.010},
    )


@pytest.fixture(scope="module")
def collar_book() -> StressPortfolio:
    """Equity and bond sleeves overlaid with a protective put collar.

    Long puts finance-hedged by short calls: the standard tail-hedge structure,
    and one whose net gamma is genuinely ambiguous, so it exercises the
    non-linear machinery rather than a trivially convex payoff.
    """
    nav = 100e6
    instruments = [
        EquityInstrument("S&P sleeve", "SPX"),
        BondInstrument("Long govt", "GOVT", duration=10.0, convexity=140.0, base_rate=0.04),
        EquityInstrument("Small cap", "SMALL"),
        EuropeanOption("SPX 90% put", "SPX", 4050.0, 1.0, OptionType.PUT),
        EuropeanOption("SPX 110% call", "SPX", 4950.0, 1.0, OptionType.CALL),
    ]
    units = np.array(
        [
            0.45 * nav / 4500.0,
            0.30 * nav / 100.0,
            0.25 * nav / 2000.0,
            0.60 * 0.45 * nav / 4500.0,
            -0.60 * 0.45 * nav / 4500.0,
        ]
    )
    return StressPortfolio(instruments, units)


@pytest.fixture(scope="module")
def tester(
    collar_book: StressPortfolio,
    base_market: MarketState,
    underlyings: dict[str, UnderlyingSpec],
) -> StressTester:
    """Engine wired to the collar book."""
    return StressTester(collar_book, base_market, underlyings)


def _spot_shock(size: float) -> ShockScenario:
    """A pure equity shock with no vol move and no ageing."""
    return ShockScenario("spot_only", {RiskFactor.EQUITY: -size}, horizon_days=0)


# ==============================================================================
# 1. Black-Scholes-Merton
# ==============================================================================


def test_put_call_parity() -> None:
    """``C - P = S e^{-qT} - K e^{-rT}`` must hold to machine precision."""
    call = black_scholes_price(SPOT, STRIKE, MATURITY, RATE, VOL, OptionType.CALL, DIVIDEND)
    put = black_scholes_price(SPOT, STRIKE, MATURITY, RATE, VOL, OptionType.PUT, DIVIDEND)
    forward = SPOT * np.exp(-DIVIDEND * MATURITY) - STRIKE * np.exp(-RATE * MATURITY)
    assert call - put == pytest.approx(forward, abs=1e-12)


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_greeks_match_finite_differences(option_type: OptionType) -> None:
    """Every analytic Greek is checked against a bump of the pricer itself.

    An analytic Greek that disagrees with its own pricer is the classic silent
    error in a risk system: positions still price correctly, and only the risk
    report is wrong.
    """
    # First derivatives use a small step; the second derivative needs a much
    # larger one. A central second difference carries roundoff of order
    # 4*eps*f/h^2, which at h=1e-4 swamps gamma entirely - the optimal step
    # scales as eps^(1/4) rather than the eps^(1/3) that suits first
    # derivatives. This is a property of the test's numerical method, not of
    # the analytic Greeks.
    step = 1e-4
    second_order_step = 1e-2

    def price(spot=SPOT, strike=STRIKE, maturity=MATURITY, rate=RATE, vol=VOL) -> float:
        return black_scholes_price(
            spot, strike, maturity, rate, vol, option_type, DIVIDEND
        )

    analytic = black_scholes_greeks(
        SPOT, STRIKE, MATURITY, RATE, VOL, option_type, DIVIDEND
    )
    base = price()

    assert analytic.delta == pytest.approx(
        (price(spot=SPOT + step) - price(spot=SPOT - step)) / (2 * step), rel=1e-6
    )
    assert analytic.gamma == pytest.approx(
        (
            price(spot=SPOT + second_order_step)
            - 2 * base
            + price(spot=SPOT - second_order_step)
        )
        / second_order_step**2,
        rel=1e-6,
    )
    assert analytic.vega == pytest.approx(
        (price(vol=VOL + step) - price(vol=VOL - step)) / (2 * step), rel=1e-6
    )
    # Theta is the derivative with respect to *calendar* time, so it is minus
    # the derivative with respect to time-to-maturity.
    assert analytic.theta == pytest.approx(
        -(price(maturity=MATURITY + step) - price(maturity=MATURITY - step)) / (2 * step),
        rel=1e-5,
    )
    assert analytic.rho == pytest.approx(
        (price(rate=RATE + step) - price(rate=RATE - step)) / (2 * step), rel=1e-6
    )


def test_call_and_put_share_gamma_and_vega() -> None:
    """Put-call parity forces it: the two differ by a forward, which is linear."""
    call = black_scholes_greeks(SPOT, STRIKE, MATURITY, RATE, VOL, OptionType.CALL, DIVIDEND)
    put = black_scholes_greeks(SPOT, STRIKE, MATURITY, RATE, VOL, OptionType.PUT, DIVIDEND)
    assert call.gamma == pytest.approx(put.gamma, rel=1e-14)
    assert call.vega == pytest.approx(put.vega, rel=1e-14)
    # Differentiating parity in spot: delta_call - delta_put = e^{-qT}.
    assert call.delta - put.delta == pytest.approx(np.exp(-DIVIDEND * MATURITY), abs=1e-12)


@pytest.mark.parametrize(
    "maturity, volatility", [(0.0, VOL), (MATURITY, 0.0), (0.0, 0.0)]
)
def test_degenerate_inputs_fall_back_to_intrinsic(
    maturity: float, volatility: float
) -> None:
    """At expiry or with no volatility the option is worth intrinsic value.

    Stress scenarios routinely push options to these boundaries, so the pricer
    must return a number rather than raise.
    """
    call = black_scholes_price(
        SPOT, STRIKE, maturity, RATE, volatility, OptionType.CALL, DIVIDEND
    )
    assert call == pytest.approx(max(SPOT - STRIKE, 0.0) * np.exp(-RATE * maturity))

    greeks = black_scholes_greeks(
        SPOT, STRIKE, maturity, RATE, volatility, OptionType.CALL, DIVIDEND
    )
    assert greeks.delta == pytest.approx(1.0)  # in the money
    assert greeks.gamma == 0.0  # no curvature left


def test_option_price_is_monotone_and_above_intrinsic() -> None:
    """Elementary no-arbitrage properties of a call."""
    spots = np.linspace(50.0, 150.0, 40)
    prices = np.array(
        [
            black_scholes_price(s, STRIKE, MATURITY, RATE, VOL, OptionType.CALL, DIVIDEND)
            for s in spots
        ]
    )
    assert np.all(np.diff(prices) > 0.0)  # increasing in spot
    assert np.all(prices >= 0.0)
    discounted_intrinsic = np.maximum(
        spots * np.exp(-DIVIDEND * MATURITY) - STRIKE * np.exp(-RATE * MATURITY), 0.0
    )
    assert np.all(prices >= discounted_intrinsic - 1e-9)


def test_gamma_peaks_near_the_money() -> None:
    """Curvature concentrates at the strike - the reason gamma risk is local."""
    def gamma_at(spot: float) -> float:
        return black_scholes_greeks(
            spot, STRIKE, MATURITY, RATE, VOL, OptionType.CALL, DIVIDEND
        ).gamma

    assert gamma_at(STRIKE) > gamma_at(STRIKE * 0.6)
    assert gamma_at(STRIKE) > gamma_at(STRIKE * 1.6)


# ==============================================================================
# 2. Scenario library and market state
# ==============================================================================


def test_library_contains_the_required_scenarios() -> None:
    """2008 and 2020 are the two the engine is specified to cover."""
    assert "gfc_2008" in HISTORICAL_SCENARIOS
    assert "covid_2020" in HISTORICAL_SCENARIOS
    assert set(list_scenarios()) == set(HISTORICAL_SCENARIOS)


def test_every_scenario_is_documented_and_directionally_sane() -> None:
    """Equity falls, volatility rises, and each carries its provenance."""
    for name, scenario in HISTORICAL_SCENARIOS.items():
        assert scenario.name == name
        assert scenario.description and scenario.window and scenario.source
        assert scenario.horizon_days >= 1
        assert scenario.get(RiskFactor.EQUITY) < 0.0
        assert scenario.get(RiskFactor.VOLATILITY) > 0.0


def test_2022_is_the_scenario_where_the_bond_hedge_fails() -> None:
    """Rates rose with equities falling - the case 2008 and 2020 do not cover.

    A stress library containing only flight-to-quality crises would certify a
    duration hedge that in fact failed badly in 2022, so the sign of this shock
    is asserted deliberately.
    """
    crisis = get_scenario("inflation_shock_2022")
    assert crisis.get(RiskFactor.RATES) > 0.0
    for name in ("gfc_2008", "covid_2020", "black_monday_1987", "euro_crisis_2011"):
        assert get_scenario(name).get(RiskFactor.RATES) < 0.0


def test_unknown_scenario_names_are_rejected_helpfully() -> None:
    """A typo must fail loudly, listing what is available."""
    with pytest.raises(KeyError, match="Available"):
        get_scenario("gfc_2007")


def test_scenario_scaling_is_uniform() -> None:
    """Severity scaling multiplies every factor and leaves the rest alone."""
    base = get_scenario("gfc_2008")
    doubled = base.scaled(2.0)
    for factor, move in base.shocks.items():
        assert doubled.get(factor) == pytest.approx(2.0 * move)
    assert doubled.horizon_days == base.horizon_days


def test_market_state_applies_factor_betas(
    base_market: MarketState, underlyings: dict[str, UnderlyingSpec]
) -> None:
    """Spots, vols, rates and the clock all move as specified."""
    scenario = get_scenario("gfc_2008")
    stressed = base_market.apply(scenario, underlyings)

    spx = underlyings["SPX"]
    expected_return = (
        1.00 * scenario.get(RiskFactor.EQUITY) + (-1.50) * scenario.get(RiskFactor.RATES)
    )
    assert spx.spot_return(scenario) == pytest.approx(expected_return)
    assert stressed.spot("SPX") == pytest.approx(4500.0 * (1.0 + expected_return))
    # Vol beta 1.0 against a +200% relative shock trebles implied volatility.
    assert stressed.volatility("SPX") == pytest.approx(0.18 * 3.0)
    assert stressed.rate == pytest.approx(0.04 + scenario.get(RiskFactor.RATES))
    assert stressed.time == pytest.approx(scenario.horizon_days / 365.0)


def test_market_state_can_skip_ageing(
    base_market: MarketState, underlyings: dict[str, UnderlyingSpec]
) -> None:
    """An instantaneous jump leaves the clock untouched."""
    stressed = base_market.apply(get_scenario("gfc_2008"), underlyings, age_forward=False)
    assert stressed.time == pytest.approx(0.0)


def test_spot_cannot_go_negative(
    base_market: MarketState, underlyings: dict[str, UnderlyingSpec]
) -> None:
    """A shock beyond -100% floors the spot at zero rather than inverting it."""
    catastrophe = ShockScenario("wipeout", {RiskFactor.EQUITY: -2.0})
    stressed = base_market.apply(catastrophe, underlyings)
    assert stressed.spot("SPX") == 0.0
    assert stressed.spot("SMALL") == 0.0


def test_missing_underlying_specification_is_rejected(base_market: MarketState) -> None:
    """An unspecified underlying would pass through a scenario unshocked."""
    with pytest.raises(KeyError, match="No UnderlyingSpec"):
        base_market.apply(get_scenario("gfc_2008"), {"SPX": UnderlyingSpec("SPX")})


def test_missing_spot_is_rejected(base_market: MarketState) -> None:
    """Defaulting an absent spot to zero would silently void a whole sleeve."""
    with pytest.raises(KeyError, match="No spot for underlying"):
        base_market.spot("NIKKEI")


# ==============================================================================
# 3. Instruments, Greeks and the portfolio
# ==============================================================================


def test_greeks_are_additive_and_scalable() -> None:
    """Aggregation across positions is a plain vector-space operation."""
    a = Greeks(delta=0.5, gamma=0.01, vega=10.0)
    b = Greeks(delta=-0.2, gamma=0.02, rho=3.0)
    assert (a + b).delta == pytest.approx(0.3)
    assert (a + b).gamma == pytest.approx(0.03)
    assert (a * -2.0).delta == pytest.approx(-1.0)
    assert (3.0 * a).vega == pytest.approx(30.0)


def test_equity_is_linear(base_market: MarketState) -> None:
    """Unit delta, zero gamma: a delta stress test is exact for cash equity."""
    equity = EquityInstrument("S&P", "SPX")
    assert equity.value(base_market) == pytest.approx(4500.0)
    assert equity.greeks(base_market).delta == pytest.approx(1.0)
    assert equity.greeks(base_market).gamma == 0.0
    assert equity.is_linear


def test_bond_convexity_softens_a_yield_rise(base_market: MarketState) -> None:
    """Positive convexity means duration alone overstates the loss.

    For the +232bp move of the 2022 scenario the convexity term is worth
    several per cent of the sleeve, which is the bond-market analogue of gamma
    and the reason duration-only stress reports are conservative in one
    direction and wrong in the other.
    """
    bond = BondInstrument("Long govt", "GOVT", duration=10.0, convexity=140.0, base_rate=0.04)
    shift = 0.0232
    shocked = MarketState(spots=base_market.spots, rate=0.04 + shift)

    actual = bond.value(shocked) / bond.value(base_market) - 1.0
    duration_only = -10.0 * shift
    convexity_gain = 0.5 * 140.0 * shift**2

    assert actual == pytest.approx(duration_only + convexity_gain)
    assert actual > duration_only  # convexity is a gain, always
    assert convexity_gain > 0.03  # and a material one at this size of move


def test_bond_greeks_carry_duration_and_convexity(base_market: MarketState) -> None:
    """``rho`` is minus dollar duration; ``rate_gamma`` is dollar convexity."""
    bond = BondInstrument("Govt", "GOVT", duration=10.0, convexity=140.0, base_rate=0.04)
    greeks = bond.greeks(base_market)
    assert greeks.rho == pytest.approx(-10.0 * 100.0)
    assert greeks.rate_gamma == pytest.approx(140.0 * 100.0)
    assert not bond.is_linear  # convexity makes it non-linear in yield


def test_bond_validates_its_risk_measures() -> None:
    """Negative duration or convexity is not a vanilla bond."""
    with pytest.raises(ValueError, match="duration"):
        BondInstrument("bad", "GOVT", duration=-1.0)
    with pytest.raises(ValueError, match="convexity"):
        BondInstrument("bad", "GOVT", duration=5.0, convexity=-10.0)


def test_option_validates_its_terms() -> None:
    """Contract terms must be economically meaningful."""
    with pytest.raises(ValueError, match="strike"):
        EuropeanOption("bad", "SPX", 0.0, 1.0, OptionType.CALL)
    with pytest.raises(ValueError, match="maturity"):
        EuropeanOption("bad", "SPX", 100.0, -0.5, OptionType.CALL)
    with pytest.raises(ValueError, match="contract_size"):
        EuropeanOption("bad", "SPX", 100.0, 1.0, OptionType.CALL, contract_size=0.0)


def test_option_ages_with_the_market_clock(base_market: MarketState) -> None:
    """Advancing time shortens the option, and a long option loses value by it."""
    option = EuropeanOption("SPX put", "SPX", 4500.0, 1.0, OptionType.PUT)
    later = MarketState(
        spots=base_market.spots,
        volatilities=base_market.volatilities,
        rate=base_market.rate,
        dividend_yields=base_market.dividend_yields,
        time=0.5,
    )
    assert option.remaining_maturity(later) == pytest.approx(0.5)
    assert option.value(later) < option.value(base_market)


def test_portfolio_from_weights_allocates_the_full_value(
    base_market: MarketState,
) -> None:
    """The bridge from an optimised weight vector to position sizes."""
    instruments = [
        EquityInstrument("S&P", "SPX"),
        EquityInstrument("Govt", "GOVT"),
        EquityInstrument("Small", "SMALL"),
    ]
    weights = np.array([0.452013, 0.115573, 0.432414])  # the paper's Table 3
    book = StressPortfolio.from_weights(weights, instruments, base_market, total_value=1e6)

    assert book.value(base_market) == pytest.approx(1e6)
    # Each sleeve holds its weighted share of the total.
    assert book.instrument_values(base_market) == pytest.approx(weights * 1e6)


def test_portfolio_from_weights_validates_inputs(base_market: MarketState) -> None:
    """Length mismatches and unpriceable instruments are rejected."""
    instruments = [EquityInstrument("S&P", "SPX")]
    with pytest.raises(ValueError, match="entries but 1 instruments"):
        StressPortfolio.from_weights([0.5, 0.5], instruments, base_market)

    zero_market = MarketState(spots={"SPX": 0.0})
    with pytest.raises(ValueError, match="non-positive price"):
        StressPortfolio.from_weights([1.0], instruments, zero_market)


def test_cash_greeks_of_a_pure_equity_book(base_market: MarketState) -> None:
    """Delta cash of unhedged equity is exactly its market value."""
    book = StressPortfolio([EquityInstrument("S&P", "SPX")], np.array([1000.0]))
    greeks = book.cash_greeks(base_market)
    assert greeks["delta_cash"] == pytest.approx(1000.0 * 4500.0)
    assert greeks["gamma_cash_1pct"] == 0.0
    assert greeks["dv01"] == 0.0


def test_dv01_is_negative_for_a_long_bond(base_market: MarketState) -> None:
    """A long bond loses value when yields rise, so its DV01 is negative."""
    book = StressPortfolio(
        [BondInstrument("Govt", "GOVT", duration=10.0, base_rate=0.04)],
        np.array([1000.0]),
    )
    assert book.cash_greeks(base_market)["dv01"] < 0.0


def test_optionality_is_detected(collar_book: StressPortfolio) -> None:
    """The book knows whether any position carries curvature."""
    assert collar_book.has_optionality
    linear = StressPortfolio([EquityInstrument("S&P", "SPX")], np.array([1.0]))
    assert not linear.has_optionality


# ==============================================================================
# 4. Stress testing: exactness, convergence, and the linearisation error
# ==============================================================================


def test_linear_book_has_zero_approximation_error(
    base_market: MarketState, underlyings: dict[str, UnderlyingSpec]
) -> None:
    """For a book with no curvature the delta approximation is *exact*.

    This is the control: any non-zero error here would mean the decomposition
    itself is broken, and every reported curvature effect elsewhere would be an
    artefact.
    """
    book = StressPortfolio(
        [EquityInstrument("S&P", "SPX"), EquityInstrument("Small", "SMALL")],
        np.array([1000.0, 500.0]),
    )
    engine = StressTester(book, base_market, underlyings)
    for name in list_scenarios():
        result = engine.run(get_scenario(name))
        assert result.pnl_delta == pytest.approx(result.pnl_full, rel=1e-12)
        assert result.delta_error == pytest.approx(0.0, abs=1e-12)
        assert result.gamma_contribution == pytest.approx(0.0, abs=1e-9)


def test_taylor_expansion_converges_at_the_right_orders(tester: StressTester) -> None:
    """Delta error is ``O(h^2)`` and delta-gamma error is ``O(h^3)``.

    The single most informative test in this module.  Halving the shock must
    cut the delta error by about four and the delta-gamma error by about eight.
    A wrong sign, a mis-scaled Greek, or a term dropped from the decomposition
    would all break these rates, so the orders of convergence verify the entire
    expansion at once - far more sharply than comparing P&L at one shock size.
    """
    errors_delta, errors_delta_gamma = [], []
    for size in (0.02, 0.01, 0.005):
        result = tester.run(_spot_shock(size), age_forward=False)
        errors_delta.append(abs(result.pnl_delta - result.pnl_full))
        errors_delta_gamma.append(abs(result.pnl_delta_gamma - result.pnl_full))

    for coarse, fine in zip(errors_delta, errors_delta[1:]):
        assert 3.5 < coarse / fine < 4.5  # second order
    for coarse, fine in zip(errors_delta_gamma, errors_delta_gamma[1:]):
        assert 7.0 < coarse / fine < 9.0  # third order


def test_delta_gamma_beats_delta_for_small_shocks(tester: StressTester) -> None:
    """Inside its radius of validity the second-order term is a strict gain."""
    for size in (0.005, 0.01, 0.02, 0.05):
        result = tester.run(_spot_shock(size), age_forward=False)
        assert abs(result.pnl_delta_gamma - result.pnl_full) < abs(
            result.pnl_delta - result.pnl_full
        )


def test_linearisation_error_is_material_in_a_real_crisis(tester: StressTester) -> None:
    """At crisis scale the approximations are wrong by an amount that matters.

    This is the finding the module exists to produce: a delta-only stress
    report on an option book misstates a 2008-sized loss by roughly a tenth of
    the loss itself, which on this book is millions of currency units.
    """
    result = tester.run(get_scenario("gfc_2008"))
    assert abs(result.delta_error) > 0.05
    assert abs(result.pnl_delta - result.pnl_full) > 1e6


def test_full_revaluation_is_the_reference(tester: StressTester) -> None:
    """``pnl_full`` is the difference of two independently computed valuations."""
    result = tester.run(get_scenario("covid_2020"))
    assert result.stressed_value - result.base_value == pytest.approx(result.pnl_full)
    assert result.instrument_pnl.sum() == pytest.approx(result.pnl_full, rel=1e-12)
    assert result.pnl(RevaluationMethod.FULL) == result.pnl_full
    assert result.pnl(RevaluationMethod.DELTA) == result.pnl_delta


def test_protective_put_gains_in_a_crash(
    base_market: MarketState, underlyings: dict[str, UnderlyingSpec]
) -> None:
    """A long put is worth far more after a -46% move - the point of holding it.

    It also demonstrates the convexity that the linear approximation misses:
    the put's exact gain exceeds what its delta alone predicts, because delta
    steepens towards -1 as the option goes into the money.
    """
    put = EuropeanOption("SPX put", "SPX", 4050.0, 1.0, OptionType.PUT)
    book = StressPortfolio([put], np.array([1000.0]))
    engine = StressTester(book, base_market, underlyings)

    result = engine.run(_spot_shock(0.46), age_forward=False)
    assert result.pnl_full > 0.0
    assert result.pnl_full > result.pnl_delta  # convexity works in the holder's favour


def test_short_gamma_book_loses_more_than_delta_predicts(
    base_market: MarketState, underlyings: dict[str, UnderlyingSpec]
) -> None:
    """The mirror image: a short option position is concave and bleeds."""
    call = EuropeanOption("SPX call", "SPX", 4500.0, 1.0, OptionType.CALL)
    book = StressPortfolio([call], np.array([-1000.0]))  # short
    engine = StressTester(book, base_market, underlyings)

    # A rally hurts a short call by more than its delta suggests.
    rally = ShockScenario("rally", {RiskFactor.EQUITY: +0.25}, horizon_days=0)
    result = engine.run(rally, age_forward=False)
    assert result.pnl_full < result.pnl_delta
    assert result.gamma_contribution < 0.0


def test_delta_drifts_during_the_shock(tester: StressTester) -> None:
    """Cash delta after the move differs from before - the practical cost of gamma.

    A hedge set at the base state is stale by the time the move is over, which
    is why gamma matters operationally and not only as an accounting term.
    """
    result = tester.run(get_scenario("gfc_2008"))
    assert result.stressed_greeks["delta_cash"] != pytest.approx(
        result.base_greeks["delta_cash"], rel=0.05
    )


def test_ageing_costs_a_long_option_book_time_value(
    base_market: MarketState, underlyings: dict[str, UnderlyingSpec]
) -> None:
    """A six-month crisis also burns six months of premium.

    Ignoring the horizon flatters a long-option hedge, since it books the
    protection's payoff without its carry.
    """
    put = EuropeanOption("SPX put", "SPX", 4050.0, 2.0, OptionType.PUT)
    book = StressPortfolio([put], np.array([1000.0]))
    engine = StressTester(book, base_market, underlyings)

    scenario = get_scenario("gfc_2008")
    assert engine.run(scenario, age_forward=True).pnl_full < engine.run(
        scenario, age_forward=False
    ).pnl_full


def test_severity_scaling_reveals_convexity(tester: StressTester) -> None:
    """Doubling a shock more than doubles the loss on a book with curvature."""
    half = tester.run(_spot_shock(0.10), age_forward=False).pnl_full
    full = tester.run(_spot_shock(0.20), age_forward=False).pnl_full
    # Purely linear would give exactly 2x; the deviation is the convexity.
    assert full != pytest.approx(2.0 * half, rel=1e-4)


def test_worst_case_selects_the_largest_loss(tester: StressTester) -> None:
    """The number that goes to the risk committee."""
    worst = tester.worst_case()
    assert worst.pnl_full == min(result.pnl_full for result in tester.run_all())
    assert worst.pnl_full < 0.0


def test_run_all_covers_the_whole_library(tester: StressTester) -> None:
    """Defaulting to every scenario prevents a crisis being skipped by omission."""
    results = tester.run_all()
    assert len(results) == len(HISTORICAL_SCENARIOS)
    assert {result.scenario.name for result in results} == set(HISTORICAL_SCENARIOS)


def test_results_tabulate(tester: StressTester) -> None:
    """The report frame carries P&L by method alongside the errors."""
    frame = tester.to_frame()
    assert len(frame) == len(HISTORICAL_SCENARIOS)
    assert {"pnl_full", "pnl_delta", "delta_error", "gamma_contribution"} <= set(
        frame.columns
    )


def test_bond_rate_exposure_cannot_be_double_counted(base_market: MarketState) -> None:
    """Duration *and* a rates beta would apply the yield move twice."""
    book = StressPortfolio(
        [BondInstrument("Govt", "GOVT", duration=10.0, base_rate=0.04)], np.array([1.0])
    )
    with pytest.raises(ValueError, match="applied twice"):
        StressTester(book, base_market, default_underlyings())  # GOVT keeps its beta


def test_bond_base_rate_must_match_the_market(
    base_market: MarketState, underlyings: dict[str, UnderlyingSpec]
) -> None:
    """A mismatched base rate biases every result before any shock is applied."""
    book = StressPortfolio(
        [BondInstrument("Govt", "GOVT", duration=10.0, base_rate=0.02)], np.array([1.0])
    )
    with pytest.raises(ValueError, match="base_rate"):
        StressTester(book, base_market, underlyings)


# ==============================================================================
# 5. Bridge back to the CVaR engine
# ==============================================================================


def test_stress_shifts_the_distribution_by_the_factor_return() -> None:
    """The mean of each column moves by exactly that asset's implied return."""
    returns = sample_normal_returns(2_048, engine="sobol", seed=5)
    specs = list(default_underlyings().values())
    scenario = get_scenario("gfc_2008")

    stressed = stress_return_scenarios(returns, scenario, specs)
    shifts = np.array([spec.spot_return(scenario) for spec in specs])
    assert stressed.mean(axis=0) == pytest.approx(returns.mean(axis=0) + shifts, abs=1e-12)


def test_stress_widens_the_distribution() -> None:
    """A crisis fattens the tails as well as moving the centre.

    Shifting alone would produce a portfolio that looks merely poorer; the
    widening is what makes it genuinely riskier, and it is calibrated by the
    same episode's volatility shock.
    """
    returns = sample_normal_returns(2_048, engine="sobol", seed=5)
    specs = list(default_underlyings().values())
    scenario = get_scenario("gfc_2008")

    stressed = stress_return_scenarios(returns, scenario, specs)
    multiplier = 1.0 + scenario.get(RiskFactor.VOLATILITY)
    assert stressed.std(axis=0) == pytest.approx(returns.std(axis=0) * multiplier, rel=1e-12)


def test_stress_rejects_a_specification_mismatch() -> None:
    """One specification per return column, or the mapping is ambiguous."""
    returns = sample_normal_returns(256, engine="sobol", seed=5)
    with pytest.raises(ValueError, match="must correspond"):
        stress_return_scenarios(returns, get_scenario("gfc_2008"), [UnderlyingSpec("SPX")])


def test_stressed_cvar_exceeds_base_cvar() -> None:
    """The optimised portfolio's tail risk deteriorates under every crisis.

    This closes the loop: weights produced in Steps 1-2 are re-scored on a
    stressed distribution using the Step 1 estimator, so a stress scenario
    re-enters the risk measurement rather than living in a separate report.
    """
    returns = sample_normal_returns(4_096, engine="sobol", seed=9)
    weights = np.array([0.452013, 0.115573, 0.432414])  # the paper's Table 3
    specs = list(default_underlyings().values())

    for name in list_scenarios():
        base, stressed = stressed_risk_estimate(
            weights, returns, get_scenario(name), specs, beta=0.95
        )
        assert stressed.cvar > base.cvar
        assert stressed.var > base.var
        assert stressed.cvar >= stressed.var  # coherence survives the transformation


def test_stressed_estimate_matches_a_manual_recomputation() -> None:
    """The bridge is a thin wrapper: it must agree with doing it by hand."""
    returns = sample_normal_returns(2_048, engine="sobol", seed=11)
    weights = np.array([0.4, 0.2, 0.4])
    specs = list(default_underlyings().values())
    scenario = get_scenario("covid_2020")

    _, stressed = stressed_risk_estimate(weights, returns, scenario, specs, beta=0.99)
    manual = historical_var_cvar(
        -(stress_return_scenarios(returns, scenario, specs) @ weights), 0.99
    )
    assert stressed.cvar == pytest.approx(manual.cvar, rel=1e-12)
    assert stressed.var == pytest.approx(manual.var, rel=1e-12)


def test_bond_sleeve_helps_in_2008_and_hurts_in_2022() -> None:
    """The diversification story is scenario-dependent, and the library shows it.

    A duration hedge that is validated only against flight-to-quality crises
    looks unambiguously good; tested against 2022 it does not. Comparing an
    all-bond portfolio's stressed loss across the two scenarios makes the point
    quantitatively.
    """
    returns = sample_normal_returns(2_048, engine="sobol", seed=13)
    all_bonds = np.array([0.0, 1.0, 0.0])
    specs = list(default_underlyings().values())

    _, gfc = stressed_risk_estimate(all_bonds, returns, get_scenario("gfc_2008"), specs)
    _, inflation = stressed_risk_estimate(
        all_bonds, returns, get_scenario("inflation_shock_2022"), specs
    )
    # Yields fell in 2008 (the sleeve gains) and rose in 2022 (it loses).
    assert gfc.mean_loss < 0.0
    assert inflation.mean_loss > 0.0
