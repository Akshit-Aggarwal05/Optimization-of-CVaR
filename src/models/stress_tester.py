"""Non-linear stress testing under historical macroeconomic shock scenarios.

Steps 1 and 2 measure and optimise risk against a *sampled* loss distribution.
That distribution is only as good as the sample: it contains the tail events
that happened to be drawn, and nothing else.  Stress testing asks the
complementary question - what does this book lose in a *named* crisis, whether
or not anything like it appears in the scenario set?

Scope of this module
--------------------
1.  A library of calibrated historical shock scenarios (2008, 2020, 1987, 2011,
    2022), each expressed as moves in macroeconomic **risk factors** rather
    than in instrument returns, so a single scenario applies to any book.
2.  Analytic Black-Scholes-Merton pricing and Greeks, plus duration/convexity
    for bonds, so a portfolio containing options can be revalued.
3.  Three revaluation methods run side by side - delta-only, delta-gamma, and
    full repricing - so the **error committed by linearising** is reported as a
    number rather than assumed away.
4.  A bridge back to Steps 1-2: applying a scenario to the scenario matrix and
    recomputing stressed VaR/CVaR for an optimised portfolio.

Why non-linearity is the whole point
------------------------------------
For a linear book, first-order sensitivity is exact and stress testing is
arithmetic.  For a book with options it is not, and the error grows with the
*square* of the shock - precisely where stress testing operates.  A -34% equity
move is roughly a five standard-deviation event; the delta approximation is
calibrated at the current spot and has no information about curvature that far
away.  Long options gain from convexity (delta-only *understates* their value);
short options bleed from it (delta-only *overstates*).  Either way the sign of
the error is systematic, not random, so it does not diversify across a book.

This is the same phenomenon Rockafellar & Uryasev meet in Section 4 of the
paper.  Their NIKKEI butterfly-spread portfolio has a loss distribution that
"the normal distribution fits poorly", and the reason is that its payoff is a
non-linear function of the underlying.  Their response - scenario-based CVaR
rather than a parametric formula - is the same one taken here: when the
mapping from risk factors to P&L is non-linear, reprice, do not approximate.

Calibration note
----------------
The shock magnitudes in :data:`HISTORICAL_SCENARIOS` are approximate
calibrations taken from published index levels over the stated windows.  They
are realistic and internally consistent, and they are adequate for model
development and for the tests in this repository.  They are **not** a
substitute for a desk's own calibration: before production use, recalibrate
each factor against the firm's own market data and risk-factor definitions,
and have the result signed off by model validation.  Every scenario is a plain
dataclass precisely so that it can be replaced.

References
----------
Rockafellar, R.T. and Uryasev, S. (2000).  *Optimization of Conditional
Value-at-Risk*.  Journal of Risk, 2(3), 21-41.  Section 4 for the option book.

Black, F. and Scholes, M. (1973).  *The pricing of options and corporate
liabilities*.  Journal of Political Economy, 81(3), 637-654.

Merton, R.C. (1973).  *Theory of rational option pricing*.  Bell Journal of
Economics and Management Science, 4(1), 141-183.  (Continuous dividend yield.)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Final, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from src.models.risk_metrics import (
    RiskEstimate,
    _as_1d,
    _as_2d,
    _validate_beta,
    historical_var_cvar,
)

__all__ = [
    "RiskFactor",
    "ShockScenario",
    "HISTORICAL_SCENARIOS",
    "get_scenario",
    "list_scenarios",
    "Greeks",
    "UnderlyingSpec",
    "MarketState",
    "Instrument",
    "EquityInstrument",
    "BondInstrument",
    "EuropeanOption",
    "OptionType",
    "black_scholes_price",
    "black_scholes_greeks",
    "StressPortfolio",
    "RevaluationMethod",
    "StressResult",
    "StressTester",
    "default_underlyings",
    "stress_return_scenarios",
    "stressed_risk_estimate",
]

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

#: Trading days per year, used to convert horizons to option time-to-maturity.
TRADING_DAYS: Final[float] = 252.0


# ==============================================================================
# 1. Risk factors and the historical scenario library
# ==============================================================================


class RiskFactor(str, Enum):
    """Macroeconomic risk factors a shock scenario can move.

    Units matter and are fixed by convention here, because a scenario library
    where each factor carries its own implicit scaling is the most reliable
    source of silent errors in a stress framework.

    Attributes
    ----------
    EQUITY
        Broad equity index **return**, as a decimal.  ``-0.46`` is a 46% fall.
    RATES
        Parallel shift in the government yield curve, as a decimal **level**
        change.  ``+0.0232`` is +232 basis points.
    CREDIT
        Change in credit spread, as a decimal level change.  ``+0.04`` is +400
        basis points of widening.
    VOLATILITY
        **Relative** change in implied volatility.  ``+2.0`` means implied vol
        triples (a 20 vol goes to 60).  Relative rather than absolute because
        vol shocks scale with the starting level.
    FX
        Return on the relevant currency pair, as a decimal.
    COMMODITY
        Commodity index return, as a decimal.
    LIQUIDITY
        Multiplier on bid-ask spreads.  ``5.0`` means spreads widen fivefold.
        Carried through to the hedging-cost objective of Step 2 rather than to
        instrument prices.
    """

    EQUITY = "equity"
    RATES = "rates"
    CREDIT = "credit"
    VOLATILITY = "volatility"
    FX = "fx"
    COMMODITY = "commodity"
    LIQUIDITY = "liquidity"


@dataclass(frozen=True)
class ShockScenario:
    """A named macroeconomic shock expressed in risk-factor moves.

    Attributes
    ----------
    name
        Short identifier, used as the library key.
    shocks
        Factor moves, in the units documented on :class:`RiskFactor`.  Factors
        omitted from the mapping are treated as unshocked.
    description
        What happened, and what the scenario is designed to test.
    window
        The historical window the calibration is drawn from.
    horizon_days
        Calendar length of the episode.  Used to age options through the
        scenario, since a crisis that unfolds over six months also burns six
        months of time value - an effect a pure spot shock misses.
    source
        Provenance of the calibration.
    """

    name: str
    shocks: Mapping[RiskFactor, float]
    description: str = ""
    window: str = ""
    horizon_days: int = 1
    source: str = ""

    def get(self, factor: RiskFactor) -> float:
        """Return the move for ``factor``, or ``0.0`` if it is unshocked."""
        return float(self.shocks.get(factor, 0.0))

    def scaled(self, multiplier: float, name: str | None = None) -> "ShockScenario":
        """Return a copy with every factor move scaled.

        Severity scaling is how a desk tests whether a book fails gracefully or
        falls off a cliff: a portfolio whose loss doubles when the shock
        doubles is linear, and one whose loss quadruples is short convexity.

        Parameters
        ----------
        multiplier
            Scale factor.  ``0.5`` is a half-severity variant, ``2.0`` a
            double-severity one.
        name
            Name for the variant; defaults to ``"<name>_x<multiplier>"``.

        Returns
        -------
        ShockScenario
        """
        return replace(
            self,
            name=name if name is not None else f"{self.name}_x{multiplier:g}",
            shocks={factor: move * multiplier for factor, move in self.shocks.items()},
            description=f"{self.description} (severity x{multiplier:g})",
        )

    def __str__(self) -> str:  # pragma: no cover - presentation only
        moves = ", ".join(
            f"{factor.value}={move:+.4g}" for factor, move in self.shocks.items()
        )
        return f"{self.name} [{self.window}]: {moves}"


#: Calibrated historical scenarios.  See the module docstring on calibration:
#: these are approximations from published index levels, meant for development
#: and testing, and should be re-derived from the firm's own data in production.
HISTORICAL_SCENARIOS: Final[dict[str, ShockScenario]] = {
    "gfc_2008": ShockScenario(
        name="gfc_2008",
        shocks={
            RiskFactor.EQUITY: -0.46,
            RiskFactor.RATES: -0.0100,
            RiskFactor.CREDIT: +0.0400,
            RiskFactor.VOLATILITY: +2.00,
            RiskFactor.LIQUIDITY: 5.0,
        },
        description=(
            "Global Financial Crisis. Equity collapse with a flight to quality: "
            "government yields fell while credit spreads and implied volatility "
            "exploded. The canonical test of whether a government-bond sleeve "
            "actually hedges equity risk - here it does."
        ),
        window="2008-09-01 to 2009-03-09",
        horizon_days=189,
        source="S&P 500, 10Y UST, IG OAS, VIX (approximate published levels)",
    ),
    "covid_2020": ShockScenario(
        name="covid_2020",
        shocks={
            RiskFactor.EQUITY: -0.339,
            RiskFactor.RATES: -0.0080,
            RiskFactor.CREDIT: +0.0270,
            RiskFactor.VOLATILITY: +4.00,
            RiskFactor.LIQUIDITY: 10.0,
        },
        description=(
            "COVID-19 liquidity shock. The fastest 30% drawdown on record, with "
            "the largest volatility spike in the VIX series and a breakdown in "
            "Treasury market liquidity. Severity is comparable to 2008 but the "
            "speed is not: five weeks against six months, which is why it "
            "stresses liquidity and gamma rather than carry."
        ),
        window="2020-02-19 to 2020-03-23",
        horizon_days=33,
        source="S&P 500, 10Y UST, IG OAS, VIX (approximate published levels)",
    ),
    "black_monday_1987": ShockScenario(
        name="black_monday_1987",
        shocks={
            RiskFactor.EQUITY: -0.2047,
            RiskFactor.RATES: -0.0050,
            RiskFactor.VOLATILITY: +3.00,
            RiskFactor.LIQUIDITY: 8.0,
        },
        description=(
            "Black Monday: a one-day equity crash with essentially no warning. "
            "The single most demanding test of gamma, because there is no "
            "opportunity to rehedge - a delta hedge set on Friday is carried "
            "through the entire move."
        ),
        window="1987-10-19 (single session)",
        horizon_days=1,
        source="S&P 500 close-to-close; VXO for volatility",
    ),
    "euro_crisis_2011": ShockScenario(
        name="euro_crisis_2011",
        shocks={
            RiskFactor.EQUITY: -0.183,
            RiskFactor.RATES: -0.0116,
            RiskFactor.CREDIT: +0.0150,
            RiskFactor.VOLATILITY: +1.60,
            RiskFactor.LIQUIDITY: 3.0,
        },
        description=(
            "European sovereign debt crisis and the US credit rating downgrade. "
            "A moderate-severity scenario, useful because tail testing against "
            "extremes alone leaves the middle of the distribution unexamined."
        ),
        window="2011-07-22 to 2011-10-03",
        horizon_days=73,
        source="S&P 500, 10Y UST, IG OAS, VIX (approximate published levels)",
    ),
    "inflation_shock_2022": ShockScenario(
        name="inflation_shock_2022",
        shocks={
            RiskFactor.EQUITY: -0.254,
            RiskFactor.RATES: +0.0232,
            RiskFactor.CREDIT: +0.0073,
            RiskFactor.VOLATILITY: +1.00,
            RiskFactor.LIQUIDITY: 2.0,
        },
        description=(
            "Inflation and rate-hiking shock. The essential complement to 2008 "
            "and 2020: rates rose *with* equities falling, so the government-bond "
            "sleeve amplified the loss instead of offsetting it. Any portfolio "
            "whose tail risk is controlled by a duration hedge must be tested "
            "here, because this is the scenario in which that hedge fails."
        ),
        window="2022-01-03 to 2022-10-12",
        horizon_days=282,
        source="S&P 500, 10Y UST, IG OAS, VIX (approximate published levels)",
    ),
}


def list_scenarios() -> list[str]:
    """Names of every scenario in the built-in library."""
    return sorted(HISTORICAL_SCENARIOS)


def get_scenario(name: str) -> ShockScenario:
    """Fetch a scenario from the library by name.

    Parameters
    ----------
    name
        Library key, e.g. ``"gfc_2008"``.

    Returns
    -------
    ShockScenario

    Raises
    ------
    KeyError
        If the name is unknown; the message lists the available scenarios,
        since a typo here would otherwise silently skip a stress test.
    """
    try:
        return HISTORICAL_SCENARIOS[name]
    except KeyError:
        raise KeyError(
            f"Unknown scenario {name!r}. Available: {', '.join(list_scenarios())}."
        ) from None


# ==============================================================================
# 2. Greeks and market state
# ==============================================================================


@dataclass(frozen=True)
class Greeks:
    """First- and second-order sensitivities of an instrument, **per unit held**.

    Attributes
    ----------
    delta
        First derivative of value with respect to the underlying spot.
    gamma
        Second derivative with respect to spot.  The quantity this whole module
        exists to account for: it is exactly the curvature that a linear stress
        test discards.
    vega
        Derivative with respect to implied volatility, per ``1.00`` of vol
        (i.e. per 100 vol points).  Divide by 100 for the "per vol point"
        convention used on most desks.
    theta
        Derivative with respect to calendar time, **per year**.  Negative for a
        long option: time value bleeds away.
    rho
        Derivative with respect to the risk-free rate, per ``1.00``.  For a
        bond this is minus its dollar duration.
    rate_gamma
        Second derivative with respect to the risk-free rate - a bond's
        **convexity** in cash terms.  Carried separately from :attr:`gamma`
        because the two are curvature in different variables and cannot be
        added.  Zero for options, where it is an order of magnitude below the
        terms already retained and is conventionally ignored.
    """

    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    rate_gamma: float = 0.0

    def __add__(self, other: "Greeks") -> "Greeks":
        """Greeks are additive across positions in the same underlying."""
        return Greeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            vega=self.vega + other.vega,
            theta=self.theta + other.theta,
            rho=self.rho + other.rho,
            rate_gamma=self.rate_gamma + other.rate_gamma,
        )

    def __mul__(self, units: float) -> "Greeks":
        """Scale by a position size; negative units give a short position."""
        factor = float(units)
        return Greeks(
            delta=self.delta * factor,
            gamma=self.gamma * factor,
            vega=self.vega * factor,
            theta=self.theta * factor,
            rho=self.rho * factor,
            rate_gamma=self.rate_gamma * factor,
        )

    __rmul__ = __mul__

    def as_dict(self) -> dict[str, float]:
        """Return the Greeks as a plain dictionary, for reporting."""
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
            "rate_gamma": self.rate_gamma,
        }


@dataclass(frozen=True)
class UnderlyingSpec:
    """How one underlying responds to the macroeconomic risk factors.

    This is the mapping that lets a single scenario be applied to any book: the
    scenario speaks in factors, the specification translates into a move in
    *this* underlying.

    Attributes
    ----------
    name
        Underlying identifier, matched against :class:`Instrument.underlying`.
    factor_betas
        Return sensitivity to each price-driving factor.  A beta of ``1.35``
        against :attr:`RiskFactor.EQUITY` means the underlying falls 13.5% when
        the broad index falls 10%.  Betas against :attr:`RiskFactor.RATES` and
        :attr:`RiskFactor.CREDIT` are expressed as return per unit of level
        change - a rates beta of ``-2.0`` costs 2% of value per 100bp of yield
        rise.
    volatility_beta
        Sensitivity of this underlying's implied volatility to the
        :attr:`RiskFactor.VOLATILITY` factor, as a multiple of the relative
        move.  ``1.0`` tracks the index vol one for one.
    """

    name: str
    factor_betas: Mapping[RiskFactor, float] = field(default_factory=dict)
    volatility_beta: float = 1.0

    def spot_return(self, scenario: ShockScenario) -> float:
        """Return implied for this underlying by a scenario.

        The factor contributions are summed linearly.  That is a modelling
        choice, and a deliberate one: cross-factor interaction terms are not
        identifiable from the handful of historical episodes available, so
        inventing them would add precision the data cannot support.  Curvature
        in the *instrument* is captured exactly, by repricing - which is where
        it actually matters.
        """
        total = 0.0
        for factor, beta in self.factor_betas.items():
            total += beta * scenario.get(factor)
        return total

    def volatility_multiplier(self, scenario: ShockScenario) -> float:
        """Multiplicative factor applied to this underlying's implied vol."""
        relative = self.volatility_beta * scenario.get(RiskFactor.VOLATILITY)
        # A vol multiplier can never be negative; a shock of -1.5 would
        # otherwise imply negative volatility.
        return max(1.0 + relative, 0.0)


@dataclass(frozen=True)
class MarketState:
    """A snapshot of everything needed to price the book.

    Attributes
    ----------
    spots
        Spot level per underlying.
    volatilities
        Annualised implied volatility per underlying, as a decimal.
    rate
        Continuously compounded risk-free rate.
    dividend_yields
        Continuous dividend yield per underlying; absent entries are zero.
    time
        Elapsed time in years, used to age options.  Increasing it shortens
        every option's remaining life.
    """

    spots: Mapping[str, float]
    volatilities: Mapping[str, float] = field(default_factory=dict)
    rate: float = 0.0
    dividend_yields: Mapping[str, float] = field(default_factory=dict)
    time: float = 0.0

    def spot(self, underlying: str) -> float:
        """Spot level of ``underlying``.

        Raises
        ------
        KeyError
            If the underlying is missing.  Defaulting to zero here would price
            an entire sleeve at nothing and quietly understate the loss.
        """
        try:
            return float(self.spots[underlying])
        except KeyError:
            raise KeyError(
                f"No spot for underlying {underlying!r}; known: "
                f"{sorted(self.spots)}."
            ) from None

    def volatility(self, underlying: str) -> float:
        """Implied volatility of ``underlying``; zero when unspecified."""
        return float(self.volatilities.get(underlying, 0.0))

    def dividend_yield(self, underlying: str) -> float:
        """Continuous dividend yield of ``underlying``; zero when unspecified."""
        return float(self.dividend_yields.get(underlying, 0.0))

    def apply(
        self,
        scenario: ShockScenario,
        underlyings: Mapping[str, UnderlyingSpec],
        age_forward: bool = True,
    ) -> "MarketState":
        """Return the market state implied by a scenario.

        Spots move by their factor-implied return, implied volatilities are
        multiplied by their vol beta, and the risk-free rate takes the parallel
        shift.  When ``age_forward`` is set, the clock advances by the
        scenario's horizon, so options lose time value as well as spot - a
        six-month crisis is not the same as an instantaneous jump of the same
        size, and treating it as one flatters a long-option book.

        Parameters
        ----------
        scenario
            The shock to apply.
        underlyings
            Specifications keyed by underlying name.
        age_forward
            Whether to advance the clock by ``scenario.horizon_days``.

        Returns
        -------
        MarketState
            A new state; the original is left untouched.

        Raises
        ------
        KeyError
            If an underlying present in ``spots`` has no specification, which
            would otherwise leave it silently unshocked.
        """
        missing = set(self.spots) - set(underlyings)
        if missing:
            raise KeyError(
                f"No UnderlyingSpec for {sorted(missing)}; every underlying must "
                "declare its factor sensitivities or it would pass through the "
                "scenario unshocked."
            )

        shocked_spots: dict[str, float] = {}
        shocked_vols: dict[str, float] = {}
        for name, spot in self.spots.items():
            spec = underlyings[name]
            # A return below -100% would imply a negative price.
            shocked_spots[name] = float(spot) * max(1.0 + spec.spot_return(scenario), 0.0)
            shocked_vols[name] = self.volatility(name) * spec.volatility_multiplier(scenario)

        horizon = scenario.horizon_days / 365.0 if age_forward else 0.0
        return MarketState(
            spots=shocked_spots,
            volatilities=shocked_vols,
            rate=self.rate + scenario.get(RiskFactor.RATES),
            dividend_yields=dict(self.dividend_yields),
            time=self.time + horizon,
        )


# ==============================================================================
# 3. Black-Scholes-Merton pricing and Greeks
# ==============================================================================


class OptionType(str, Enum):
    """Option right."""

    CALL = "call"
    PUT = "put"

    @property
    def sign(self) -> float:
        """``+1`` for a call, ``-1`` for a put - the omega used in BSM formulae."""
        return 1.0 if self is OptionType.CALL else -1.0


def _d1_d2(
    spot: float, strike: float, maturity: float, rate: float, dividend: float, vol: float
) -> tuple[float, float]:
    """The standard BSM auxiliary quantities.

    .. math::

        d_1 = \\frac{\\ln(S/K) + (r - q + \\sigma^{2}/2)T}{\\sigma\\sqrt{T}},
        \\qquad d_2 = d_1 - \\sigma\\sqrt{T}.
    """
    sqrt_t = np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate - dividend + 0.5 * vol**2) * maturity) / (
        vol * sqrt_t
    )
    return float(d1), float(d1 - vol * sqrt_t)


def _is_degenerate(maturity: float, vol: float, spot: float, strike: float) -> bool:
    """Whether the BSM formula is undefined and intrinsic value must be used."""
    return maturity <= 0.0 or vol <= 0.0 or spot <= 0.0 or strike <= 0.0


def black_scholes_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> float:
    """Black-Scholes-Merton price of a European option.

    .. math::

        C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \\qquad
        P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1).

    Degenerate inputs - expiry reached, zero volatility, or a worthless
    underlying - fall back to discounted intrinsic value rather than raising.
    Stress scenarios routinely push options to expiry or drive a spot to zero,
    and a stress engine that throws on its most extreme scenario is useless
    exactly when it is needed.

    Parameters
    ----------
    spot
        Underlying spot level.
    strike
        Strike.
    maturity
        Time to expiry in years.
    rate
        Continuously compounded risk-free rate.
    volatility
        Annualised implied volatility as a decimal.
    option_type
        Call or put.
    dividend_yield
        Continuous dividend yield.

    Returns
    -------
    float
        Option value per unit.
    """
    spot, strike = float(spot), float(strike)
    maturity, vol = float(maturity), float(volatility)

    if _is_degenerate(maturity, vol, spot, strike):
        intrinsic = max(option_type.sign * (spot - strike), 0.0)
        return float(intrinsic * np.exp(-rate * max(maturity, 0.0)))

    d1, d2 = _d1_d2(spot, strike, maturity, rate, dividend_yield, vol)
    discounted_spot = spot * np.exp(-dividend_yield * maturity)
    discounted_strike = strike * np.exp(-rate * maturity)

    if option_type is OptionType.CALL:
        return float(discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2))
    return float(discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1))


def black_scholes_greeks(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> Greeks:
    """Analytic Black-Scholes-Merton Greeks.

    .. math::

        \\Delta_{\\text{call}} &= e^{-qT} N(d_1), \\quad
        \\Delta_{\\text{put}} = -e^{-qT} N(-d_1) \\\\
        \\Gamma &= \\frac{e^{-qT}\\varphi(d_1)}{S\\sigma\\sqrt{T}} \\\\
        \\mathcal{V} &= S e^{-qT}\\varphi(d_1)\\sqrt{T}

    Gamma and vega are identical for calls and puts, as put-call parity
    requires: the two differ by a forward, which is linear in spot and
    independent of volatility.

    At expiry gamma is a Dirac spike at the strike and vega vanishes; both are
    reported as zero, with delta taking the step-function limit.  That is the
    correct limit for a book that has already paid off, and it keeps the
    aggregate finite.

    Parameters
    ----------
    See :func:`black_scholes_price`.

    Returns
    -------
    Greeks
        Per-unit sensitivities.
    """
    spot, strike = float(spot), float(strike)
    maturity, vol = float(maturity), float(volatility)

    if _is_degenerate(maturity, vol, spot, strike):
        # Past expiry (or with no uncertainty left) the payoff is piecewise
        # linear: unit delta in the money, zero outside, and no curvature.
        in_the_money = option_type.sign * (spot - strike) > 0.0
        return Greeks(delta=option_type.sign if in_the_money else 0.0)

    d1, d2 = _d1_d2(spot, strike, maturity, rate, dividend_yield, vol)
    sqrt_t = np.sqrt(maturity)
    dividend_discount = np.exp(-dividend_yield * maturity)
    rate_discount = np.exp(-rate * maturity)
    pdf_d1 = float(norm.pdf(d1))

    gamma = dividend_discount * pdf_d1 / (spot * vol * sqrt_t)
    vega = spot * dividend_discount * pdf_d1 * sqrt_t
    common_theta = -spot * dividend_discount * pdf_d1 * vol / (2.0 * sqrt_t)

    if option_type is OptionType.CALL:
        delta = dividend_discount * float(norm.cdf(d1))
        theta = (
            common_theta
            - rate * strike * rate_discount * float(norm.cdf(d2))
            + dividend_yield * spot * dividend_discount * float(norm.cdf(d1))
        )
        rho = strike * maturity * rate_discount * float(norm.cdf(d2))
    else:
        delta = -dividend_discount * float(norm.cdf(-d1))
        theta = (
            common_theta
            + rate * strike * rate_discount * float(norm.cdf(-d2))
            - dividend_yield * spot * dividend_discount * float(norm.cdf(-d1))
        )
        rho = -strike * maturity * rate_discount * float(norm.cdf(-d2))

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


# ==============================================================================
# 4. Instruments
# ==============================================================================


class Instrument(ABC):
    """An instrument that can be priced and differentiated in a market state.

    Concrete subclasses supply a valuation and its Greeks.  The stress engine
    depends only on this interface, so adding a new payoff - a barrier option,
    a swap, a convertible - requires no change to the engine.
    """

    # Declared as bare annotations, deliberately without values: dataclass
    # subclasses resolve field defaults through getattr on the MRO, so
    # assigning here would make 'name' and 'underlying' defaulted fields and
    # force every subsequent subclass field to carry a default too.
    #: Instrument label used in reports.
    name: str

    #: Name of the underlying whose spot drives this instrument.
    underlying: str

    @abstractmethod
    def value(self, market: MarketState) -> float:
        """Value per unit in the given market state."""

    @abstractmethod
    def greeks(self, market: MarketState) -> Greeks:
        """Per-unit sensitivities in the given market state."""

    @property
    def is_linear(self) -> bool:
        """Whether the payoff is linear in the underlying, i.e. gamma is zero.

        Used to explain stress reports: when a book is entirely linear, the
        delta approximation is exact and any residual is numerical noise.
        """
        return True

    def __repr__(self) -> str:  # pragma: no cover - presentation only
        return f"{type(self).__name__}(name={self.name!r}, underlying={self.underlying!r})"


@dataclass
class EquityInstrument(Instrument):
    """A cash equity or index position - linear in the underlying.

    Delta is one by definition (the position *is* the underlying) and gamma is
    zero, so a delta-only stress test is exact for this instrument.

    Attributes
    ----------
    name
        Instrument label.
    underlying
        Underlying whose spot it tracks.
    """

    name: str
    underlying: str

    def value(self, market: MarketState) -> float:
        """Value per unit: the spot level itself."""
        return market.spot(self.underlying)

    def greeks(self, market: MarketState) -> Greeks:
        """Unit delta, no curvature."""
        return Greeks(delta=1.0)


@dataclass
class BondInstrument(Instrument):
    """A bond position priced by duration and convexity.

    .. math::

        \\frac{\\Delta P}{P} \\;\\approx\\; -D\\,\\Delta y
            \\;+\\; \\tfrac{1}{2} C\\, (\\Delta y)^{2}

    Convexity is the bond market's gamma, and it enters with the same sign
    convention and the same significance: for the +232bp move in the 2022
    scenario, the convexity term offsets a meaningful part of the duration
    loss, and a duration-only stress overstates the damage.  Including it here
    keeps the treatment of rate curvature consistent with the treatment of
    equity curvature.

    The bond's rate sensitivity lives **here**, in its duration and convexity -
    not in its underlying's :attr:`RiskFactor.RATES` beta.  Specifying both
    would double count the yield move, so :class:`StressTester` refuses that
    combination rather than silently reporting twice the loss.

    Attributes
    ----------
    name
        Instrument label.
    underlying
        Underlying key; its spot carries the bond's clean price.
    duration
        Modified duration in years.
    convexity
        Convexity in years squared.  Always positive for a vanilla bond, which
        is why a duration-only stress *overstates* the loss from a yield rise:
        the 2022 scenario's +232bp move produces a convexity gain worth several
        per cent of the sleeve.
    base_rate
        The rate level at which the quoted price applies.  Yield changes are
        measured from here, so it must equal the base market's rate -
        :class:`StressTester` checks this at construction.
    """

    name: str
    underlying: str
    duration: float
    convexity: float = 0.0
    base_rate: float = 0.0

    def __post_init__(self) -> None:
        """Validate the risk measures."""
        if self.duration < 0.0:
            raise ValueError(f"'duration' must be non-negative, got {self.duration!r}.")
        if self.convexity < 0.0:
            raise ValueError(
                f"'convexity' must be non-negative for a vanilla bond, "
                f"got {self.convexity!r}."
            )

    @property
    def is_linear(self) -> bool:
        """A bond with convexity is non-linear in yield."""
        return self.convexity == 0.0

    def yield_change(self, market: MarketState) -> float:
        """Yield move implied by a market state, relative to :attr:`base_rate`."""
        return market.rate - self.base_rate

    def value(self, market: MarketState) -> float:
        """Clean price after the duration and convexity response to rates.

        .. math::

            P(y_0 + \\Delta y) \\;=\\; P_0 \\left[
                1 - D\\,\\Delta y + \\tfrac{1}{2} C\\,(\\Delta y)^{2}\\right]

        This is a full revaluation *of this model*: the second-order term is
        included exactly, so the difference between full and delta-only P&L for
        a bond is precisely its convexity contribution.
        """
        change = self.yield_change(market)
        relative = -self.duration * change + 0.5 * self.convexity * change**2
        # A price cannot go negative; the quadratic form is only a local
        # expansion and would eventually turn upward for extreme moves.
        return market.spot(self.underlying) * max(1.0 + relative, 0.0)

    def greeks(self, market: MarketState) -> Greeks:
        """Rate sensitivities mapped into the Greek container.

        ``rho`` carries minus the dollar duration - the first derivative of
        price with respect to yield - so bond rate risk aggregates alongside
        the option book's rho.  ``rate_gamma`` carries dollar convexity.
        ``delta`` is one with respect to the bond's own price, matching the
        :class:`EquityInstrument` convention.
        """
        price = self.value(market)
        return Greeks(
            delta=1.0,
            rho=-self.duration * price,
            rate_gamma=self.convexity * price,
        )


@dataclass
class EuropeanOption(Instrument):
    """A European option priced by Black-Scholes-Merton.

    The instrument that makes this module necessary.  Its value is a convex
    (long) or concave (short) function of spot, so no first-order sensitivity
    can describe its behaviour under a large shock.

    Attributes
    ----------
    name
        Instrument label.
    underlying
        Underlying whose spot drives it.
    strike
        Strike price.
    maturity
        Time to expiry in years, measured from :attr:`MarketState.time` zero.
        Ageing the market state reduces the remaining life accordingly.
    option_type
        Call or put.
    contract_size
        Units of the underlying per contract; ``1.0`` prices per unit.
    """

    name: str
    underlying: str
    strike: float
    maturity: float
    option_type: OptionType
    contract_size: float = 1.0

    def __post_init__(self) -> None:
        """Validate contract terms."""
        if self.strike <= 0.0:
            raise ValueError(f"'strike' must be positive, got {self.strike!r}.")
        if self.maturity < 0.0:
            raise ValueError(f"'maturity' must be non-negative, got {self.maturity!r}.")
        if self.contract_size <= 0.0:
            raise ValueError(
                f"'contract_size' must be positive, got {self.contract_size!r}."
            )

    @property
    def is_linear(self) -> bool:
        """Options are never linear in the underlying."""
        return False

    def remaining_maturity(self, market: MarketState) -> float:
        """Time to expiry left after the market state's clock, floored at zero."""
        return max(self.maturity - market.time, 0.0)

    def value(self, market: MarketState) -> float:
        """Black-Scholes-Merton value, scaled by the contract size."""
        return self.contract_size * black_scholes_price(
            spot=market.spot(self.underlying),
            strike=self.strike,
            maturity=self.remaining_maturity(market),
            rate=market.rate,
            volatility=market.volatility(self.underlying),
            option_type=self.option_type,
            dividend_yield=market.dividend_yield(self.underlying),
        )

    def greeks(self, market: MarketState) -> Greeks:
        """Analytic Greeks, scaled by the contract size."""
        raw = black_scholes_greeks(
            spot=market.spot(self.underlying),
            strike=self.strike,
            maturity=self.remaining_maturity(market),
            rate=market.rate,
            volatility=market.volatility(self.underlying),
            option_type=self.option_type,
            dividend_yield=market.dividend_yield(self.underlying),
        )
        return raw * self.contract_size


# ==============================================================================
# 5. Portfolio
# ==============================================================================


@dataclass
class StressPortfolio:
    """A book of instruments with position sizes.

    Attributes
    ----------
    instruments
        The instruments held.
    units
        Position size per instrument, same length and order.  Negative units
        are short positions - and short option positions are where a
        delta-only stress test is most dangerous, because concavity means the
        linear approximation *overstates* the value that survives a shock.
    """

    instruments: Sequence[Instrument]
    units: FloatArray

    def __post_init__(self) -> None:
        """Validate and coerce the position vector."""
        self.units = _as_1d(self.units, "units")
        if len(self.instruments) != self.units.size:
            raise ValueError(
                f"{len(self.instruments)} instruments but {self.units.size} unit "
                "entries; they must correspond one to one."
            )

    def __len__(self) -> int:
        """Number of instruments in the book."""
        return len(self.instruments)

    @classmethod
    def from_weights(
        cls,
        weights: npt.ArrayLike,
        instruments: Sequence[Instrument],
        market: MarketState,
        total_value: float = 1_000_000.0,
    ) -> "StressPortfolio":
        """Build a book from portfolio weights - the bridge from Steps 1 and 2.

        Converts a weight vector on the budget simplex into position sizes:
        ``units_i = weights_i * total_value / price_i``.  This is the point
        where an abstract optimised allocation becomes a set of positions that
        can be revalued instrument by instrument.

        Parameters
        ----------
        weights
            Portfolio weights, shape ``(n,)``, aligned with ``instruments``.
        instruments
            One instrument per weight.
        market
            Base market state, supplying the prices used to convert value to
            units.
        total_value
            Total portfolio value to allocate.

        Returns
        -------
        StressPortfolio

        Raises
        ------
        ValueError
            On a length mismatch, or if any instrument prices at zero or below
            - a position size cannot be derived from a zero price, and silently
            dropping the sleeve would understate the book's risk.
        """
        w = _as_1d(weights, "weights")
        if w.size != len(instruments):
            raise ValueError(
                f"'weights' has {w.size} entries but {len(instruments)} instruments "
                "were supplied."
            )

        prices = np.array([instrument.value(market) for instrument in instruments])
        invalid = np.flatnonzero(prices <= 0.0)
        if invalid.size:
            names = [instruments[i].name for i in invalid]
            raise ValueError(
                f"Cannot size positions from a non-positive price: {names}. "
                "Check the base market state."
            )
        return cls(instruments=list(instruments), units=w * float(total_value) / prices)

    def value(self, market: MarketState) -> float:
        """Total book value in the given market state."""
        return float(
            sum(
                unit * instrument.value(market)
                for unit, instrument in zip(self.units, self.instruments)
            )
        )

    def instrument_values(self, market: MarketState) -> FloatArray:
        """Value of each position, shape ``(n,)``."""
        return np.array(
            [
                unit * instrument.value(market)
                for unit, instrument in zip(self.units, self.instruments)
            ]
        )

    def greeks_by_underlying(self, market: MarketState) -> dict[str, Greeks]:
        """Position-weighted Greeks, aggregated per underlying.

        Aggregation is per underlying rather than book-wide because delta and
        gamma are derivatives with respect to *a particular* spot: adding the
        delta of an S&P option to the delta of a small-cap option produces a
        number with no meaning.  Cash-equivalent measures
        (:meth:`cash_greeks`) are what aggregate across underlyings.
        """
        totals: dict[str, Greeks] = {}
        for unit, instrument in zip(self.units, self.instruments):
            scaled = instrument.greeks(market) * unit
            key = instrument.underlying
            totals[key] = totals[key] + scaled if key in totals else scaled
        return totals

    def cash_greeks(self, market: MarketState) -> dict[str, float]:
        """Book-level cash Greeks, in currency units.

        Desk conventions, chosen because they are additive across underlyings:

        ``delta_cash``
            ``sum_i units_i * delta_i * S_i`` - the equivalent underlying
            notional.  A book with 12m of delta cash gains 120k on a 1% rally.
        ``gamma_cash_1pct``
            ``sum_i units_i * gamma_i * S_i^2 * 0.01`` - how much ``delta_cash``
            *changes* for a +1% move.  This is the number that says how fast
            the hedge goes stale.
        ``vega_cash_1pt``
            Value change per one volatility point.
        ``theta_cash_1d``
            Value decay per calendar day.
        ``dv01``
            Value change per one basis point of parallel yield rise - the
            standard rate-risk aggregate, and negative for a long bond book.

        Returns
        -------
        dict
            The five aggregates above.
        """
        delta_cash = gamma_cash = vega_cash = theta_cash = dv01 = 0.0
        for unit, instrument in zip(self.units, self.instruments):
            greeks = instrument.greeks(market) * unit
            spot = market.spot(instrument.underlying)
            delta_cash += greeks.delta * spot
            gamma_cash += greeks.gamma * spot**2 * 0.01
            vega_cash += greeks.vega * 0.01
            theta_cash += greeks.theta / 365.0
            dv01 += greeks.rho * 0.0001
        return {
            "delta_cash": delta_cash,
            "gamma_cash_1pct": gamma_cash,
            "vega_cash_1pt": vega_cash,
            "theta_cash_1d": theta_cash,
            "dv01": dv01,
        }

    @property
    def has_optionality(self) -> bool:
        """Whether any position is non-linear, i.e. whether gamma matters here."""
        return any(not instrument.is_linear for instrument in self.instruments)


# ==============================================================================
# 6. Stress testing engine
# ==============================================================================


class RevaluationMethod(str, Enum):
    """How a scenario's P&L is computed.

    Attributes
    ----------
    DELTA
        First order only: ``dV = sum_i delta_i * dS_i``.  Fast, and the standard
        content of a risk report, but for an option book it is wrong by a
        margin that grows with the square of the shock.
    DELTA_GAMMA
        Second order, plus the vega term:
        ``dV = sum_i [delta_i * dS_i + 0.5 * gamma_i * dS_i^2 + vega_i * dsigma_i]``.
        The industry standard approximation, and usually good to a few per cent
        for moderate shocks.
    FULL
        Reprice every instrument in the shocked market.  Exact by construction,
        with no dependence on the size of the move, and the only method that is
        trustworthy for a crisis-sized scenario.  This is the reference every
        other method is scored against.
    """

    DELTA = "delta"
    DELTA_GAMMA = "delta_gamma"
    FULL = "full"


@dataclass(frozen=True)
class StressResult:
    """Outcome of one scenario applied to one book.

    Attributes
    ----------
    scenario
        The scenario that was run.
    base_value, stressed_value
        Book value before and after, under full revaluation.
    pnl_full, pnl_delta, pnl_delta_gamma
        Profit and loss under each method.  ``pnl_full`` is exact.
    delta_error, delta_gamma_error
        Approximation error against full revaluation, as a *fraction* of the
        exact P&L.  These are the headline diagnostics: a delta error of 0.30
        means a linear stress report understated or overstated the loss by 30%.
    gamma_contribution
        The part of ``pnl_delta_gamma`` attributable to curvature.  Positive
        for a long-gamma book, which gains from large moves in either
        direction; negative for a short-gamma book, which bleeds.
    vega_contribution
        The part attributable to the implied-volatility move.
    instrument_pnl
        Exact P&L per instrument, shape ``(n,)``.
    base_greeks, stressed_greeks
        Cash Greeks before and after.  The change in ``delta_cash`` shows how
        far the hedge drifted during the move, which is the practical cost of
        gamma.
    base_market, stressed_market
        The two market states, retained so a result can be re-examined without
        re-running.
    """

    scenario: ShockScenario
    base_value: float
    stressed_value: float
    pnl_full: float
    pnl_delta: float
    pnl_delta_gamma: float
    delta_error: float
    delta_gamma_error: float
    gamma_contribution: float
    vega_contribution: float
    instrument_pnl: FloatArray
    base_greeks: dict[str, float]
    stressed_greeks: dict[str, float]
    base_market: MarketState
    stressed_market: MarketState

    @property
    def return_pct(self) -> float:
        """Exact P&L as a fraction of the starting book value."""
        return self.pnl_full / self.base_value if self.base_value else float("nan")

    def pnl(self, method: RevaluationMethod) -> float:
        """P&L under the requested revaluation method."""
        return {
            RevaluationMethod.FULL: self.pnl_full,
            RevaluationMethod.DELTA: self.pnl_delta,
            RevaluationMethod.DELTA_GAMMA: self.pnl_delta_gamma,
        }[method]

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"{self.scenario.name}: "
            f"P&L {self.pnl_full:,.0f} ({self.return_pct:+.2%})  |  "
            f"delta-only {self.pnl_delta:,.0f} (err {self.delta_error:+.1%})  |  "
            f"delta-gamma {self.pnl_delta_gamma:,.0f} "
            f"(err {self.delta_gamma_error:+.1%})"
        )


class StressTester:
    """Applies shock scenarios to a book and reports non-linear P&L.

    Every scenario is evaluated by all three methods of
    :class:`RevaluationMethod`, because the *difference* between them is the
    output that matters.  Reporting a single number would hide exactly the
    effect the module exists to expose.

    Parameters
    ----------
    portfolio
        The book to stress.
    base_market
        The market state the book is currently marked at.
    underlyings
        Factor sensitivities per underlying.  Every underlying appearing in
        ``base_market.spots`` must have one.

    Raises
    ------
    KeyError
        If an underlying lacks a specification.
    ValueError
        If a :class:`BondInstrument` would have its rate exposure counted
        twice, or if its ``base_rate`` disagrees with the base market.
    """

    def __init__(
        self,
        portfolio: StressPortfolio,
        base_market: MarketState,
        underlyings: Mapping[str, UnderlyingSpec],
    ) -> None:
        missing = set(base_market.spots) - set(underlyings)
        if missing:
            raise KeyError(f"No UnderlyingSpec for {sorted(missing)}.")
        self.portfolio = portfolio
        self.base_market = base_market
        self.underlyings = dict(underlyings)
        self._validate_bond_configuration()

        if not portfolio.has_optionality:
            LOGGER.info(
                "Book is entirely linear: delta-only P&L will match full "
                "revaluation exactly, and the approximation errors will be zero."
            )

    def _validate_bond_configuration(self) -> None:
        """Guard the two ways a bond's rate exposure can be misconfigured.

        A :class:`BondInstrument` responds to rates through its own duration
        and convexity.  If its underlying *also* carries a
        :attr:`RiskFactor.RATES` beta, the yield move is applied twice - once
        to the spot and once through duration - and the reported loss is
        roughly doubled.  Nothing downstream would flag that, so it is caught
        here.

        Likewise a bond whose ``base_rate`` differs from the base market's rate
        is already off its quoted price before any scenario is applied, which
        silently biases every result.
        """
        for instrument in self.portfolio.instruments:
            if not isinstance(instrument, BondInstrument):
                continue

            spec = self.underlyings[instrument.underlying]
            rates_beta = spec.factor_betas.get(RiskFactor.RATES, 0.0)
            if rates_beta != 0.0:
                raise ValueError(
                    f"Bond '{instrument.name}' carries duration "
                    f"{instrument.duration:g}, but its underlying "
                    f"'{instrument.underlying}' also has a RATES beta of "
                    f"{rates_beta:g}. The yield move would be applied twice. "
                    "Set the underlying's RATES beta to zero and let duration "
                    "and convexity carry the rate response."
                )

            if abs(instrument.base_rate - self.base_market.rate) > 1e-12:
                raise ValueError(
                    f"Bond '{instrument.name}' has base_rate="
                    f"{instrument.base_rate:g} but the base market rate is "
                    f"{self.base_market.rate:g}; the quoted price would not "
                    "correspond to the base state."
                )

    def run(self, scenario: ShockScenario, age_forward: bool = True) -> StressResult:
        """Apply one scenario and return the full diagnostic set.

        Parameters
        ----------
        scenario
            The shock to apply.
        age_forward
            Whether to advance the clock by the scenario horizon, so options
            lose time value across the episode as well as spot.

        Returns
        -------
        StressResult
        """
        base = self.base_market
        stressed = base.apply(scenario, self.underlyings, age_forward=age_forward)

        base_values = self.portfolio.instrument_values(base)
        stressed_values = self.portfolio.instrument_values(stressed)
        base_value = float(base_values.sum())
        pnl_full = float(stressed_values.sum() - base_value)

        pnl_delta, pnl_gamma, pnl_vega = self._taylor_terms(base, stressed)

        return StressResult(
            scenario=scenario,
            base_value=base_value,
            stressed_value=float(stressed_values.sum()),
            pnl_full=pnl_full,
            pnl_delta=pnl_delta,
            pnl_delta_gamma=pnl_delta + pnl_gamma + pnl_vega,
            delta_error=_relative_error(pnl_delta, pnl_full),
            delta_gamma_error=_relative_error(pnl_delta + pnl_gamma + pnl_vega, pnl_full),
            gamma_contribution=pnl_gamma,
            vega_contribution=pnl_vega,
            instrument_pnl=stressed_values - base_values,
            base_greeks=self.portfolio.cash_greeks(base),
            stressed_greeks=self.portfolio.cash_greeks(stressed),
            base_market=base,
            stressed_market=stressed,
        )

    def _taylor_terms(
        self, base: MarketState, stressed: MarketState
    ) -> tuple[float, float, float]:
        """Decompose the Taylor expansion into first-order, curvature and vega terms.

        .. math::

            \\Delta V \\;\\approx\\;
            \\underbrace{\\sum_i \\Delta_i\\, \\delta S_i + \\rho_i\\, \\delta r}
                _{\\text{first order}}
            \\;+\\; \\underbrace{\\tfrac{1}{2}\\sum_i \\Gamma_i\\, \\delta S_i^{2}
                + \\tfrac{1}{2}\\Gamma^{r}_i\\, \\delta r^{2}}
                _{\\text{curvature}}
            \\;+\\; \\underbrace{\\sum_i \\mathcal{V}_i\\, \\delta \\sigma_i}
                _{\\text{vega}}

        Rate terms are included in both orders.  Without them a bond sleeve
        would contribute nothing to the approximation while contributing its
        full P&L to the exact revaluation, and the resulting "approximation
        error" would be an artefact of the decomposition rather than a
        statement about curvature.

        All sensitivities are taken at the **base** market state, which is the
        point of the exercise: a risk report is produced before the crisis, not
        during it, so the approximation can only ever use today's Greeks.

        Returns
        -------
        (float, float, float)
            The first-order, curvature and vega contributions.
        """
        rate_move = stressed.rate - base.rate
        first_order = curvature = vega_term = 0.0

        for unit, instrument in zip(self.portfolio.units, self.portfolio.instruments):
            greeks = instrument.greeks(base) * unit
            underlying = instrument.underlying
            spot_move = stressed.spot(underlying) - base.spot(underlying)
            vol_move = stressed.volatility(underlying) - base.volatility(underlying)

            first_order += greeks.delta * spot_move + greeks.rho * rate_move
            curvature += (
                0.5 * greeks.gamma * spot_move**2 + 0.5 * greeks.rate_gamma * rate_move**2
            )
            vega_term += greeks.vega * vol_move
        return first_order, curvature, vega_term

    def run_all(
        self,
        scenarios: Sequence[ShockScenario] | None = None,
        age_forward: bool = True,
    ) -> list[StressResult]:
        """Run a set of scenarios, defaulting to the whole historical library.

        Parameters
        ----------
        scenarios
            Scenarios to run; defaults to every entry in
            :data:`HISTORICAL_SCENARIOS`.
        age_forward
            Passed through to :meth:`run`.

        Returns
        -------
        list of StressResult
            One per scenario, in the order supplied.
        """
        chosen = (
            [HISTORICAL_SCENARIOS[name] for name in list_scenarios()]
            if scenarios is None
            else list(scenarios)
        )
        return [self.run(scenario, age_forward=age_forward) for scenario in chosen]

    def worst_case(
        self, scenarios: Sequence[ShockScenario] | None = None
    ) -> StressResult:
        """The scenario producing the largest loss under full revaluation.

        Returns
        -------
        StressResult
            The single worst outcome - the number that goes to the risk
            committee.
        """
        results = self.run_all(scenarios)
        return min(results, key=lambda result: result.pnl_full)

    def to_frame(self, results: Sequence[StressResult] | None = None):
        """Summarise results as a ``pandas.DataFrame``.

        Imported lazily so the stress core carries no hard pandas dependency.

        Parameters
        ----------
        results
            Results to tabulate; defaults to running the full library.

        Returns
        -------
        pandas.DataFrame
            One row per scenario with P&L by method and the approximation
            errors.
        """
        import pandas as pd

        rows = self.run_all() if results is None else list(results)
        return pd.DataFrame(
            [
                {
                    "scenario": result.scenario.name,
                    "pnl_full": result.pnl_full,
                    "return_pct": result.return_pct,
                    "pnl_delta": result.pnl_delta,
                    "pnl_delta_gamma": result.pnl_delta_gamma,
                    "delta_error": result.delta_error,
                    "delta_gamma_error": result.delta_gamma_error,
                    "gamma_contribution": result.gamma_contribution,
                    "vega_contribution": result.vega_contribution,
                    "delta_cash_base": result.base_greeks["delta_cash"],
                    "delta_cash_stressed": result.stressed_greeks["delta_cash"],
                    "gamma_cash_1pct": result.base_greeks["gamma_cash_1pct"],
                }
                for result in rows
            ]
        )


def _relative_error(approximation: float, exact: float) -> float:
    """Signed relative error of an approximation, guarded against a zero base."""
    if abs(exact) < 1e-12:
        return 0.0 if abs(approximation) < 1e-12 else float("inf")
    return (approximation - exact) / abs(exact)


# ==============================================================================
# 7. Bridge back to the CVaR engine
# ==============================================================================


def default_underlyings(bond_duration_in_instrument: bool = False) -> dict[str, UnderlyingSpec]:
    """Factor sensitivities for the three instruments of the paper's Section 3.

    Calibrated to be economically sensible rather than estimated:

    * **S&P 500** - unit equity beta by definition; mild negative rate
      sensitivity, since a rise in discount rates compresses valuations.
    * **Gov Bond** - no equity beta; its risk is duration, so a 100bp yield
      rise costs roughly ten per cent of value for the long-dated sleeve the
      paper uses.
    * **Small Cap** - higher equity beta and, unlike large caps, genuine credit
      sensitivity: small issuers are funded at a spread and are hurt directly
      when that spread widens.  Its implied vol also moves more than the
      index's.

    Parameters
    ----------
    bond_duration_in_instrument
        Where the bond's rate response lives.  ``False`` (the default) puts it
        in the ``GOVT`` rates beta, which is what
        :func:`stress_return_scenarios` needs: at asset level there are no
        instruments, so the beta is the only channel through which a yield move
        can reach the returns.  Set ``True`` when the bond sleeve is modelled
        as a :class:`BondInstrument`, whose duration and convexity then carry
        the response - :class:`StressTester` rejects the combination of both,
        since it would double count the yield move.

    Returns
    -------
    dict
        Specifications keyed by underlying name.
    """
    govt_betas: dict[RiskFactor, float] = (
        {} if bond_duration_in_instrument else {RiskFactor.RATES: -10.0}
    )
    return {
        "SPX": UnderlyingSpec(
            name="SPX",
            factor_betas={RiskFactor.EQUITY: 1.00, RiskFactor.RATES: -1.50},
            volatility_beta=1.0,
        ),
        "GOVT": UnderlyingSpec(
            name="GOVT",
            factor_betas=govt_betas,
            volatility_beta=0.5,
        ),
        "SMALL": UnderlyingSpec(
            name="SMALL",
            factor_betas={
                RiskFactor.EQUITY: 1.35,
                RiskFactor.RATES: -2.00,
                RiskFactor.CREDIT: -2.50,
            },
            volatility_beta=1.3,
        ),
    }


def stress_return_scenarios(
    returns: npt.ArrayLike,
    scenario: ShockScenario,
    underlyings: Sequence[UnderlyingSpec],
    volatility_multiplier: float | None = None,
) -> FloatArray:
    """Shift and widen a scenario return matrix to reflect a stress scenario.

    The transformation applied to each asset column is

    .. math::

        \\tilde{y}_j \\;=\\; \\bar{y}_j + s_j
            \\;+\\; \\kappa\\,\\big(y_j - \\bar{y}_j\\big),

    a **shift** by the scenario's implied return :math:`s_j` together with a
    **widening** of the dispersion by :math:`\\kappa`.  Both parts matter and
    for different reasons: a crisis moves the centre of the distribution *and*
    fattens it, and shifting alone would produce a stressed portfolio that
    looks merely poorer rather than genuinely riskier.  By default
    :math:`\\kappa` is derived from the scenario's volatility shock, so the
    widening is calibrated by the same episode as the shift.

    Feeding the result to :class:`~src.models.risk_metrics.HistoricalRiskEstimator`
    or the Step 2 optimisers gives *stressed* VaR, CVaR and efficient
    frontiers - which is how a stress scenario re-enters the optimisation
    rather than remaining a standalone report.

    Parameters
    ----------
    returns
        Base scenario return matrix, shape ``(q, n)``.
    scenario
        The shock to apply.
    underlyings
        One specification per column of ``returns``, in column order.
    volatility_multiplier
        Override for :math:`\\kappa`.  Defaults to
        ``1 + volatility shock``, floored at 1: a crisis does not compress
        dispersion.

    Returns
    -------
    FloatArray
        Stressed return matrix, shape ``(q, n)``.

    Raises
    ------
    ValueError
        If the number of specifications does not match the column count.
    """
    matrix = _as_2d(returns, "returns")
    if len(underlyings) != matrix.shape[1]:
        raise ValueError(
            f"{len(underlyings)} underlying specifications for "
            f"{matrix.shape[1]} return columns; they must correspond."
        )

    shifts = np.array([spec.spot_return(scenario) for spec in underlyings])
    kappa = (
        max(1.0 + scenario.get(RiskFactor.VOLATILITY), 1.0)
        if volatility_multiplier is None
        else float(volatility_multiplier)
    )
    if kappa < 0.0:
        raise ValueError(
            f"'volatility_multiplier' must be non-negative, got {kappa!r}."
        )

    means = matrix.mean(axis=0)
    return means + shifts + kappa * (matrix - means)


def stressed_risk_estimate(
    weights: npt.ArrayLike,
    returns: npt.ArrayLike,
    scenario: ShockScenario,
    underlyings: Sequence[UnderlyingSpec],
    beta: float = 0.95,
    volatility_multiplier: float | None = None,
) -> tuple[RiskEstimate, RiskEstimate]:
    """VaR and CVaR of a portfolio before and after a stress scenario.

    The direct answer to "what does this optimised portfolio look like in a
    crisis": it takes weights straight from Step 1 or Step 2 and re-runs the
    Step 1 estimator on a stressed distribution.

    Parameters
    ----------
    weights
        Portfolio weights, shape ``(n,)``.
    returns
        Base scenario return matrix, shape ``(q, n)``.
    scenario
        The shock to apply.
    underlyings
        One specification per asset, in column order.
    beta
        Confidence level.
    volatility_multiplier
        Override for the dispersion widening; see
        :func:`stress_return_scenarios`.

    Returns
    -------
    (RiskEstimate, RiskEstimate)
        ``(base, stressed)`` estimates.  Comparing the two gives the
        deterioration in tail risk attributable to the scenario.
    """
    x = _as_1d(weights, "weights")
    base_returns = _as_2d(returns, "returns")
    beta = _validate_beta(beta)

    stressed_returns = stress_return_scenarios(
        base_returns, scenario, underlyings, volatility_multiplier
    )
    base = historical_var_cvar(-(base_returns @ x), beta)
    stressed = historical_var_cvar(-(stressed_returns @ x), beta)
    return base, stressed
