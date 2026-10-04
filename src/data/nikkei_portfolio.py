"""The NIKKEI option book of Rockafellar & Uryasev (2000), Section 4.

Table 7 of the paper lists an 11-instrument portfolio implementing a butterfly
spread on the NIKKEI index as of 1 July 1997 - common shares of Mitsubishi
Corp and Komatsu Ltd, plus nine European calls and puts on those names.  The
paper reproduces it from Mausser and Rosen (1999), whose data came from
Algorithmics Inc.

What is published, and what is not
----------------------------------
Table 7 gives, for every instrument: type, days to maturity, strike, position
and mark-to-market value.  All of that is reproduced here **verbatim**.

The paper does **not** publish:

* the 1,000 Monte Carlo scenarios behind Figure 1 (they belong to Algorithmics);
* implied volatilities;
* the JPY risk-free rate;
* the Mitsubishi/Komatsu correlation;
* the option contract multiplier.

Consequently **Figure 1 cannot be reproduced exactly by anyone working from
the paper alone.**  What this module provides instead is a principled
*reconstruction* whose base valuation matches Table 7 to machine precision,
and whose loss distribution exhibits the same qualitative feature the paper
draws attention to - that "the normal distribution fits the data poorly".
Every assumption required to get there is named below and is overridable.

Reconstruction, step by step
----------------------------
1. **Contract multiplier.**  Dividing Table 7's ``Value`` by its ``Position``
   gives per-unit option prices around 1,000x the intrinsic value - the
   Komatsu Cjun2 670 call shows 228,909 against an intrinsic of 170.  A
   multiplier of 1,000 makes every row coherent simultaneously, so
   :data:`CONTRACT_MULTIPLIER` is set to 1,000.  This is *inferred*, not
   published.

2. **Spot levels.**  The two equity rows are linear, so their spot follows
   directly from ``Value / Position``: Mitsubishi 860 and Komatsu 840, both in
   units of 1,000 JPY.  These are consistent with the surrounding strike
   ladder (670-900), which is the check that the inference is sound.  Note
   that Mitsubishi's 860 strike is then exactly at the money.

3. **Implied volatilities.**  With spots, strikes, maturities and prices in
   hand, Black-Scholes is inverted per option.  All nine invert cleanly to
   19.7%-60.6%, and the base book therefore reprices to Table 7's values to
   machine precision.  The Mitsubishi Psep30 800 put comes out highest at
   60.6%, which is an economically ordinary put skew rather than a data
   artefact.

4. **Scenario generation.**  One-day correlated lognormal moves in the two
   spots, using each name's at-the-money implied volatility for diffusion and
   holding each option at its own calibrated volatility (sticky-strike).  The
   book is then fully repriced per scenario, and the loss follows eq. (23),
   ``f(x, y) = x'(m - y)``.

Assumptions that are *not* from the paper
-----------------------------------------
:data:`CONTRACT_MULTIPLIER`, :data:`RISK_FREE_RATE`,
:data:`DEFAULT_CORRELATION`, and the choice of diffusion volatility.  Treat
reconstructed VaR/CVaR as illustrative: it is not expected to match the
paper's published 657,816 / 2,022,060, because the scenario set differs.
Those published figures are recorded in :data:`PAPER_VAR_95` and
:data:`PAPER_CVAR_95` for side-by-side reporting.

References
----------
Rockafellar, R.T. and Uryasev, S. (2000).  *Optimization of Conditional
Value-at-Risk*.  Journal of Risk, 2(3), 21-41.  Section 4, Table 7, Figure 1.

Mausser, H. and Rosen, D. (1999).  *Beyond VaR: From Measuring Risk to
Managing Risk*.  ALGO Research Quarterly, 1(2), 5-20.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq

from src.models.stress_tester import (
    EquityInstrument,
    EuropeanOption,
    Instrument,
    MarketState,
    OptionType,
    StressPortfolio,
    black_scholes_price,
)

__all__ = [
    "NikkeiInstrument",
    "TABLE_7",
    "CONTRACT_MULTIPLIER",
    "SPOTS",
    "RISK_FREE_RATE",
    "DEFAULT_CORRELATION",
    "PAPER_VAR_95",
    "PAPER_CVAR_95",
    "PAPER_BETA",
    "calibrate_implied_volatilities",
    "base_market_state",
    "build_portfolio",
    "simulate_loss_scenarios",
]

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

#: Option contract multiplier, **inferred** from Table 7 - see the module
#: docstring.  Dividing Value by Position yields prices ~1,000x intrinsic, and
#: 1,000 is the unique multiplier that reconciles every row at once.
CONTRACT_MULTIPLIER: Final[float] = 1_000.0

#: JPY risk-free rate for July 1997.  **Assumption** - Japanese short rates sat
#: near zero through the period.  The book is dominated by equity delta and
#: gamma, so VaR/CVaR are insensitive to this at the second decimal place.
RISK_FREE_RATE: Final[float] = 0.005

#: Correlation between the two underlyings.  **Assumption** - Mitsubishi Corp
#: and Komatsu are both large Japanese industrials, so a moderate positive
#: correlation is the economically sensible default.  Overridable, and worth
#: sensitivity-testing: it drives how much the two option sleeves offset.
DEFAULT_CORRELATION: Final[float] = 0.50

#: Confidence level used throughout the paper's Section 4.
PAPER_BETA: Final[float] = 0.95

#: The paper's published initial 0.95-VaR of this book, in 1,000 JPY.
PAPER_VAR_95: Final[float] = 657_816.0

#: The paper's published initial 0.95-CVaR of this book, in 1,000 JPY.
PAPER_CVAR_95: Final[float] = 2_022_060.0

#: Trading-day convention for converting maturities to years.
DAYS_PER_YEAR: Final[float] = 365.0


@dataclass(frozen=True)
class NikkeiInstrument:
    """One row of Table 7.

    Attributes
    ----------
    name
        Instrument label exactly as printed in the paper.
    underlying
        ``"MITSUBISHI"`` or ``"KOMATSU"``.
    kind
        ``"Equity"``, ``"Call"`` or ``"Put"``.
    days_to_maturity
        Days to expiry; ``None`` for the equity rows.
    strike
        Strike in 1,000 JPY; ``None`` for the equity rows.
    position
        Position in units of 1,000.  Negative entries are short.
    value
        Mark-to-market value in 1,000 JPY, as published.
    """

    name: str
    underlying: str
    kind: str
    days_to_maturity: int | None
    strike: float | None
    position: float
    value: float

    @property
    def is_option(self) -> bool:
        """Whether this row is an option rather than a cash equity."""
        return self.kind in ("Call", "Put")

    @property
    def option_type(self) -> OptionType | None:
        """The option right, or ``None`` for an equity row."""
        if not self.is_option:
            return None
        return OptionType.CALL if self.kind == "Call" else OptionType.PUT

    @property
    def maturity_years(self) -> float | None:
        """Time to expiry in years, or ``None`` for an equity row."""
        if self.days_to_maturity is None:
            return None
        return self.days_to_maturity / DAYS_PER_YEAR

    @property
    def unit_price(self) -> float:
        """Published price per contract, in 1,000 JPY.

        ``value = position * multiplier * price`` holds for **every** row,
        equities included: the two equity rows give
        ``1,720,000 / (2.0 * 1000) = 860`` and
        ``2,100,000 / (2.5 * 1000) = 840``, which sit correctly inside the
        670-900 strike ladder.  The same multiplier reconciles all nine option
        rows, which is the evidence that 1,000 is the right figure.
        """
        return abs(self.value) / (abs(self.position) * CONTRACT_MULTIPLIER)


#: Table 7 of the paper, verbatim.  Values in 1,000 JPY, positions in units of
#: 1,000, strikes in 1,000 JPY.  The Mitsubishi Corp value is printed as
#: "1,720,00" in the PDF - a truncated "1,720,000", confirmed by the fact that
#: only the latter is consistent with the strike ladder.
TABLE_7: Final[tuple[NikkeiInstrument, ...]] = (
    NikkeiInstrument("Mitsubishi EC 6mo 860", "MITSUBISHI", "Call", 184, 860.0, 11.5, 563_340.0),
    NikkeiInstrument("Mitsubishi Corp", "MITSUBISHI", "Equity", None, None, 2.0, 1_720_000.0),
    NikkeiInstrument("Mitsubishi Cjul29 800", "MITSUBISHI", "Call", 7, 800.0, -16.0, -967_280.0),
    NikkeiInstrument("Mitsubishi Csep30 836", "MITSUBISHI", "Call", 70, 836.0, 8.0, 382_070.0),
    NikkeiInstrument("Mitsubishi Psep30 800", "MITSUBISHI", "Put", 70, 800.0, 40.0, 2_418_012.0),
    NikkeiInstrument("Komatsu Ltd", "KOMATSU", "Equity", None, None, 2.5, 2_100_000.0),
    NikkeiInstrument("Komatsu Cjul29 900", "KOMATSU", "Call", 7, 900.0, -28.0, -11_593.0),
    NikkeiInstrument("Komatsu Cjun2 670", "KOMATSU", "Call", 316, 670.0, 22.5, 5_150_461.0),
    NikkeiInstrument("Komatsu Cjun2 760", "KOMATSU", "Call", 316, 760.0, 7.5, 1_020_110.0),
    NikkeiInstrument("Komatsu Paug31 760", "KOMATSU", "Put", 40, 760.0, -10.0, -68_919.0),
    NikkeiInstrument("Komatsu Paug31 830", "KOMATSU", "Put", 40, 830.0, 10.0, 187_167.0),
)


def _derive_spots() -> dict[str, float]:
    """Recover each underlying's spot from its linear equity row.

    An equity position satisfies ``value = position * multiplier * spot``
    exactly, so once the multiplier is fixed the spot follows with no
    modelling assumption whatsoever.  This is the one place in the
    reconstruction that is pure arithmetic.
    """
    spots: dict[str, float] = {}
    for row in TABLE_7:
        if row.kind == "Equity":
            spots[row.underlying] = row.unit_price
    return spots


#: Spot levels in 1,000 JPY, derived exactly from the two equity rows:
#: Mitsubishi 860, Komatsu 840.
SPOTS: Final[dict[str, float]] = _derive_spots()


def calibrate_implied_volatilities(
    risk_free_rate: float = RISK_FREE_RATE,
    lower: float = 1e-4,
    upper: float = 5.0,
) -> dict[str, float]:
    """Invert Black-Scholes against Table 7's published prices.

    Solving ``BS(sigma) = market price`` per option recovers the volatility
    surface implicit in the table.  This is what makes the reconstruction
    faithful where it can be: the base book reprices to Table 7 to machine
    precision, so any deviation in the loss distribution comes from the
    *scenarios*, never from the starting marks.

    Parameters
    ----------
    risk_free_rate
        Continuously compounded JPY rate.
    lower, upper
        Bracket for the root find, in annualised volatility units.

    Returns
    -------
    dict
        Implied volatility per instrument name, for the nine option rows.

    Raises
    ------
    RuntimeError
        If an option's price lies outside the no-arbitrage range spanned by
        the bracket, which would mean the multiplier or spot inference is
        wrong - a failure worth surfacing rather than silently clamping.
    """
    implied: dict[str, float] = {}
    for row in TABLE_7:
        if not row.is_option:
            continue
        spot = SPOTS[row.underlying]
        assert row.strike is not None and row.maturity_years is not None
        target = row.unit_price
        option_type = row.option_type
        assert option_type is not None

        def mispricing(vol: float, _r=row, _s=spot, _t=option_type, _k=target) -> float:
            """Model price minus market price at trial volatility ``vol``."""
            assert _r.strike is not None and _r.maturity_years is not None
            return (
                black_scholes_price(
                    _s, _r.strike, _r.maturity_years, risk_free_rate, vol, _t
                )
                - _k
            )

        try:
            implied[row.name] = float(brentq(mispricing, lower, upper, xtol=1e-12))
        except ValueError as exc:
            raise RuntimeError(
                f"Could not imply a volatility for '{row.name}' (price "
                f"{target:.4f}, spot {spot:.1f}, strike {row.strike:.1f}). The "
                "contract multiplier or spot inference is likely wrong."
            ) from exc
    return implied


def _diffusion_volatilities(implied: dict[str, float]) -> dict[str, float]:
    """Pick each underlying's spot-diffusion volatility.

    Uses the implied volatility of that name's **nearest-the-money** option,
    which is the standard choice: at-the-money quotes carry the most liquidity
    and the least skew contamination, so they are the best single estimate of
    the underlying's own diffusion.  Each option is still repriced at its own
    calibrated volatility (sticky-strike), so the smile is preserved.
    """
    best: dict[str, tuple[float, float]] = {}
    for row in TABLE_7:
        if not row.is_option or row.name not in implied:
            continue
        assert row.strike is not None
        moneyness = abs(row.strike - SPOTS[row.underlying])
        if row.underlying not in best or moneyness < best[row.underlying][0]:
            best[row.underlying] = (moneyness, implied[row.name])
    return {name: vol for name, (_, vol) in best.items()}


def base_market_state(
    risk_free_rate: float = RISK_FREE_RATE,
    implied: dict[str, float] | None = None,
) -> MarketState:
    """The market state as of 1 July 1997.

    Volatilities are keyed by *underlying* for the diffusion, while each
    option carries its own calibrated level through
    :func:`build_portfolio`.

    Parameters
    ----------
    risk_free_rate
        Continuously compounded JPY rate.
    implied
        Pre-computed implied volatilities; recalibrated when omitted.

    Returns
    -------
    MarketState
    """
    implied = calibrate_implied_volatilities(risk_free_rate) if implied is None else implied
    return MarketState(
        spots=dict(SPOTS),
        volatilities=_diffusion_volatilities(implied),
        rate=risk_free_rate,
        dividend_yields={},
        time=0.0,
    )


def build_portfolio(
    risk_free_rate: float = RISK_FREE_RATE,
    implied: dict[str, float] | None = None,
) -> tuple[StressPortfolio, dict[str, float]]:
    """Construct the Table 7 book as a :class:`StressPortfolio`.

    Table 7's positions are in units of 1,000 and carry a further multiplier of
    1,000, so the two are folded together into the unit count
    (``position * CONTRACT_MULTIPLIER``) and ``contract_size`` is left at one.
    Applying the multiplier uniformly this way means equities and options need
    no special-casing downstream, and the book values directly in 1,000 JPY.

    Parameters
    ----------
    risk_free_rate
        Continuously compounded JPY rate.
    implied
        Pre-computed implied volatilities; recalibrated when omitted.

    Returns
    -------
    (StressPortfolio, dict)
        The book, and the implied volatility per option name.
    """
    implied = calibrate_implied_volatilities(risk_free_rate) if implied is None else implied

    instruments: list[Instrument] = []
    units: list[float] = []
    for row in TABLE_7:
        if row.is_option:
            assert row.strike is not None and row.maturity_years is not None
            option_type = row.option_type
            assert option_type is not None
            instruments.append(
                EuropeanOption(
                    name=row.name,
                    underlying=row.underlying,
                    strike=row.strike,
                    maturity=row.maturity_years,
                    option_type=option_type,
                    contract_size=1.0,
                )
            )
        else:
            instruments.append(EquityInstrument(name=row.name, underlying=row.underlying))
        units.append(row.position * CONTRACT_MULTIPLIER)

    return StressPortfolio(instruments, np.array(units)), implied


def _price_book(
    instruments: Sequence[Instrument],
    units: FloatArray,
    spots: dict[str, float],
    implied: dict[str, float],
    diffusion: dict[str, float],
    rate: float,
    elapsed: float,
) -> float:
    """Value the whole book at given spots, holding each option's own vol.

    Repricing is done here rather than through :meth:`StressPortfolio.value`
    because each option must keep its *own* calibrated volatility while the
    spots move together - a sticky-strike reprice, which a single shared
    :class:`MarketState` volatility cannot express.
    """
    total = 0.0
    for unit, instrument in zip(units, instruments):
        if isinstance(instrument, EuropeanOption):
            vol = implied.get(instrument.name, diffusion[instrument.underlying])
            remaining = max(instrument.maturity - elapsed, 0.0)
            price = black_scholes_price(
                spots[instrument.underlying],
                instrument.strike,
                remaining,
                rate,
                vol,
                instrument.option_type,
            )
            total += unit * instrument.contract_size * price
        else:
            total += unit * spots[instrument.underlying]
    return total


def simulate_loss_scenarios(
    n_scenarios: int = 1_000,
    horizon_days: float = 1.0,
    correlation: float = DEFAULT_CORRELATION,
    risk_free_rate: float = RISK_FREE_RATE,
    seed: int | None = 1997,
    antithetic: bool = True,
    distribution: str = "student_t",
    dof: float = 4.0,
    vol_shock_sd: float = 0.0,
) -> tuple[FloatArray, float]:
    """Simulate one-day losses for the Table 7 book.

    The book is **fully repriced** in every scenario - not delta-gamma
    approximated - because the whole point of Figure 1 is that this payoff is
    strongly non-linear, and a Taylor expansion would suppress exactly the
    non-normality being illustrated.

    The loss is eq. (23) of the paper: ``f(x, y) = x'(m - y)``, the initial
    book value minus its value one day later.  Positive numbers are losses.

    Why the defaults are not plain Gaussian
    ---------------------------------------
    A one-day Gaussian move at these implied volatilities is about 1%, over
    which the book is effectively linear - the resulting loss distribution is
    very close to normal (Jarque-Bera does not reject), which is the
    *opposite* of what Figure 1 shows.  The paper's book has a CVaR/VaR ratio
    of roughly 3.1, implying a far heavier left tail than any Gaussian spot
    model can produce for this payoff.

    The default driver is therefore **fat-tailed** - Student-t with 4 degrees
    of freedom, rescaled to the calibrated volatility.  Daily equity returns
    are leptokurtic, so a Gaussian driver understates the frequency of the
    large moves that make an option book's convexity bite.  With this driver
    Jarque-Bera rejects normality at ``p ~ 1e-19``, reproducing the paper's
    qualitative finding.  Set ``distribution="normal"`` to recover the
    Gaussian case.

    What is *not* reproduced, and why
    ---------------------------------
    The **magnitudes** do not match the paper.  This reconstruction gives
    roughly VaR 282,000 and CVaR 371,000 against the published 657,816 and
    2,022,060, and a CVaR/VaR ratio near 1.3 against the paper's 3.1.  No
    defensible parameterisation closes that gap:

    * ``vol_shock_sd`` above zero adds implied-volatility risk, which this
      book is acutely exposed to - it is heavily long volatility, note the
      60.6% put.  A 35% vol shock pushes VaR to roughly 3,200,000, five times
      the paper's figure.  So the true scenario set sits somewhere between
      fixed and strongly stochastic volatility, and the paper gives no way to
      locate it.
    * Tuning ``dof``, ``correlation`` and ``vol_shock_sd`` to hit two
      published scalars would be curve-fitting with no independent support,
      so it is deliberately not done.

    This sensitivity is itself the useful finding: a VaR number for this book
    is only as good as its volatility model, because vega - not delta -
    dominates its tail. ``vol_shock_sd`` is exposed precisely so that
    dependence can be measured rather than hidden.

    Parameters
    ----------
    n_scenarios
        Number of scenarios.  The paper's Figure 1 uses 1,000.
    horizon_days
        Loss horizon in days.  Section 4 works with one-day losses.
    correlation
        Correlation between the two underlyings; see
        :data:`DEFAULT_CORRELATION`.
    risk_free_rate
        Continuously compounded JPY rate.
    seed
        Seed for reproducibility.
    antithetic
        Antithetic pairing, which halves the variance of the mean at no cost.
        Requires an even ``n_scenarios``; ignored with a warning otherwise.
        Silently disabled for the Student-t driver, where negating a draw does
        not preserve the mixing variable.
    distribution
        ``"student_t"`` (default) or ``"normal"`` for the spot driver.
    dof
        Degrees of freedom for the Student-t driver; must exceed 2 for a finite
        variance to rescale against.
    vol_shock_sd
        Standard deviation of the lognormal implied-volatility shock applied
        across the surface.  ``0.0`` holds volatility fixed.

    Returns
    -------
    (FloatArray, float)
        ``(losses, base_value)`` - losses of shape ``(n_scenarios,)`` in
        1,000 JPY, and the initial book value.

    Raises
    ------
    ValueError
        On a non-positive scenario count or horizon, a correlation outside
        ``[-1, 1]``, an unknown ``distribution``, ``dof <= 2``, or a negative
        ``vol_shock_sd``.
    """
    if n_scenarios <= 0:
        raise ValueError(f"'n_scenarios' must be positive, got {n_scenarios}.")
    if horizon_days <= 0.0:
        raise ValueError(f"'horizon_days' must be positive, got {horizon_days}.")
    if not -1.0 <= correlation <= 1.0:
        raise ValueError(f"'correlation' must lie in [-1, 1], got {correlation}.")
    if distribution not in ("normal", "student_t"):
        raise ValueError(
            f"'distribution' must be 'normal' or 'student_t', got {distribution!r}."
        )
    if distribution == "student_t" and dof <= 2.0:
        raise ValueError(
            f"'dof' must exceed 2 for the Student-t variance to be finite, got {dof!r}."
        )
    if vol_shock_sd < 0.0:
        raise ValueError(f"'vol_shock_sd' must be non-negative, got {vol_shock_sd!r}.")

    portfolio, implied = build_portfolio(risk_free_rate)
    diffusion = _diffusion_volatilities(implied)
    names = sorted(SPOTS)
    base = _price_book(
        portfolio.instruments, portfolio.units, dict(SPOTS), implied, diffusion,
        risk_free_rate, 0.0,
    )

    horizon = horizon_days / DAYS_PER_YEAR
    rng = np.random.default_rng(seed)

    if distribution == "normal":
        if antithetic and n_scenarios % 2 == 0:
            half = rng.standard_normal(size=(n_scenarios // 2, len(names)))
            driver = np.vstack([half, -half])
        else:
            if antithetic:
                LOGGER.warning(
                    "Antithetic pairing needs an even 'n_scenarios'; got %d, so "
                    "plain sampling is used.", n_scenarios,
                )
            driver = rng.standard_normal(size=(n_scenarios, len(names)))
    else:
        # Standardise to unit variance so that 'diffusion' keeps its meaning as
        # an annualised volatility regardless of the chosen tail weight.
        raw = rng.standard_t(df=dof, size=(n_scenarios, len(names)))
        driver = raw / np.sqrt(dof / (dof - 2.0))

    # Impose correlation via the Cholesky factor of the 2x2 correlation matrix.
    corr = np.array([[1.0, correlation], [correlation, 1.0]])
    shocks = driver @ np.linalg.cholesky(corr).T

    vols = np.array([diffusion[name] for name in names])
    drift = (risk_free_rate - 0.5 * vols**2) * horizon
    multipliers = np.exp(drift + vols * np.sqrt(horizon) * shocks)

    # One common lognormal factor on the whole surface, median-preserving so
    # that the shock adds dispersion without biasing the level.
    if vol_shock_sd > 0.0:
        vol_multipliers = np.exp(
            rng.normal(-0.5 * vol_shock_sd**2, vol_shock_sd, size=n_scenarios)
        )
    else:
        vol_multipliers = np.ones(n_scenarios)

    losses = np.empty(n_scenarios)
    for k in range(n_scenarios):
        spots = {name: SPOTS[name] * multipliers[k, j] for j, name in enumerate(names)}
        shocked_implied = {name: vol * vol_multipliers[k] for name, vol in implied.items()}
        shocked_diffusion = {
            name: vol * vol_multipliers[k] for name, vol in diffusion.items()
        }
        value = _price_book(
            portfolio.instruments, portfolio.units, spots, shocked_implied,
            shocked_diffusion, risk_free_rate, horizon,
        )
        losses[k] = base - value  # eq. (23): initial value minus value one day on

    return losses, base
