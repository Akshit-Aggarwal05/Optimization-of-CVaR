"""Command-line interface for the CVaR portfolio risk engine.

The engine deliberately does **not** fetch arbitrary live tickers.  Its purpose
is to reproduce and extend Rockafellar & Uryasev (2000), so the CLI exposes the
paper's own experiments as named commands:

``paper-portfolio``
    Section 3 - the three-instrument portfolio (S&P 500, long-term US
    government bonds, small-cap stocks) optimised by the Rockafellar-Uryasev
    linear programme, with a side-by-side comparison against Tables 3, 4 and 6.

``nikkei``
    Section 4 - the eleven-instrument NIKKEI option book of Table 7, priced and
    risked, with the best normal approximation fitted to its one-day loss
    distribution as in Figure 1.

``validate``
    The full paper regression in one command, with a non-zero exit status if
    any published figure is missed.  Suitable for CI.

``figures``
    Regenerate the figures in ``docs/images/``.

Run ``cvar-engine <command> --help`` for the options of each.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Final, Sequence

import numpy as np

__all__ = ["main", "build_parser"]

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

#: Width of the report rules, chosen to fit a standard 80-column terminal.
_RULE: Final[int] = 78


def _heading(text: str) -> None:
    """Print a section heading bracketed by rules."""
    print("=" * _RULE)
    print(text)
    print("=" * _RULE)


def _percentage_difference(actual: float, reference: float) -> float:
    """Signed percentage deviation of ``actual`` from ``reference``."""
    return 100.0 * (actual - reference) / reference if reference else float("nan")


# ==============================================================================
# Section 3 - the three-instrument portfolio
# ==============================================================================


def run_paper_portfolio(arguments: argparse.Namespace) -> int:
    """Optimise the paper's three-instrument portfolio and report against it.

    Minimises beta-CVaR subject to the budget and no-short-selling conditions
    of eq. (11) and the expected-return floor of eq. (15), then compares the
    result with the published Tables 3, 4 and 6.

    Parameters
    ----------
    arguments
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit status; ``0`` on success.
    """
    from src.data.paper_datasets import (
        COVARIANCE, INSTRUMENTS, MEAN_RETURNS, MIN_VARIANCE_WEIGHTS,
        REFERENCE_CVAR, REFERENCE_VAR, sample_normal_returns,
    )
    from src.models.risk_metrics import (
        CVaRLinearProgram, ParametricRiskEstimator, PortfolioConstraints,
    )

    beta = arguments.beta
    _heading(
        f"Rockafellar-Uryasev minimum-CVaR portfolio  |  beta = {beta:.2f}  |  "
        f"R = {arguments.target_return}"
    )
    print(f"Instruments : {', '.join(INSTRUMENTS)}")
    print(f"Scenarios   : {arguments.scenarios:,} ({arguments.engine})")
    print(f"Constraints : sum(x) = 1, x >= 0 (eq. 11); -x'm <= -R (eq. 15)")
    print()

    returns = sample_normal_returns(
        arguments.scenarios, engine=arguments.engine, seed=arguments.seed
    )
    constraints = PortfolioConstraints(
        budget=1.0, lower_bounds=0.0, min_expected_return=arguments.target_return
    )
    solution = CVaRLinearProgram.from_returns(
        returns, beta, constraints, expected_returns=MEAN_RETURNS
    ).solve()

    print("OPTIMAL PORTFOLIO")
    for name, weight in zip(INSTRUMENTS, solution.weights):
        print(f"  {name:<14} {weight: .6f}")
    print(f"  {'sum':<14} {solution.weights.sum(): .6f}")
    print()
    print(f"  {'beta-VaR':<14} {solution.reported_var: .6f}")
    print(f"  {'beta-CVaR':<14} {solution.cvar: .6f}")
    print(f"  {'E[return]':<14} {solution.expected_return: .6f}")
    print(f"  solver: {solution.message} ({solution.n_iterations} iterations)")
    print()

    if arguments.compare:
        _heading("COMPARISON WITH THE PAPER")
        print("Table 3 - minimum-variance benchmark (Markowitz, problem P3)")
        print(f"  {'':<14}{'paper':>12}{'this run':>12}{'abs diff':>12}")
        for name, published, obtained in zip(
            INSTRUMENTS, MIN_VARIANCE_WEIGHTS, solution.weights
        ):
            print(
                f"  {name:<14}{published:>12.6f}{obtained:>12.6f}"
                f"{abs(published - obtained):>12.2e}"
            )
        print()

        if beta in REFERENCE_VAR:
            analytic = ParametricRiskEstimator(MEAN_RETURNS, COVARIANCE).estimate(
                MIN_VARIANCE_WEIGHTS, beta
            )
            print("Table 4 - VaR/CVaR of the benchmark, eqs. (18)-(19)")
            print(f"  {'':<14}{'paper':>12}{'analytic':>12}{'LP':>12}{'LP dev %':>11}")
            print(
                f"  {'VaR':<14}{REFERENCE_VAR[beta]:>12.6f}{analytic.var:>12.6f}"
                f"{solution.reported_var:>12.6f}"
                f"{_percentage_difference(solution.reported_var, REFERENCE_VAR[beta]):>11.2f}"
            )
            print(
                f"  {'CVaR':<14}{REFERENCE_CVAR[beta]:>12.6f}{analytic.cvar:>12.6f}"
                f"{solution.cvar:>12.6f}"
                f"{_percentage_difference(solution.cvar, REFERENCE_CVAR[beta]):>11.2f}"
            )
            print()
            print(
                "  The Proposition of Section 3 guarantees these coincide under"
                "\n  normality when eq. (15) is active, so Table 4 is an exact"
                "\n  analytic benchmark rather than a second estimate."
            )
        else:
            print(
                f"  No published Table 4 row for beta = {beta:.2f}; "
                "the paper reports 0.90, 0.95 and 0.99."
            )

    if arguments.json:
        payload: dict[str, Any] = {
            "beta": beta,
            "scenarios": int(returns.shape[0]),
            "engine": arguments.engine,
            "target_return": arguments.target_return,
            "instruments": list(INSTRUMENTS),
            "weights": solution.weights.tolist(),
            "var": solution.reported_var,
            "cvar": solution.cvar,
            "expected_return": solution.expected_return,
        }
        print()
        print(json.dumps(payload, indent=2))
    return 0


# ==============================================================================
# Section 4 - the NIKKEI option book
# ==============================================================================


def run_nikkei(arguments: argparse.Namespace) -> int:
    """Price and risk the Table 7 NIKKEI book, fitting the normal approximation.

    Reproduces the construction behind Figure 1: the eleven-instrument option
    book is repriced over Monte Carlo scenarios, and a maximum-likelihood
    normal is fitted to the resulting one-day losses so the quality of that fit
    can be judged numerically rather than by eye.

    Parameters
    ----------
    arguments
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit status; ``0`` on success.
    """
    from scipy.stats import jarque_bera, kurtosis, norm, skew

    from src.data.nikkei_portfolio import (
        CONTRACT_MULTIPLIER, PAPER_CVAR_95, PAPER_VAR_95, RISK_FREE_RATE, SPOTS,
        TABLE_7, build_portfolio, calibrate_implied_volatilities,
        simulate_loss_scenarios,
    )
    from src.models.risk_metrics import historical_var_cvar

    _heading("NIKKEI option book (paper Table 7, 1 July 1997)")
    print(
        "RECONSTRUCTION. The portfolio, strikes, maturities, positions and marks\n"
        "are the paper's. The 1,000 Monte Carlo scenarios behind Figure 1 are NOT\n"
        "published - they belonged to Algorithmics - so they are simulated here.\n"
        "Magnitudes will not match the paper's published VaR/CVaR; see below."
    )
    print()

    implied = calibrate_implied_volatilities(arguments.rate)
    book, _ = build_portfolio(arguments.rate)

    print("CALIBRATION")
    print(f"  contract multiplier : {CONTRACT_MULTIPLIER:,.0f}  (inferred from Table 7)")
    print(f"  risk-free rate      : {arguments.rate:.4f}  (assumption)")
    for name, spot in sorted(SPOTS.items()):
        print(f"  spot {name:<12}   : {spot:,.2f}  (exact, from the equity row)")
    print()
    print("  implied volatilities, inverted from Table 7's published values:")
    for row in TABLE_7:
        if row.name in implied:
            print(f"    {row.name:<24} {implied[row.name]:.4f}")
    print()

    published = sum(row.value for row in TABLE_7)
    losses, base_value = simulate_loss_scenarios(
        n_scenarios=arguments.scenarios,
        correlation=arguments.correlation,
        risk_free_rate=arguments.rate,
        seed=arguments.seed,
        distribution=arguments.distribution,
        dof=arguments.dof,
        vol_shock_sd=arguments.vol_shock_sd,
    )
    print("BASE VALUATION")
    print(f"  reconstructed : {base_value:>18,.2f}  (1,000 JPY)")
    print(f"  Table 7 sum   : {published:>18,.2f}")
    print(f"  residual      : {base_value - published:>18.3e}")
    print()

    estimate = historical_var_cvar(losses, arguments.beta)
    mean, sd = float(losses.mean()), float(losses.std(ddof=1))
    test = jarque_bera(losses)

    _heading(f"ONE-DAY LOSS DISTRIBUTION  ({arguments.scenarios:,} scenarios)")
    print(f"  driver        : {arguments.distribution}"
          + (f" (dof = {arguments.dof:g})" if arguments.distribution == "student_t" else "")
          + f", vol shock sd = {arguments.vol_shock_sd:g}")
    print(f"  mean          : {mean:>18,.0f}")
    print(f"  st. deviation : {sd:>18,.0f}")
    print(f"  skewness      : {float(skew(losses)):>18.4f}")
    print(f"  excess kurt.  : {float(kurtosis(losses)):>18.4f}")
    print()
    print(f"  {'':<16}{'this run':>16}{'paper':>16}{'ratio':>10}")
    print(
        f"  {'VaR  ' + f'{arguments.beta:.2f}':<16}{estimate.var:>16,.0f}"
        f"{PAPER_VAR_95:>16,.0f}{estimate.var / PAPER_VAR_95:>10.2f}"
    )
    print(
        f"  {'CVaR ' + f'{arguments.beta:.2f}':<16}{estimate.cvar:>16,.0f}"
        f"{PAPER_CVAR_95:>16,.0f}{estimate.cvar / PAPER_CVAR_95:>10.2f}"
    )
    print(
        f"  {'CVaR / VaR':<16}{estimate.cvar / estimate.var:>16.2f}"
        f"{PAPER_CVAR_95 / PAPER_VAR_95:>16.2f}"
    )
    print()

    _heading("BEST NORMAL APPROXIMATION (Figure 1)")
    print(f"  fitted N(mu, sigma) : mu = {mean:,.0f}, sigma = {sd:,.0f}")
    normal_var = mean + sd * float(norm.ppf(arguments.beta))
    normal_cvar = mean + sd * float(norm.pdf(norm.ppf(arguments.beta))) / (
        1.0 - arguments.beta
    )
    print(f"  {'':<16}{'empirical':>16}{'normal fit':>16}{'error %':>10}")
    print(
        f"  {'VaR':<16}{estimate.var:>16,.0f}{normal_var:>16,.0f}"
        f"{_percentage_difference(normal_var, estimate.var):>10.1f}"
    )
    print(
        f"  {'CVaR':<16}{estimate.cvar:>16,.0f}{normal_cvar:>16,.0f}"
        f"{_percentage_difference(normal_cvar, estimate.cvar):>10.1f}"
    )
    print()
    print(f"  Jarque-Bera statistic {float(test.statistic):,.1f}, p = {test.pvalue:.3e}")
    verdict = (
        "REJECTED - the normal distribution fits the data poorly, as the paper reports"
        if test.pvalue < 0.01
        else "not rejected at the 1% level"
    )
    print(f"  Normality: {verdict}")

    if arguments.plot:
        from src.visualization import plot_nikkei_loss_distribution

        print()
        path = plot_nikkei_loss_distribution(
            n_scenarios=arguments.scenarios,
            seed=arguments.seed,
            distribution=arguments.distribution,
            dof=arguments.dof,
            vol_shock_sd=arguments.vol_shock_sd,
            correlation=arguments.correlation,
            risk_free_rate=arguments.rate,
        )
        print(f"  figure written to {path}")

    if arguments.json:
        print()
        print(json.dumps({
            "base_value": base_value,
            "table_7_sum": published,
            "scenarios": arguments.scenarios,
            "beta": arguments.beta,
            "var": estimate.var,
            "cvar": estimate.cvar,
            "paper_var": PAPER_VAR_95,
            "paper_cvar": PAPER_CVAR_95,
            "mean": mean,
            "sd": sd,
            "jarque_bera_p": float(test.pvalue),
            "implied_volatilities": implied,
        }, indent=2))
    return 0


# ==============================================================================
# Validation
# ==============================================================================


def run_validate(arguments: argparse.Namespace) -> int:
    """Check every published figure the engine claims to reproduce.

    Returns a non-zero exit status if any check fails, so the command can be
    wired straight into CI as a regression gate on the paper's results.

    Parameters
    ----------
    arguments
        Parsed CLI arguments.

    Returns
    -------
    int
        ``0`` if every check passes, ``1`` otherwise.
    """
    from src.data.paper_datasets import (
        COVARIANCE, MEAN_RETURNS, MIN_VARIANCE_VARIANCE, MIN_VARIANCE_WEIGHTS,
        REFERENCE_CVAR, REFERENCE_VAR, TARGET_RETURN, sample_normal_returns,
    )
    from src.models.risk_metrics import (
        CVaRLinearProgram, ParametricRiskEstimator, PortfolioConstraints,
    )

    _heading("PAPER REGRESSION")
    checks: list[tuple[str, bool, str]] = []

    engine = ParametricRiskEstimator(MEAN_RETURNS, COVARIANCE)
    variance = engine.volatility(MIN_VARIANCE_WEIGHTS) ** 2
    ok = abs(variance - MIN_VARIANCE_VARIANCE) < 1e-8
    checks.append((
        "Table 3 variance", ok,
        f"{variance:.8f} vs {MIN_VARIANCE_VARIANCE:.8f}",
    ))

    for beta in (0.90, 0.95, 0.99):
        estimate = engine.estimate(MIN_VARIANCE_WEIGHTS, beta)
        for label, obtained, published in (
            ("VaR", estimate.var, REFERENCE_VAR[beta]),
            ("CVaR", estimate.cvar, REFERENCE_CVAR[beta]),
        ):
            ok = abs(obtained - published) < 1e-5
            checks.append((
                f"Table 4 {label} beta={beta:.2f}", ok,
                f"{obtained:.6f} vs {published:.6f}",
            ))

    constraints = PortfolioConstraints(
        budget=1.0, lower_bounds=0.0, min_expected_return=TARGET_RETURN
    )
    for beta in (0.90, 0.95, 0.99):
        returns = sample_normal_returns(
            arguments.scenarios, engine="sobol", seed=20_000
        )
        solution = CVaRLinearProgram.from_returns(
            returns, beta, constraints, expected_returns=MEAN_RETURNS
        ).solve()
        deviation = abs(
            _percentage_difference(solution.cvar, REFERENCE_CVAR[beta])
        )
        ok = deviation < arguments.tolerance
        checks.append((
            f"Table 6 CVaR beta={beta:.2f}", ok,
            f"deviation {deviation:.3f}% (limit {arguments.tolerance}%)",
        ))

    width = max(len(name) for name, _, _ in checks)
    failures = 0
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        failures += 0 if ok else 1
        print(f"  [{status}] {name:<{width}}  {detail}")

    print()
    if failures:
        print(f"{failures} of {len(checks)} checks FAILED.")
        return 1
    print(f"All {len(checks)} checks passed.")
    return 0


# ==============================================================================
# Figures
# ==============================================================================


def run_figures(arguments: argparse.Namespace) -> int:
    """Regenerate the project's figures.

    Parameters
    ----------
    arguments
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit status.
    """
    from src.visualization import generate_all

    _heading("FIGURE GENERATION")
    written = generate_all(only=arguments.only, output_dir=arguments.output_dir)
    print()
    for path in written:
        print(f"  {path}")
    return 0


# ==============================================================================
# Parser
# ==============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="cvar-engine",
        description=(
            "Portfolio risk engine implementing Rockafellar & Uryasev (2000). "
            "Commands reproduce the paper's own experiments rather than taking "
            "arbitrary tickers."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  cvar-engine paper-portfolio --beta 0.95 --compare\n"
            "  cvar-engine nikkei --scenarios 1000 --plot\n"
            "  cvar-engine validate\n"
            "  cvar-engine figures --only nikkei-losses\n"
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable INFO-level logging"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- Section 3 ------------------------------------------------------------
    section3 = subparsers.add_parser(
        "paper-portfolio",
        help="Section 3: S&P 500 / Gov Bond / Small Cap minimum-CVaR portfolio",
        description=(
            "Minimise beta-CVaR for the paper's three-instrument portfolio "
            "subject to eq. (11) and eq. (15)."
        ),
    )
    section3.add_argument("--beta", type=float, default=0.95,
                          help="confidence level (default: 0.95)")
    section3.add_argument("--scenarios", type=int, default=16_384,
                          help="scenario count (default: 16384)")
    section3.add_argument("--engine", choices=("sobol", "pseudo"), default="sobol",
                          help="sampling engine (default: sobol)")
    section3.add_argument("--target-return", type=float, default=0.011,
                          help="R in eq. (15) (default: 0.011, the paper's value)")
    section3.add_argument("--seed", type=int, default=20_000, help="random seed")
    section3.add_argument("--compare", action="store_true", default=True,
                          help="compare against Tables 3 and 4 (default: on)")
    section3.add_argument("--no-compare", dest="compare", action="store_false",
                          help="suppress the comparison tables")
    section3.add_argument("--json", action="store_true", help="also emit JSON")
    section3.set_defaults(handler=run_paper_portfolio)

    # -- Section 4 ------------------------------------------------------------
    section4 = subparsers.add_parser(
        "nikkei",
        help="Section 4: NIKKEI option book with the best normal approximation",
        description=(
            "Price and risk the Table 7 NIKKEI option book, fitting a normal "
            "approximation to the one-day loss distribution as in Figure 1. "
            "The scenarios are a reconstruction; the originals are unpublished."
        ),
    )
    section4.add_argument("--scenarios", type=int, default=1_000,
                          help="scenario count (default: 1000, as in Figure 1)")
    section4.add_argument("--beta", type=float, default=0.95,
                          help="confidence level (default: 0.95)")
    section4.add_argument("--distribution", choices=("student_t", "normal"),
                          default="student_t",
                          help="spot return driver (default: student_t)")
    section4.add_argument("--dof", type=float, default=4.0,
                          help="Student-t degrees of freedom (default: 4)")
    section4.add_argument("--vol-shock-sd", type=float, default=0.0,
                          help="lognormal implied-vol shock sd (default: 0, fixed vol)")
    section4.add_argument("--correlation", type=float, default=0.50,
                          help="Mitsubishi/Komatsu correlation (default: 0.5)")
    section4.add_argument("--rate", type=float, default=0.005,
                          help="JPY risk-free rate (default: 0.005)")
    section4.add_argument("--seed", type=int, default=1997, help="random seed")
    section4.add_argument("--plot", action="store_true",
                          help="also write the Figure 1 reconstruction")
    section4.add_argument("--json", action="store_true", help="also emit JSON")
    section4.set_defaults(handler=run_nikkei)

    # -- Validation -----------------------------------------------------------
    validate = subparsers.add_parser(
        "validate",
        help="run the full paper regression; non-zero exit on failure",
        description="Check every published figure the engine claims to reproduce.",
    )
    validate.add_argument("--scenarios", type=int, default=16_384,
                          help="scenario count for the LP checks (default: 16384)")
    validate.add_argument("--tolerance", type=float, default=1.0,
                          help="max CVaR deviation in %% for Table 6 (default: 1.0)")
    validate.set_defaults(handler=run_validate)

    # -- Figures --------------------------------------------------------------
    figures = subparsers.add_parser(
        "figures", help="regenerate the figures in docs/images/",
        description="Regenerate project figures.",
    )
    figures.add_argument("--only", nargs="+", metavar="FIGURE",
                         help="generate only these figures")
    figures.add_argument("--output-dir", default=None,
                         help="destination directory (default: docs/images)")
    figures.set_defaults(handler=run_figures)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``cvar-engine`` console script.

    Parameters
    ----------
    argv
        Argument list; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status.
    """
    parser = build_parser()
    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if arguments.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(arguments.handler(arguments))
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
