"""Figure generation for the README and for reports.

Every function writes a PNG into ``docs/images/`` and returns its path, so the
module doubles as the script that populates the README's figure placeholders.
Run it as a whole with::

    python -m src.visualization

or one figure at a time through the CLI::

    cvar-engine figures --only nikkei-losses

The headline figure is :func:`plot_nikkei_loss_distribution`, which recreates
Figure 1 of Rockafellar & Uryasev (2000) - the distribution of one-day losses
on the NIKKEI option book with the best normal approximation overlaid.  Read
the docstring there and the one on
:mod:`src.data.nikkei_portfolio` before quoting its numbers: the portfolio and
its marks are the paper's, but the scenarios are a reconstruction, because the
original 1,000 draws are not published.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import numpy.typing as npt

__all__ = [
    "IMAGE_DIR",
    "plot_nikkei_loss_distribution",
    "plot_efficient_frontier",
    "plot_pareto_front",
    "plot_weight_distribution",
    "plot_nsga2_convergence",
    "plot_stress_attribution",
    "generate_all",
    "main",
]

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

#: Output directory for every figure, relative to the repository root.
IMAGE_DIR: Final[Path] = Path("docs") / "images"

#: Shared styling, kept in one place so the figure set reads as a single system.
_FIGSIZE: Final[tuple[float, float]] = (9.0, 5.5)
_DPI: Final[int] = 150
_GRID: Final[dict[str, object]] = {"alpha": 0.25, "linewidth": 0.6}


def _prepare_axes(title: str, xlabel: str, ylabel: str):
    """Create a styled figure and axis, importing matplotlib lazily.

    The import is deferred so that the optimisation and risk modules stay
    usable in headless or minimal environments where matplotlib is absent.
    The Agg backend is selected explicitly because these figures are written
    to disk, never shown interactively, and the default backend can fail
    outright on a machine with no display.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=_FIGSIZE)
    axis.set_title(title, fontsize=13, fontweight="bold")
    axis.set_xlabel(xlabel, fontsize=11)
    axis.set_ylabel(ylabel, fontsize=11)
    axis.grid(True, **_GRID)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    return figure, axis


def _save(figure, filename: str, output_dir: Path | None = None) -> Path:
    """Write a figure to ``output_dir`` and close it."""
    import matplotlib.pyplot as plt

    directory = IMAGE_DIR if output_dir is None else Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    figure.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(figure)
    LOGGER.info("wrote %s", path)
    return path


# ==============================================================================
# Figure 1 of the paper
# ==============================================================================


def plot_nikkei_loss_distribution(
    n_scenarios: int = 1_000,
    seed: int | None = 1997,
    output_dir: Path | None = None,
    **simulation_kwargs: object,
) -> Path:
    """Recreate Figure 1: NIKKEI one-day losses with the best normal fit.

    The paper's caption reads "Distribution of losses for the NIKKEI portfolio
    with best normal approximation, (1,000 scenarios), reproduced from Mausser
    and Rosen (1999)", and its purpose in the argument is to show that "the
    normal distribution fits the data poorly" - which is why the Minimum CVaR
    and Minimum Variance approaches diverge for this book.

    **This is a reconstruction, not a reproduction.**  The 11-instrument
    portfolio, its strikes, maturities, positions and marks are Table 7
    verbatim, and the implied volatilities are inverted from those marks so the
    base book reprices exactly.  The 1,000 scenarios, however, belonged to
    Algorithmics and are not published, so they are simulated here.  The figure
    therefore reproduces the paper's *qualitative* claim - visible left skew and
    a Jarque-Bera rejection of normality - but not its exact shape or scale.
    The annotation on the figure says so, so that the image cannot be
    mistaken for the original once it is lifted out of this repository.

    The "best normal approximation" is the maximum-likelihood fit, i.e. a
    Gaussian matched to the sample mean and standard deviation - the same
    object the paper overlays.

    Parameters
    ----------
    n_scenarios
        Scenario count; 1,000 matches the paper.
    seed
        Seed for reproducibility.
    output_dir
        Destination directory; defaults to :data:`IMAGE_DIR`.
    **simulation_kwargs
        Forwarded to
        :func:`~src.data.nikkei_portfolio.simulate_loss_scenarios` - for
        instance ``distribution="normal"`` or ``vol_shock_sd=0.35``.

    Returns
    -------
    Path
        The written PNG.
    """
    from scipy.stats import jarque_bera, norm

    from src.data.nikkei_portfolio import (
        PAPER_BETA,
        PAPER_CVAR_95,
        PAPER_VAR_95,
        simulate_loss_scenarios,
    )
    from src.models.risk_metrics import historical_var_cvar

    losses, base_value = simulate_loss_scenarios(
        n_scenarios=n_scenarios, seed=seed, **simulation_kwargs  # type: ignore[arg-type]
    )
    estimate = historical_var_cvar(losses, PAPER_BETA)
    test = jarque_bera(losses)

    # Scale to millions of JPY: the raw figures are in thousands, and axis
    # labels of eight digits are unreadable.
    scaled = losses / 1_000.0
    figure, axis = _prepare_axes(
        "Distribution of Losses, NIKKEI Portfolio (reconstruction of Figure 1)",
        "One-day loss (million JPY)",
        "Probability density",
    )

    axis.hist(
        scaled, bins=50, density=True, alpha=0.70,
        color="#4C72B0", edgecolor="white", linewidth=0.5,
        label=f"Simulated losses ({n_scenarios:,} scenarios)",
    )

    # The "best normal approximation" the paper overlays: a Gaussian matched to
    # the sample moments.
    grid = np.linspace(scaled.min(), scaled.max(), 500)
    axis.plot(
        grid, norm.pdf(grid, scaled.mean(), scaled.std(ddof=1)),
        color="#C44E52", linewidth=2.2, label="Best normal approximation",
    )

    axis.axvline(
        estimate.var / 1_000.0, color="#55A868", linestyle="--", linewidth=1.8,
        label=f"VaR {PAPER_BETA:.2f} = {estimate.var / 1_000.0:,.0f}M",
    )
    axis.axvline(
        estimate.cvar / 1_000.0, color="#8172B2", linestyle="-.", linewidth=1.8,
        label=f"CVaR {PAPER_BETA:.2f} = {estimate.cvar / 1_000.0:,.0f}M",
    )
    axis.legend(frameon=False, fontsize=9, loc="upper left")

    axis.text(
        0.985, 0.97,
        "RECONSTRUCTION - not the paper's data\n"
        f"Book: Table 7 verbatim, base value {base_value / 1_000.0:,.0f}M JPY\n"
        f"Scenarios simulated (originals unpublished)\n"
        f"Jarque-Bera p = {test.pvalue:.2e} "
        f"({'normality rejected' if test.pvalue < 0.01 else 'normality not rejected'})\n"
        f"Paper's published VaR {PAPER_VAR_95 / 1_000.0:,.0f}M / "
        f"CVaR {PAPER_CVAR_95 / 1_000.0:,.0f}M",
        transform=axis.transAxes, ha="right", va="top", fontsize=8,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#FFF4E6",
              "edgecolor": "#E8A33D", "linewidth": 1.0},
    )
    return _save(figure, "nikkei_loss_distribution.png", output_dir)


# ==============================================================================
# Step 1 - mean-CVaR frontier
# ==============================================================================


def _frontier(n_scenarios: int = 8_192, n_points: int = 25, beta: float = 0.95):
    """Solve the mean-CVaR efficient frontier used by two of the figures."""
    from src.data.paper_datasets import MEAN_RETURNS, sample_normal_returns
    from src.models.risk_metrics import PortfolioConstraints, cvar_efficient_frontier

    returns = sample_normal_returns(n_scenarios, engine="sobol", seed=20_000)
    targets = np.linspace(0.0055, 0.0135, n_points)
    points = cvar_efficient_frontier(
        returns, beta, targets,
        PortfolioConstraints(budget=1.0, lower_bounds=0.0),
        expected_returns=MEAN_RETURNS,
    )
    return points, returns


def plot_efficient_frontier(
    beta: float = 0.95, output_dir: Path | None = None
) -> Path:
    """Mean-CVaR efficient frontier, with the paper's benchmark marked.

    Every point is a global optimum, because the problem is convex.  The
    Table 3 minimum-variance portfolio is overlaid: by the paper's Proposition
    it must lie on this frontier at ``R = 0.011`` under normality, so its
    position is an independent visual check on the solver.

    Parameters
    ----------
    beta
        Confidence level.
    output_dir
        Destination directory.

    Returns
    -------
    Path
    """
    from src.data.paper_datasets import (
        MEAN_RETURNS, MIN_VARIANCE_WEIGHTS, REFERENCE_CVAR, TARGET_RETURN,
    )

    points, _ = _frontier(beta=beta)
    figure, axis = _prepare_axes(
        f"Mean-CVaR Efficient Frontier (beta = {beta:.2f})",
        f"Conditional Value-at-Risk at {beta:.0%}",
        "Expected monthly return",
    )
    axis.plot(
        [p.cvar for p in points], [p.expected_return for p in points],
        "o-", color="#4C72B0", markersize=4.5, linewidth=1.8,
        label="Minimum-CVaR frontier (globally optimal)",
    )
    if beta in REFERENCE_CVAR:
        axis.scatter(
            [REFERENCE_CVAR[beta]], [TARGET_RETURN],
            marker="*", s=320, color="#C44E52", zorder=5,
            label=f"Paper Table 3/4 benchmark (R = {TARGET_RETURN})",
        )
    axis.legend(frameon=False, fontsize=9, loc="lower right")
    axis.text(
        0.02, 0.97,
        "Benchmark portfolio: "
        + ", ".join(f"{w:.4f}" for w in MIN_VARIANCE_WEIGHTS)
        + f"\nE[R] = {float(MEAN_RETURNS @ MIN_VARIANCE_WEIGHTS):.6f}",
        transform=axis.transAxes, ha="left", va="top", fontsize=8,
    )
    return _save(figure, "efficient_frontier.png", output_dir)


def plot_weight_distribution(
    beta: float = 0.95, output_dir: Path | None = None
) -> Path:
    """Stacked composition of the optimal portfolio along the frontier.

    Shows how capital rotates out of bonds and into small caps as the return
    target rises - the allocation story behind the frontier curve.

    Parameters
    ----------
    beta
        Confidence level.
    output_dir
        Destination directory.

    Returns
    -------
    Path
    """
    from src.data.paper_datasets import INSTRUMENTS

    points, _ = _frontier(beta=beta)
    targets = np.array([p.expected_return for p in points])
    weights = np.vstack([p.weights for p in points])

    figure, axis = _prepare_axes(
        f"Optimal Portfolio Composition Along the Frontier (beta = {beta:.2f})",
        "Expected monthly return target",
        "Portfolio weight",
    )
    axis.stackplot(
        targets, weights.T, labels=list(INSTRUMENTS),
        colors=["#4C72B0", "#55A868", "#C44E52"], alpha=0.88,
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_xlim(targets.min(), targets.max())
    axis.legend(frameon=False, fontsize=9, loc="upper center", ncol=3)
    return _save(figure, "weight_distribution.png", output_dir)


# ==============================================================================
# Step 2 - multi-objective
# ==============================================================================


def _three_objective_problem(n_scenarios: int = 4_096, beta: float = 0.95):
    """Build the canonical return / CVaR / hedging-cost problem."""
    from src.data.paper_datasets import MEAN_RETURNS, sample_normal_returns
    from src.models.risk_metrics import PortfolioConstraints
    from src.optimization import (
        CVaRObjective, ExpectedReturnObjective, HedgingCostModel,
        HedgingCostObjective, ObjectiveSet,
    )

    returns = sample_normal_returns(n_scenarios, engine="sobol", seed=7)
    objectives = ObjectiveSet([
        ExpectedReturnObjective(MEAN_RETURNS),
        CVaRObjective.from_returns(returns, beta=beta),
        HedgingCostObjective(HedgingCostModel(
            n_assets=3,
            baseline_weights=np.array([1.0, 0.0, 0.0]),
            spreads=[0.0010, 0.0005, 0.0040],
            carry=[0.0, 0.0, 0.0020],
        )),
    ])
    box = PortfolioConstraints(budget=1.0, lower_bounds=0.0, upper_bounds=1.0)
    return objectives, box


def plot_pareto_front(output_dir: Path | None = None) -> Path:
    """Three-objective Pareto front: exact LP solution against NSGA-II.

    The exact Tchebycheff front is ground truth - every point a global
    optimum - so the genetic algorithm's points can only ever lie on it or
    behind it.  Plotting both on the same axes makes the gap visible rather
    than a summary statistic.

    Parameters
    ----------
    output_dir
        Destination directory.

    Returns
    -------
    Path
    """
    from src.optimization import (
        NSGAII, NSGAIIConfig, TchebycheffLPSolver, compare_fronts,
    )

    objectives, box = _three_objective_problem()
    exact = TchebycheffLPSolver(objectives, box).solve_front(n_partitions=12)
    evolved = NSGAII(
        objectives, box, NSGAIIConfig(population_size=100, n_generations=100, seed=42)
    ).run()
    metrics = compare_fronts(evolved, exact)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(9.5, 7.0))
    axis = figure.add_subplot(111, projection="3d")
    axis.set_title(
        "Three-Objective Pareto Front: Exact LP vs NSGA-II",
        fontsize=13, fontweight="bold",
    )
    axis.scatter(
        exact.objectives[:, 1], exact.objectives[:, 2], exact.objectives[:, 0],
        c="#C44E52", s=34, depthshade=False, label=f"Tchebycheff LP, exact ({len(exact)})",
    )
    axis.scatter(
        evolved.objectives[:, 1], evolved.objectives[:, 2], evolved.objectives[:, 0],
        c="#4C72B0", s=14, alpha=0.75, depthshade=False,
        label=f"NSGA-II ({len(evolved)})",
    )
    axis.set_xlabel("CVaR 0.95", fontsize=10)
    axis.set_ylabel("Hedging cost", fontsize=10)
    axis.set_zlabel("Expected return", fontsize=10)
    axis.view_init(elev=22, azim=-128)
    axis.legend(frameon=False, fontsize=9, loc="upper left")
    axis.text2D(
        0.02, 0.02,
        f"hypervolume ratio {metrics['hypervolume_ratio']:.4f}   "
        f"IGD {metrics['igd']:.4f}   "
        f"coverage(GA>LP) {metrics['coverage_candidate_over_reference']:.3f}",
        transform=axis.transAxes, fontsize=9,
    )
    return _save(figure, "pareto_front.png", output_dir)


def plot_nsga2_convergence(output_dir: Path | None = None) -> Path:
    """Hypervolume against generation, at two and three objectives.

    The two curves make the crowding-distance limitation concrete: at two
    objectives hypervolume climbs and settles, while at three it plateaus
    early and then wanders, which is why the exact reference front is kept.

    Parameters
    ----------
    output_dir
        Destination directory.

    Returns
    -------
    Path
    """
    from src.data.paper_datasets import MEAN_RETURNS, sample_normal_returns
    from src.models.risk_metrics import PortfolioConstraints
    from src.optimization import (
        NSGAII, NSGAIIConfig, CVaRObjective, ExpectedReturnObjective, ObjectiveSet,
    )

    objectives_3, box = _three_objective_problem()
    returns = sample_normal_returns(4_096, engine="sobol", seed=7)
    objectives_2 = ObjectiveSet([
        ExpectedReturnObjective(MEAN_RETURNS),
        CVaRObjective.from_returns(returns, beta=0.95),
    ])

    figure, axis = _prepare_axes(
        "NSGA-II Convergence: Hypervolume by Generation",
        "Generation", "Hypervolume (normalised to final value)",
    )
    for label, objective_set, colour in (
        ("Two objectives (return, CVaR)", objectives_2, "#4C72B0"),
        ("Three objectives (+ hedging cost)", objectives_3, "#C44E52"),
    ):
        front = NSGAII(
            objective_set, box,
            NSGAIIConfig(population_size=100, n_generations=200, seed=42, track_every=10),
        ).run()
        generations = [record["generation"] for record in front.history]
        volumes = np.array([record["hypervolume"] for record in front.history])
        # Each problem's hypervolume has its own units, so normalise to make the
        # two convergence *shapes* comparable on one axis.
        axis.plot(
            generations, volumes / volumes[-1], "o-",
            color=colour, markersize=3.5, linewidth=1.7, label=label,
        )
    axis.legend(frameon=False, fontsize=9, loc="lower right")
    return _save(figure, "nsga2_convergence.png", output_dir)


# ==============================================================================
# Step 3 - stress testing
# ==============================================================================


def plot_stress_attribution(output_dir: Path | None = None) -> Path:
    """P&L by revaluation method across every historical scenario.

    The gap between the bars *is* the linearisation error.  A delta-only risk
    report on a book with options is wrong by the visible amount, and the error
    grows with the square of the shock - which is why the crisis scenarios show
    the widest gaps.

    Parameters
    ----------
    output_dir
        Destination directory.

    Returns
    -------
    Path
    """
    from src.models.stress_tester import (
        BondInstrument, EquityInstrument, EuropeanOption, MarketState, OptionType,
        StressPortfolio, StressTester, default_underlyings,
    )

    market = MarketState(
        spots={"SPX": 4500.0, "GOVT": 100.0, "SMALL": 2000.0},
        volatilities={"SPX": 0.18, "GOVT": 0.06, "SMALL": 0.24},
        rate=0.04,
        dividend_yields={"SPX": 0.015, "SMALL": 0.010},
    )
    book = StressPortfolio(
        instruments=[
            EquityInstrument("S&P sleeve", "SPX"),
            BondInstrument("Long govt", "GOVT", duration=10.0, convexity=140.0,
                           base_rate=0.04),
            EquityInstrument("Small cap", "SMALL"),
            EuropeanOption("SPX 90% put", "SPX", 4050.0, 1.0, OptionType.PUT),
        ],
        units=np.array([10_000.0, 300_000.0, 12_500.0, 6_000.0]),
    )
    tester = StressTester(
        book, market, default_underlyings(bond_duration_in_instrument=True)
    )
    results = tester.run_all()

    names = [r.scenario.name for r in results]
    series = {
        "Full revaluation (exact)": ([r.pnl_full for r in results], "#4C72B0"),
        "Delta-gamma + vega": ([r.pnl_delta_gamma for r in results], "#55A868"),
        "Delta only": ([r.pnl_delta for r in results], "#C44E52"),
    }

    figure, axis = _prepare_axes(
        "Stress Test P&L by Revaluation Method",
        "", "Profit and loss (million, book currency)",
    )
    positions = np.arange(len(names))
    width = 0.26
    for offset, (label, (values, colour)) in enumerate(series.items()):
        axis.bar(
            positions + (offset - 1) * width,
            np.array(values) / 1e6, width,
            label=label, color=colour, edgecolor="white", linewidth=0.6,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.legend(frameon=False, fontsize=9, loc="lower left")
    worst = min(results, key=lambda r: r.pnl_full)
    axis.text(
        0.99, 0.04,
        f"Worst case: {worst.scenario.name} at {worst.return_pct:.1%}\n"
        f"Delta-only error there: {worst.delta_error:+.1%}",
        transform=axis.transAxes, ha="right", va="bottom", fontsize=8,
    )
    return _save(figure, "stress_attribution.png", output_dir)


# ==============================================================================
# Driver
# ==============================================================================

#: Figure name to generator, in the order the README presents them.
_FIGURES: Final[dict[str, object]] = {
    "nikkei-losses": plot_nikkei_loss_distribution,
    "efficient-frontier": plot_efficient_frontier,
    "weight-distribution": plot_weight_distribution,
    "pareto-front": plot_pareto_front,
    "nsga2-convergence": plot_nsga2_convergence,
    "stress-attribution": plot_stress_attribution,
}


def generate_all(
    only: Sequence[str] | None = None, output_dir: Path | None = None
) -> list[Path]:
    """Generate every figure, or a named subset.

    Parameters
    ----------
    only
        Figure keys to generate; ``None`` means all of them.  Valid keys are
        those of :data:`_FIGURES`.
    output_dir
        Destination directory; defaults to :data:`IMAGE_DIR`.

    Returns
    -------
    list of Path
        The written files, in generation order.

    Raises
    ------
    KeyError
        If a requested key is unknown - the message lists the valid ones,
        since a typo would otherwise silently skip a figure.
    """
    chosen = list(_FIGURES) if only is None else list(only)
    unknown = [key for key in chosen if key not in _FIGURES]
    if unknown:
        raise KeyError(
            f"Unknown figure(s) {unknown}. Available: {', '.join(_FIGURES)}."
        )

    written: list[Path] = []
    for key in chosen:
        generator = _FIGURES[key]
        print(f"  generating {key} ...", flush=True)
        written.append(generator(output_dir=output_dir))  # type: ignore[operator]
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m src.visualization``.

    Parameters
    ----------
    argv
        Argument list; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.visualization",
        description="Generate the project's figures into docs/images/.",
    )
    parser.add_argument(
        "--only", nargs="+", choices=sorted(_FIGURES), metavar="FIGURE",
        help=f"generate only these figures ({', '.join(sorted(_FIGURES))})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="destination directory (default: docs/images)",
    )
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    written = generate_all(only=arguments.only, output_dir=arguments.output_dir)
    print(f"\nWrote {len(written)} figure(s):")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
