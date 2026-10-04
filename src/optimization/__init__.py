"""Portfolio construction engines.

Step 2 of the build plan: multi-objective optimisation balancing expected
return, CVaR tail risk, and derivative hedging cost.

* :mod:`src.optimization.objectives` - the objective abstractions, including
  the LP-epigraph protocol that makes exact scalarisation possible.
* :mod:`src.optimization.multiobjective` - NSGA-II (Approach A), Tchebycheff
  scalarisation (Approach B), Pareto machinery and quality indicators.
"""

from src.optimization.multiobjective import (
    NSGAII,
    CrossoverOperator,
    MutationOperator,
    NSGAIIConfig,
    ParetoFront,
    PolynomialMutation,
    SimplexBlendCrossover,
    SimulatedBinaryCrossover,
    TchebycheffLPSolver,
    TchebycheffNonlinearSolver,
    TchebycheffSolution,
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
    LinearEpigraph,
    Objective,
    ObjectiveSense,
    ObjectiveSet,
    VarianceObjective,
)

__all__ = [
    "NSGAII",
    "CVaRObjective",
    "CrossoverOperator",
    "ExpectedReturnObjective",
    "HedgingCostModel",
    "HedgingCostObjective",
    "LinearEpigraph",
    "MutationOperator",
    "NSGAIIConfig",
    "Objective",
    "ObjectiveSense",
    "ObjectiveSet",
    "ParetoFront",
    "PolynomialMutation",
    "SimplexBlendCrossover",
    "SimulatedBinaryCrossover",
    "TchebycheffLPSolver",
    "TchebycheffNonlinearSolver",
    "TchebycheffSolution",
    "VarianceObjective",
    "WeightTransferMutation",
    "compare_fronts",
    "constrained_non_dominated_sort",
    "constraint_violation",
    "coverage",
    "crowding_distance",
    "default_reference_point",
    "fast_non_dominated_sort",
    "generational_distance",
    "hypervolume",
    "inverted_generational_distance",
    "non_dominated_mask",
    "project_onto_budget_box",
    "simplex_lattice_weights",
    "spacing",
]
