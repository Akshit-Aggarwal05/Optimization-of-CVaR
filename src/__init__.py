"""Advanced Portfolio Risk Engine & Tail-Risk Stress Tester.

Sub-packages
------------
``src.data``
    Market data adapters, scenario generation, and the published benchmark
    datasets used for regression testing.
``src.models``
    Risk measurement: VaR/CVaR estimators, the Rockafellar-Uryasev programme,
    and the non-linear stress-testing engine.
``src.optimization``
    Portfolio construction: multi-objective solvers (NSGA-II, Tchebycheff
    scalarisation) balancing return, tail risk and hedging cost.
"""

__version__ = "0.1.0"
