"""Descriptive and paired statistics over per-case metric values. Deterministic: fixed seeds.

- Summaries: n, mean, median, sample standard deviation, min, max, and a percentile bootstrap
  confidence interval for the mean (n >= 2).
- Paired comparison (same cases, both arms define the metric): per-case differences
  (variant - baseline), their mean, median and sd, a percentile bootstrap CI for the mean
  difference, Cohen's d_z (mean / sd of differences), wins / losses / ties oriented by the
  metric's direction, an exact two-sided sign test (ties dropped), and Holm adjustment of the
  sign-test p-values across the metrics of one comparison.
- A conclusion is drawn only from the bootstrap CI and only with at least
  `MIN_CASES_FOR_CONCLUSION` pairs: "variant_higher" / "variant_lower" when the CI excludes 0,
  otherwise "no_detectable_difference". Descriptive metrics (no direction) are never concluded.
  The bootstrap assumes cases are exchangeable draws from the population of interest. A small
  or hand-built dataset does not support that assumption, and the dataset source is reported
  alongside.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from rag_forge.domain.arena import CaseDifference, MetricFamily, PairedComparison, StatsMethod

CONFIDENCE = 0.95
RESAMPLES = 5000
SEED = 20261007
MIN_CASES_FOR_CONCLUSION = 10

METHOD = StatsMethod(
    name="paired-bootstrap-sign",
    version=1,
    confidence=CONFIDENCE,
    bootstrap_resamples=RESAMPLES,
    seed=SEED,
    min_cases_for_conclusion=MIN_CASES_FOR_CONCLUSION,
    multiple_comparisons="Holm across the metrics of one comparison (sign-test p-values)",
)
METHOD_ID = f"{METHOD.name}@{METHOD.version}"


def _r(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else round(float(x), 6)


def bootstrap_ci(values: Sequence[float]) -> tuple[float, float] | None:
    """Percentile bootstrap CI of the mean; None below 2 values."""
    if len(values) < 2:
        return None
    a = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    means = a[rng.integers(0, len(a), size=(RESAMPLES, len(a)))].mean(axis=1)
    alpha = (1 - CONFIDENCE) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(lo), float(hi)


def describe(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return dict.fromkeys(("mean", "median", "std", "min", "max", "ci_low", "ci_high"))
    a = np.asarray(values, dtype=np.float64)
    ci = bootstrap_ci(values)
    return {
        "mean": _r(a.mean()),
        "median": _r(float(np.median(a))),
        "std": _r(a.std(ddof=1)) if len(a) > 1 else None,
        "min": _r(a.min()),
        "max": _r(a.max()),
        "ci_low": _r(ci[0]) if ci else None,
        "ci_high": _r(ci[1]) if ci else None,
    }


def sign_test(wins: int, losses: int) -> float | None:
    """Exact two-sided binomial sign test with p = 0.5; None without non-tied pairs."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def holm(pvalues: list[float | None]) -> list[float | None]:
    """Holm step-down adjustment; None entries are left out of the family."""
    idx = sorted((p, i) for i, p in enumerate(pvalues) if p is not None)
    m = len(idx)
    out: list[float | None] = [None] * len(pvalues)
    running = 0.0
    for rank, (p, i) in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * p))
        out[i] = running
    return out


def paired(
    metric: str,
    family: MetricFamily,
    higher_is_better: bool | None,
    baseline: dict[str, float],
    variant: dict[str, float],
) -> PairedComparison:
    """Compare the cases both arms define, in sorted case order."""
    cases = sorted(set(baseline) & set(variant))
    diffs = [
        CaseDifference(
            case_id=c,
            baseline=baseline[c],
            variant=variant[c],
            difference=round(variant[c] - baseline[c], 9),
        )
        for c in cases
    ]
    d = np.asarray([x.difference for x in diffs], dtype=np.float64)
    sign = 0 if higher_is_better is None else (1 if higher_is_better else -1)
    wins = int(np.sum(d * sign > 0)) if sign else 0
    losses = int(np.sum(d * sign < 0)) if sign else 0
    ties = len(d) - wins - losses if sign else 0
    ci = bootstrap_ci(d.tolist()) if len(d) else None
    sd = float(d.std(ddof=1)) if len(d) > 1 else None
    if not len(d):
        conclusion = "insufficient_cases"
    elif higher_is_better is None:
        conclusion = "descriptive_only"
    elif len(d) < MIN_CASES_FOR_CONCLUSION or ci is None:
        conclusion = "insufficient_cases"
    elif ci[0] > 0:
        conclusion = "variant_higher"
    elif ci[1] < 0:
        conclusion = "variant_lower"
    else:
        conclusion = "no_detectable_difference"
    return PairedComparison(
        metric=metric,
        family=family,
        higher_is_better=higher_is_better,
        n_pairs=len(d),
        baseline_mean=_r(float(np.mean([x.baseline for x in diffs]))) if diffs else None,
        variant_mean=_r(float(np.mean([x.variant for x in diffs]))) if diffs else None,
        mean_difference=_r(float(d.mean())) if len(d) else None,
        median_difference=_r(float(np.median(d))) if len(d) else None,
        std_difference=_r(sd),
        ci_low=_r(ci[0]) if ci else None,
        ci_high=_r(ci[1]) if ci else None,
        effect_size_dz=_r(float(d.mean()) / sd) if sd else None,
        wins=wins,
        losses=losses,
        ties=ties,
        sign_test_p=_r(sign_test(wins, losses)) if sign else None,
        holm_p=None,
        conclusion=conclusion,
        differences=diffs,
    )


def adjust(comparisons: list[PairedComparison]) -> list[PairedComparison]:
    adjusted = holm([c.sign_test_p for c in comparisons])
    return [
        c.model_copy(update={"holm_p": _r(p)}) for c, p in zip(comparisons, adjusted, strict=True)
    ]
