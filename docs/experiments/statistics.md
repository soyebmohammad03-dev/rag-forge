# Statistical methodology

Code: `apps/api/src/rag_forge/arena/stats.py`. Part of the [Arena](README.md).

## Statistics (`paired-bootstrap-sign@1`)

Per arm and metric: n, skipped, mean, median, sample std, min, max and a 95% percentile bootstrap
CI of the mean (5000 resamples, seed 20261007, so recomputing gives identical intervals).

Paired comparison of two arms (`/runs/{id}/compare`, or `/arena/compare` across runs) uses only
cases where both arms define the metric:

- per-case differences (variant − baseline), mean, median, std, bootstrap CI of the mean
  difference, effect size `dz` = mean / sd of differences;
- wins / losses / ties oriented by the metric's direction;
- exact two-sided sign test (ties dropped), Holm-adjusted across the metrics of the comparison;
- a conclusion: `insufficient_cases` below 10 pairs, `descriptive_only` for directionless
  metrics, otherwise `variant_higher` / `variant_lower` when the CI excludes 0, else
  `no_detectable_difference`.

"Higher" describes the value; the UI resolves whether that is better from the metric's direction.
A conclusion is a statement about this dataset, never a general claim.

Incomparable arms are refused (`comparable: false`, with reasons) when a run is unfinished, or
the dataset content, corpus version, metric versions or k values differ, or no case was
evaluated by both. Warnings are added for differing case sets, failed cases, differing top-k or
evidence budgets (budget confounds), identical configurations, and the development dataset.

The leaderboard (`/runs/{id}/leaderboard?metric=`) orders arms by mean, gives tied means the same
position, gives no position to descriptive or undefined metrics, and marks arms whose CI overlaps
the leader's. It is descriptive; decisions use paired comparisons.
