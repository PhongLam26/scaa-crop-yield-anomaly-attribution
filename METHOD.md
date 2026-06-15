# Method

## Research Goal

The prototype converts a black-box crop-yield model into an event-level
interpretation tool. For each crop-region-year, it separates long-run yield trend
from short-run weather variation. For the years with unusually low detrended yield,
it estimates which sparse set of weather indicators would need to change, within
historically feasible bounds, for the model prediction to move back toward the
trend-expected normal yield.

## Modeling Unit

The unit of analysis is a US crop-region-year. The processed frame covers four
crops, twelve states, and years 1990-2025. The target is `yield_t_ha`.

## Yield Model

The yield model is an ExtraTrees regressor with:

- 350 trees
- minimum leaf size 2
- median imputation and standardization for numeric features
- most-frequent imputation and one-hot encoding for `region` and `crop`
- forward-time evaluation splits

The evaluation splits are kept separate from the attribution model:

- train <= 2015, test 2016-2021
- train <= 2018, test 2019-2025

After predictive skill is checked, a final model is fit on the full historical
frame for retrospective explanation of all detected anomalies.

## Anomaly Detection

For each crop-region series, the pipeline fits a linear trend:

```text
yield = intercept + slope * year
```

The residual is standardized within that crop-region series using the sample
standard deviation. A low-yield anomaly is any row with:

```text
trend_residual_z < -1
```

This design follows the advisor note: explain abnormal yield events instead of
explaining average prediction behavior.

## Sparse Counterfactual Attribution

For each low-yield anomaly, the pipeline uses the final yield model to compare:

- the observed model prediction for the anomalous weather year
- the trend-expected yield for that crop-region-year
- the prediction after sparse feasible weather changes

Candidate weather changes are constrained to values observed in the same
`region + window` group. Attribution uses full-season weather indicators; the
early and mid-season feature blocks are reserved for the warning baseline. The
search uses a greedy sparse budget of four changed features. At each step it tries
feasible quantile values for each not-yet-selected weather feature and keeps the
change with the best prediction gain per standardized feature movement.

The main output is:

```text
weather_recoverable_fraction =
  (counterfactual_prediction - original_prediction)
  / (trend_yield - original_prediction)
```

The fraction is clipped to `[0, 1]`. An anomaly is labeled weather-driven when the
fraction is at least 0.5. The dominant driver is the selected weather feature with
the largest standardized movement.

## Early-Warning Baseline

The early-warning experiment treats the detrended anomaly flag as a binary target.
It trains ExtraTrees classifiers with two feature sets:

- `early_third`: only `*_early` features plus year, location, crop, and region
- `early_mid_two_thirds`: `*_early` and `*_mid` features plus year, location,
  crop, and region

The split is:

- train <= 2015
- calibration 2016-2018
- test >= 2019

The conformal-style interval is based on calibration absolute probability error:

```text
abs(y_calibration - p_calibration)
```

The 90th percentile of this error is used as a probability interval radius around
test probabilities. This is a simple operational uncertainty band, not a formal
causal confidence interval.

## Caveat

All outputs are model-internal predictive explanations. They are useful for
planning, screening, and hypothesis generation, but they are not causal effects.
