# Paper Contribution Decision

Recommended main attribution method: `06_grouped_driver_scaa_temporal_holdout` because it excludes each event year from the residual-model fit before attribution.

`03_observed_analog_counterfactual` can recover more because it replaces the full weather vector with a real normal season. Use it as robustness evidence, not as the main sparse attribution method.

## Recommended Paper Framing

- Main contribution: detrended anomaly attribution using event-year temporal-holdout residual models and grouped sparse counterfactual weather changes.
- Explanation layer: report both dominant feature and physical driver group so each claim is crop-specific.
- Robustness: use observed analog counterfactuals to show the weather replacement is historically plausible.
- Supplementary contribution: crop-specific vulnerability profiles answer which weather stress lowers which crop.

## Why This Is More Publishable Than V1 Alone

- V1 proves the yield model and anomaly pipeline work, but its raw-yield attribution is conservative.
- Residual and grouped methods target the actual object of interest: abnormal detrended yield shortfall.
- Temporal holdout reduces in-sample attribution concerns before the paper makes event-level claims.
- The global claim table states crop, region, year, driver group, dominant feature, and recovered t/ha.

## Best Example Claims

- In South Dakota Oats 2006, drought/dry-spell stress was the dominant modelled driver; moving rain_mean toward feasible normal levels recovered 0.273 t/ha (71.7%) of the 0.381 t/ha detrended yield shortfall.
- In Washington Barley 2015, drought/dry-spell stress was the dominant modelled driver; moving hot_dry_days_30_1mm toward feasible normal levels recovered 0.598 t/ha (69.3%) of the 0.862 t/ha detrended yield shortfall.
- In Washington Wheat 1992, drought/dry-spell stress was the dominant modelled driver; moving max_dry_spell_1mm toward feasible normal levels recovered 0.390 t/ha (67.2%) of the 0.579 t/ha detrended yield shortfall.
- In Texas Wheat 2009, drought/dry-spell stress was the dominant modelled driver; moving dry_spell_events_14d toward feasible normal levels recovered 0.273 t/ha (66.6%) of the 0.410 t/ha detrended yield shortfall.
- In North Dakota Canola 2012, drought/dry-spell stress was the dominant modelled driver; moving hot_dry_days_30_1mm toward feasible normal levels recovered 0.183 t/ha (64.0%) of the 0.285 t/ha detrended yield shortfall.
- In South Dakota Wheat 2006, drought/dry-spell stress was the dominant modelled driver; moving rain_mean toward feasible normal levels recovered 0.306 t/ha (60.8%) of the 0.504 t/ha detrended yield shortfall.
- In North Dakota Canola 2002, drought/dry-spell stress was the dominant modelled driver; moving max_dry_spell_1mm toward feasible normal levels recovered 0.153 t/ha (58.3%) of the 0.263 t/ha detrended yield shortfall.
- In Washington Canola 2015, drought/dry-spell stress was the dominant modelled driver; moving hot_dry_days_30_1mm toward feasible normal levels recovered 0.367 t/ha (56.7%) of the 0.646 t/ha detrended yield shortfall.
- In Montana Wheat 2017, heat exposure was the dominant modelled driver; moving heat_days_30 toward feasible normal levels recovered 0.266 t/ha (54.6%) of the 0.487 t/ha detrended yield shortfall.
- In Iowa Oats 2010, drought/dry-spell stress was the dominant modelled driver; moving rain_sum toward feasible normal levels recovered 0.162 t/ha (52.0%) of the 0.312 t/ha detrended yield shortfall.

## Method Roles

- `06_grouped_driver_scaa_temporal_holdout`: main submission method; median phi 0.116, weather-driven rate 5.6%.
- `02_grouped_driver_scaa`: in-sample exploratory grouped explanation; median phi 0.589.
- `03_observed_analog_counterfactual`: robustness check; median phi 0.853.
- `04_crop_specific_vulnerability_profiles`: use as a table answering which weather driver reduces each crop.
- `05_early_mid_warning_improved`: keep as an operational extension, not the core paper contribution.

## Crop-Specific Vulnerability Claims

- For Barley, heat exposure produced a median modelled residual change of -0.134 t/ha under adverse observed extremes; the most sensitive states were Oklahoma, South Dakota, Kansas.
- For Wheat, drought/dry-spell stress produced a median modelled residual change of -0.103 t/ha under adverse observed extremes; the most sensitive states were Washington, Colorado, Oklahoma.
- For Canola, heat exposure produced a median modelled residual change of -0.086 t/ha under adverse observed extremes; the most sensitive states were Kansas, Minnesota, Oklahoma.
- For Barley, drought/dry-spell stress produced a median modelled residual change of -0.057 t/ha under adverse observed extremes; the most sensitive states were Washington, North Dakota, Nebraska.
- For Canola, drought/dry-spell stress produced a median modelled residual change of -0.049 t/ha under adverse observed extremes; the most sensitive states were Washington, North Dakota, Montana.
- For Wheat, heat exposure produced a median modelled residual change of -0.041 t/ha under adverse observed extremes; the most sensitive states were South Dakota, Montana, Minnesota.
- For Oats, heat exposure produced a median modelled residual change of -0.034 t/ha under adverse observed extremes; the most sensitive states were South Dakota, Colorado, Nebraska.
- For Oats, drought/dry-spell stress produced a median modelled residual change of -0.033 t/ha under adverse observed extremes; the most sensitive states were North Dakota, Washington, South Dakota.
