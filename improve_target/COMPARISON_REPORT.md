# Improvement Experiment Comparison

This report compares alternative ways to turn yield anomalies into crop-specific weather-driver claims.

## Method Scorecard

Note: `total_score` is kept for internal method triage only. The paper reports raw metrics instead.

`06_grouped_driver_scaa_temporal_holdout` is the main submission method. `02_grouped_driver_scaa` is kept as an in-sample exploratory comparison, and `03_observed_analog_counterfactual` is a plausibility robustness check.

| Method | Total | Idea | Crop-specific | Event | Recoverability | Plausibility | Median phi | Weather-driven |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 03_observed_analog_counterfactual | 9.02 | 8.0 | 9.0 | 9.7 | 9.1 | 10.0 | 0.853 | 0.972 |
| 02_grouped_driver_scaa | 8.74 | 9.0 | 10.0 | 8.9 | 6.6 | 9.0 | 0.589 | 0.734 |
| 06_grouped_driver_scaa_temporal_holdout | 7.90 | 10.0 | 10.0 | 9.1 | 0.9 | 9.0 | 0.116 | 0.056 |
| 01_residual_target_scaa | 7.51 | 10.0 | 8.0 | 6.3 | 4.8 | 8.0 | 0.482 | 0.472 |
| 00_baseline_v1_raw_yield_scaa | 6.50 | 7.0 | 7.0 | 8.3 | 3.2 | 7.0 | 0.379 | 0.262 |
| 04_crop_specific_vulnerability_profiles | 5.28 | 7.0 | 10.0 | 0.0 | 0.7 | 9.0 |  |  |
| 05_early_mid_warning_improved | 4.75 | 5.0 | 6.0 | 0.0 | 6.0 | 8.0 |  |  |

## Top Paper-Friendly Temporal-Holdout SCAA Claims

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
- In North Dakota Wheat 2017, drought/dry-spell stress was the dominant modelled driver; moving max_dry_spell_1mm toward feasible normal levels recovered 0.215 t/ha (51.6%) of the 0.417 t/ha detrended yield shortfall.
- In North Dakota Oats 2017, drought/dry-spell stress was the dominant modelled driver; moving max_dry_spell_1mm toward feasible normal levels recovered 0.206 t/ha (51.1%) of the 0.403 t/ha detrended yield shortfall.

## Event-Year Consistency Check 2012/2021/2022

- 03_observed_analog_counterfactual: 97.1% of event-year attributions match expected heat/drought/moisture groups.
- 06_grouped_driver_scaa_temporal_holdout: 91.4% of event-year attributions match expected heat/drought/moisture groups.
- 02_grouped_driver_scaa: 88.6% of event-year attributions match expected heat/drought/moisture groups.
- 00_baseline_v1_raw_yield_scaa: 82.9% of event-year attributions match expected heat/drought/moisture groups.
- 01_residual_target_scaa: 62.9% of event-year attributions match expected heat/drought/moisture groups.

## Event-Year Null Baselines

- Always drought: 100.0%.
- Always heat: 100.0%.
- Most frequent driver (heat): 100.0%.
- Driver-frequency random: 89.6%.
- Retrospective leave-one-event-year-out grouped SCAA: 91.4%.

## Crop Vulnerability Highlights

- Barley is most sensitive to heat: median effect -0.134 t/ha; Oklahoma (-0.253); South Dakota (-0.222); Kansas (-0.191).
- Wheat is most sensitive to drought: median effect -0.103 t/ha; Washington (-0.378); Colorado (-0.327); Oklahoma (-0.282).
- Canola is most sensitive to heat: median effect -0.086 t/ha; Kansas (-0.245); Minnesota (-0.167); Oklahoma (-0.142).
- Barley is most sensitive to drought: median effect -0.057 t/ha; Washington (-0.249); North Dakota (-0.165); Nebraska (-0.143).
- Canola is most sensitive to drought: median effect -0.049 t/ha; Washington (-0.175); North Dakota (-0.125); Montana (-0.116).
- Wheat is most sensitive to heat: median effect -0.041 t/ha; South Dakota (-0.224); Montana (-0.173); Minnesota (-0.110).
- Oats is most sensitive to heat: median effect -0.034 t/ha; South Dakota (-0.162); Colorado (-0.128); Nebraska (-0.118).
- Oats is most sensitive to drought: median effect -0.033 t/ha; North Dakota (-0.140); Washington (-0.140); South Dakota (-0.110).
