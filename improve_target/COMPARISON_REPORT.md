# Improvement Experiment Comparison

This report compares alternative ways to turn yield anomalies into crop-specific weather-driver claims.

## Method Scorecard

Note: observed analog counterfactuals may score highly because they replace the full weather vector with a real normal season; use them as robustness evidence, while residual/grouped SCAA remain the main sparse attribution candidates.

| Method | Total | Idea | Crop-specific | Event | Recoverability | Plausibility | Median phi | Weather-driven |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 03_observed_analog_counterfactual | 9.02 | 8.0 | 9.0 | 9.7 | 9.1 | 10.0 | 0.853 | 0.972 |
| 02_grouped_driver_scaa | 8.74 | 9.0 | 10.0 | 8.9 | 6.6 | 9.0 | 0.589 | 0.734 |
| 01_residual_target_scaa | 7.51 | 10.0 | 8.0 | 6.3 | 4.8 | 8.0 | 0.482 | 0.472 |
| 00_baseline_v1_raw_yield_scaa | 6.50 | 7.0 | 7.0 | 8.3 | 3.2 | 7.0 | 0.379 | 0.262 |
| 04_crop_specific_vulnerability_profiles | 5.28 | 7.0 | 10.0 | 0.0 | 0.7 | 9.0 |  |  |
| 05_early_mid_warning_improved | 4.75 | 5.0 | 6.0 | 0.0 | 6.0 | 8.0 |  |  |

## Top Paper-Friendly Grouped SCAA Claims

- In North Dakota Canola 2021, drought/dry-spell stress was the dominant modelled driver; moving dry_spell_events_7d toward feasible normal levels recovered 0.522 t/ha (100.0%) of the 0.522 t/ha detrended yield shortfall.
- In Montana Barley 2002, heat exposure was the dominant modelled driver; moving heatwave_events_3d_35 toward feasible normal levels recovered 0.387 t/ha (100.0%) of the 0.387 t/ha detrended yield shortfall.
- In North Dakota Canola 2002, drought/dry-spell stress was the dominant modelled driver; moving max_dry_spell_1mm toward feasible normal levels recovered 0.263 t/ha (100.0%) of the 0.263 t/ha detrended yield shortfall.
- In Minnesota Canola 2021, heat exposure was the dominant modelled driver; moving heat_days_30 toward feasible normal levels recovered 0.372 t/ha (98.2%) of the 0.379 t/ha detrended yield shortfall.
- In Oklahoma Oats 2007, drought/dry-spell stress was the dominant modelled driver; moving rain_sum toward feasible normal levels recovered 0.286 t/ha (92.3%) of the 0.310 t/ha detrended yield shortfall.
- In South Dakota Oats 2011, excess rainfall or wetness was the dominant modelled driver; moving max_3day_rain toward feasible normal levels recovered 0.525 t/ha (91.7%) of the 0.572 t/ha detrended yield shortfall.
- In Illinois Oats 1993, drought/dry-spell stress was the dominant modelled driver; moving dry_days_1mm toward feasible normal levels recovered 0.400 t/ha (91.5%) of the 0.437 t/ha detrended yield shortfall.
- In Colorado Wheat 2013, drought/dry-spell stress was the dominant modelled driver; moving dry_spell_events_7d toward feasible normal levels recovered 0.625 t/ha (90.9%) of the 0.687 t/ha detrended yield shortfall.
- In Montana Canola 2002, heat exposure was the dominant modelled driver; moving heatwave_events_3d_35 toward feasible normal levels recovered 0.351 t/ha (90.1%) of the 0.389 t/ha detrended yield shortfall.
- In South Dakota Oats 2021, excess rainfall or wetness was the dominant modelled driver; moving wet_days_1mm toward feasible normal levels recovered 0.441 t/ha (90.1%) of the 0.489 t/ha detrended yield shortfall.
- In Oklahoma Wheat 1996, drought/dry-spell stress was the dominant modelled driver; moving max_dry_spell_1mm toward feasible normal levels recovered 0.593 t/ha (87.1%) of the 0.681 t/ha detrended yield shortfall.
- In Montana Oats 2021, drought/dry-spell stress was the dominant modelled driver; moving hot_dry_days_30_1mm toward feasible normal levels recovered 0.325 t/ha (87.0%) of the 0.374 t/ha detrended yield shortfall.

## Event Validation 2012/2021/2022

- 03_observed_analog_counterfactual: 97.1% of event-year attributions match expected heat/drought/moisture groups.
- 02_grouped_driver_scaa: 88.6% of event-year attributions match expected heat/drought/moisture groups.
- 00_baseline_v1_raw_yield_scaa: 82.9% of event-year attributions match expected heat/drought/moisture groups.
- 01_residual_target_scaa: 62.9% of event-year attributions match expected heat/drought/moisture groups.

## Crop Vulnerability Highlights

- Barley is most sensitive to heat: median effect -0.134 t/ha; Oklahoma (-0.253); South Dakota (-0.222); Kansas (-0.191).
- Wheat is most sensitive to drought: median effect -0.103 t/ha; Washington (-0.378); Colorado (-0.327); Oklahoma (-0.282).
- Canola is most sensitive to heat: median effect -0.086 t/ha; Kansas (-0.245); Minnesota (-0.167); Oklahoma (-0.142).
- Barley is most sensitive to drought: median effect -0.057 t/ha; Washington (-0.249); North Dakota (-0.165); Nebraska (-0.143).
- Canola is most sensitive to drought: median effect -0.049 t/ha; Washington (-0.175); North Dakota (-0.125); Montana (-0.116).
- Wheat is most sensitive to heat: median effect -0.041 t/ha; South Dakota (-0.224); Montana (-0.173); Minnesota (-0.110).
- Oats is most sensitive to heat: median effect -0.034 t/ha; South Dakota (-0.162); Colorado (-0.128); Nebraska (-0.118).
- Oats is most sensitive to drought: median effect -0.033 t/ha; North Dakota (-0.140); Washington (-0.140); South Dakota (-0.110).
