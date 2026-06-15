# Paper Contribution Decision

Recommended main attribution method: `02_grouped_driver_scaa` with total score 8.74.

`03_observed_analog_counterfactual` can score higher numerically because it replaces the full weather vector with a real normal season. Use it as robustness evidence, not as the main sparse attribution method.

## Recommended Paper Framing

- Main contribution: detrended anomaly attribution using sparse counterfactual weather changes.
- Explanation layer: report both dominant feature and physical driver group so each claim is crop-specific.
- Robustness: use observed analog counterfactuals to show the weather replacement is historically plausible.
- Supplementary contribution: crop-specific vulnerability profiles answer which weather stress lowers which crop.

## Why This Is More Publishable Than V1 Alone

- V1 proves the yield model and anomaly pipeline work, but its raw-yield attribution is conservative.
- Residual and grouped methods target the actual object of interest: abnormal detrended yield shortfall.
- The global claim table states crop, region, year, driver group, dominant feature, and recovered t/ha.

## Best Example Claims

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

## Method Roles

- `02_grouped_driver_scaa`: paper-friendly grouped explanation score 8.74.
- `03_observed_analog_counterfactual`: robustness score 9.02.
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
