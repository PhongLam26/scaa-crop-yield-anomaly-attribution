# SCAA US Weather-Yield Research Prototype

This repository is a self-contained prototype for sparse counterfactual attribution
of US crop-yield anomalies to extreme-weather drivers. It follows the advisor idea
stored in `docs/reference/`: do not only predict yield, but first detrend each
crop-region yield series, identify abnormal low-yield events, and then ask which
minimal feasible weather change would move the model prediction back toward a
normal year.

The prototype also includes a small early-warning baseline that uses only early
season and early+mid season weather features to estimate the probability of a
low-yield year.

## Data

- `data/processed/us_model_frame_hemisphere_aware_1990_2025.csv`: ready-to-run
  crop-region-year modeling frame with weather features, location fields, growing
  season window, and `yield_t_ha`.
- `data/raw/us_yield_1989_2025_tha.csv`: raw harmonized USDA NASS yield table.
- `data/raw/nasa_power_daily.zip`: NASA POWER daily weather files for the twelve
  US states used in the prototype.
- `docs/reference/`: original idea artifacts, including the LaTeX paper draft,
  PDF, and advisor note image.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/run_all.py
```

The run writes reproducible outputs to:

- `outputs/`: metrics, anomaly scores, counterfactual attribution tables, and
  early-warning predictions.
- `figures/`: PNG figures for the model, anomaly timeline, attribution drivers,
  recoverable fraction, and early-warning comparison.
- `RESULTS_SUMMARY.md`: generated summary of the latest run.

## Main Workflow

1. Validate the processed frame: 1,257 rows, years 1990-2025, four crops, and no
   missing yield target.
2. Train an ExtraTrees yield model under forward-time splits:
   train through 2015, test 2016-2021; then train through 2018, test 2019-2025.
3. Detrend each crop-region yield series and flag low-yield anomalies with
   residual z-score below -1.
4. Fit a final historical explanation model and run sparse counterfactual search
   for each anomaly.
5. Train early-warning classifiers using early-season and early+mid-season feature
   groups, with a calibration-derived conformal-style probability interval.

## Improvement Experiments

After the V1 run, execute the experiment suite:

```powershell
python scripts/run_improvement_experiments.py
```

This writes `improve_target/`, with one folder per method:

- `00_baseline_v1_raw_yield_scaa`
- `01_residual_target_scaa`
- `02_grouped_driver_scaa`
- `03_observed_analog_counterfactual`
- `04_crop_specific_vulnerability_profiles`
- `05_early_mid_warning_improved`

The most important global files are:

- `improve_target/COMPARISON_REPORT.md`
- `improve_target/PAPER_CONTRIBUTION_DECISION.md`
- `improve_target/method_scorecard.csv`
- `improve_target/crop_driver_claims.csv`
- `improve_target/event_validation_2012_2021_2022.csv`

Use `02_grouped_driver_scaa` as the paper-friendly main attribution method if it
remains the top sparse/grouped method. Use `03_observed_analog_counterfactual` as
robustness evidence because it replaces the full weather vector with a real
normal season.

## Paper Package

Build the paper figures, tables, data manifest, and Overleaf archive:

```powershell
python scripts/build_paper_assets.py
python scripts/package_overleaf.py
```

Paper files are written to:

- `paper/latex_source/main.tex`
- `paper/latex_source/references.bib`
- `paper/latex_source/figures/`
- `paper/latex_source/tables/`
- `paper/overleaf_zip/scaa_crop_yield_anomaly_attribution.zip`
- `paper/DATA_MANIFEST.md`
- `paper/REPRODUCIBILITY.md`

Upload the Overleaf zip and compile `main.tex`. Reference metadata is derived from
`paper/reference_pack/crop_yield_anomaly_attribution_references.docx`; verify
recent, preprint, and 2025-2026 entries before formal submission.

## Git Push Setup

This folder is intended to be its own repository.

```powershell
git remote add origin <your-github-repo-url>
git branch -M main
git push -u origin main
```

If the remote should not store packaged data, uncomment the data ignore lines in
`.gitignore` before committing.

## Interpretation

The attribution tables are predictive, model-internal explanations. They should be
read as event-driver evidence from the fitted weather-yield model, not as proven
causal effects.
