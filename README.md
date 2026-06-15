# SCAA US Weather-Yield Research Prototype

This repository is a self-contained research artifact for sparse counterfactual
attribution of US crop-yield anomalies to extreme-weather drivers. The workflow does
not only predict yield: it detrends each crop-region yield series, identifies
below-trend events, and asks which minimal feasible weather change would move the
model prediction back toward a normal year.

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

This writes a local `improve_target/` folder, which is intentionally omitted from
the public repository because it is a generated experiment workspace:

- `00_baseline_v1_raw_yield_scaa`
- `01_residual_target_scaa`
- `02_grouped_driver_scaa`
- `03_observed_analog_counterfactual`
- `04_crop_specific_vulnerability_profiles`
- `05_early_mid_warning_improved`

Run this step before rebuilding paper assets because the manuscript tables and figures
use the regenerated comparison and attribution outputs.

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
- `paper/DAP_new.pdf`
- `paper/DATA_MANIFEST.md`
- `paper/REPRODUCIBILITY.md`

Upload the Overleaf zip and compile `main.tex`, or open the compiled
`paper/DAP_new.pdf`.

## Git Push Setup

This folder is intended to be its own repository.

```powershell
git remote add origin <your-github-repo-url>
git branch -M main
git push -u origin main
```

Generated audit workspaces and internal review notes are ignored so the public branch
stays focused on the paper, data, code, and reproducibility package.

## Interpretation

The attribution tables are predictive, model-internal explanations. They should be
read as event-driver evidence from the fitted weather-yield model, not as proven
causal effects.
