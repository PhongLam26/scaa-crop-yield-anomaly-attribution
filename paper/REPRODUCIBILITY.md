# Reproducibility

Run from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/run_all.py
python scripts/run_improvement_experiments.py
python scripts/build_paper_assets.py
python scripts/package_overleaf.py
```

The paper uses `06_grouped_driver_scaa_temporal_holdout` as the implementation of retrospective leave-one-event-year-out grouped SCAA, keeps `02_grouped_driver_scaa` as an in-sample exploratory comparison, and uses `03_observed_analog_counterfactual` as a plausibility robustness check.

The generated paper files are stored in `paper/latex_source/`; the Overleaf upload archive is stored in `paper/overleaf_zip/`.

Local TeX compilation is optional. If a TeX distribution is unavailable, upload the Overleaf zip and compile there.

Reference metadata was extracted from `paper/reference_pack/crop_yield_anomaly_attribution_references.docx`; preprint and future-year entries are tracked in `paper/REFERENCE_AUDIT.md` rather than in submit-ready BibTeX notes.
