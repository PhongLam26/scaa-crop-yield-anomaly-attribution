# Review Fix Changelog

Generated for the conference-submission polish pass.

## Completed Fixes

- Separated supplementary tables and extension figures into `paper/latex_source/supplement.tex` so `main.tex` stays focused for a 12-page target.
- Reframed residual-model results as weak-to-modest validation rather than residual forecasting skill.
- Added the absolute weather-driven count: `5.6% of anomalies (12 of 214)`.
- De-emphasized event-year consistency and stated that broad labels are easy for null baselines to match.
- Compact residual validation and null-baseline tables are generated from `scripts/build_paper_assets.py`.
- Kept audit CSV files in `paper/generated_table_csv/` instead of the Overleaf `tables/` folder.
- Regenerated the Overleaf package without CSV files.

