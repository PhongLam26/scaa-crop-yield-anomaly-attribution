# Temporal-Holdout Grouped-Driver SCAA

- For each anomaly year, the residual model is trained after excluding every row from that year.
- The residual target is raw detrended yield residual in t/ha; standardized residuals only screen anomalies.
- This is the main paper protocol because it avoids explaining event rows with a model fitted on the same year.
