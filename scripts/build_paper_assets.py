from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
LATEX = PAPER / "latex_source"
FIGURES = LATEX / "figures"
TABLES = LATEX / "tables"
SUPPLEMENT = LATEX / "supplement"
TABLE_CSV = PAPER / "generated_table_csv"
WORKFLOW_SOURCE = PAPER / "assets" / "fig01_method_workflow_clean.png"


METHOD_LABELS = {
    "00_baseline_v1_raw_yield_scaa": "Raw-yield SCAA",
    "01_residual_target_scaa": "Residual-target SCAA",
    "02_grouped_driver_scaa": "Grouped-driver SCAA",
    "06_grouped_driver_scaa_temporal_holdout": "Leave-one-event-year-out grouped SCAA",
    "03_observed_analog_counterfactual": "Observed-analog counterfactual",
}


DRIVER_GROUPS = [
    {
        "driver_group": "heat",
        "description": "High-temperature and heatwave exposure",
        "features": "heat_days_30, heat_days_35, heatwave_events_3d_30, heat_degree_days_30, season_tmax_mean",
    },
    {
        "driver_group": "drought",
        "description": "Rainfall deficit, dry days, and dry-spell persistence",
        "features": "rain_sum, rain_mean, dry_days_1mm, max_dry_spell_1mm, dry_spell_events_7d, dry_spell_events_14d, hot_dry_days_30_1mm",
    },
    {
        "driver_group": "frost_cold",
        "description": "Cold and frost exposure",
        "features": "frost_days_0, cold_days_5, min_tmin, frost_events_2d, season_tmin_mean",
    },
    {
        "driver_group": "excess_rain",
        "description": "Wetness and heavy-rainfall intensity",
        "features": "wet_days_1mm, heavy_rain_days_10, heavy_rain_days_20, max_1day_rain, max_3day_rain, max_7day_rain",
    },
    {
        "driver_group": "radiation",
        "description": "Seasonal solar-radiation anomaly",
        "features": "radiation_sum, radiation_mean",
    },
]


REFERENCE_MAP = [
    ("Introduction", "Extreme weather and crop-yield losses", "Lesk2016; Zampieri2017; Vogel2019; Ray2015"),
    ("Related work", "Machine-learning crop-yield prediction", "Paudel2021; Meroni2021; Khaki2019; LengHall2020"),
    ("Related work", "Yield anomalies and detrending", "Lu2017; Ray2015; Meng2024; Sjulgard2023"),
    ("Related work", "Interpretable ML for yield models", "LundbergLee2017; Ribeiro2016; Mohan2025"),
    ("Method", "Sparse and feasible counterfactual explanation", "Wachter2018; Mothilal2020; Ustun2019; Poyiadzi2020; Verma2024"),
    ("Method and limitations", "Event-attribution language and pitfalls", "Hannart2016; Otto2017; Oldenborgh2021; OrtizBobea2021"),
    ("Data", "Weather and yield data sources", "NASAPOWER2025; USDANASSQuickStats"),
    ("Extension", "Early warning and conformal uncertainty", "Anderson2024; Meroni2021; Singh2024; Farag2025"),
]


EVENT_EVIDENCE = [
    {
        "year": 2012,
        "expected_stress_group": "heat, drought",
        "affected_region_crop_scope": "Central United States and Plains crop belt; crop-state rows in the processed frame",
        "external_source": "NOAA/NCEI Annual 2012 Drought Report; NWS 2012 drought summary",
        "pre_specified": "yes",
    },
    {
        "year": 2021,
        "expected_stress_group": "heat, drought",
        "affected_region_crop_scope": "Northern Plains and Canadian Prairie drought context; U.S. crop-state rows in the processed frame",
        "external_source": "Drought.gov/NIDIS report on the 2020-2021 Northern Plains and Canadian Prairies drought",
        "pre_specified": "yes",
    },
    {
        "year": 2022,
        "expected_stress_group": "heat, drought, excess_rain",
        "affected_region_crop_scope": "U.S. drought and regional wetness episodes during the 2022 growing season",
        "external_source": "NOAA/NCEI Annual 2022 and August 2022 Drought Reports",
        "pre_specified": "yes",
    },
]


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    SUPPLEMENT.mkdir(parents=True, exist_ok=True)
    TABLE_CSV.mkdir(parents=True, exist_ok=True)
    (PAPER / "overleaf_zip").mkdir(parents=True, exist_ok=True)
    for stale in list(TABLES.glob("table*.tex")) + list(SUPPLEMENT.glob("tableS*.tex")):
        stale.unlink()
    for old_csv in list(TABLES.glob("*.csv")) + list(SUPPLEMENT.glob("*.csv")) + list(TABLE_CSV.glob("*.csv")):
        old_csv.unlink()
    for old_figure in FIGURES.glob("fig*.png"):
        old_figure.unlink()


def read_inputs() -> dict[str, pd.DataFrame]:
    return {
        "frame": pd.read_csv(ROOT / "data" / "processed" / "us_model_frame_hemisphere_aware_1990_2025.csv"),
        "yield_metrics": pd.read_csv(ROOT / "outputs" / "yield_model_metrics.csv"),
        "anomalies": pd.read_csv(ROOT / "outputs" / "low_yield_anomalies.csv"),
        "scorecard": pd.read_csv(ROOT / "improve_target" / "method_scorecard.csv"),
        "temporal_attr": pd.read_csv(
            ROOT
            / "improve_target"
            / "06_grouped_driver_scaa_temporal_holdout"
            / "outputs"
            / "temporal_holdout_attributions.csv"
        ),
        "claims": pd.read_csv(ROOT / "improve_target" / "crop_driver_claims.csv"),
        "event_validation": pd.read_csv(ROOT / "improve_target" / "event_validation_2012_2021_2022.csv"),
        "event_null_baselines": pd.read_csv(ROOT / "improve_target" / "event_consistency_null_baselines.csv"),
        "residual_validation": pd.read_csv(
            ROOT
            / "improve_target"
            / "06_grouped_driver_scaa_temporal_holdout"
            / "outputs"
            / "residual_model_validation.csv"
        ),
        "vulnerability": pd.read_csv(ROOT / "improve_target" / "crop_specific_vulnerability_profiles.csv"),
        "warning": pd.read_csv(ROOT / "improve_target" / "05_early_mid_warning_improved" / "outputs" / "warning_metrics.csv"),
    }


def validate_inputs(data: dict[str, pd.DataFrame]) -> None:
    frame = data["frame"]
    if len(frame) != 1257:
        raise AssertionError(f"Expected 1257 rows, found {len(frame)}")
    if int(frame["year"].min()) != 1990 or int(frame["year"].max()) != 2025:
        raise AssertionError("Expected year range 1990-2025")
    if set(frame["crop"].unique()) != {"Barley", "Canola", "Oats", "Wheat"}:
        raise AssertionError("Unexpected crop set")
    claims = data["claims"]
    if (claims["recovered_gap_t_ha"] > claims["yield_gap_t_ha"] + 1e-9).any():
        raise AssertionError("A claim recovers more than the observed detrended shortfall")
    if claims["claim_sentence"].isna().any():
        raise AssertionError("Missing claim sentence")
    temporal = data["temporal_attr"]
    if temporal.empty:
        raise AssertionError("Temporal-holdout grouped SCAA output is empty")
    if (temporal["recovered_gap_t_ha"] > temporal["yield_gap_t_ha"] + 1e-9).any():
        raise AssertionError("Temporal-holdout recovery exceeds observed detrended shortfall")
    residual = data["residual_validation"]
    if residual.empty or residual[["r2", "rmse_t_ha", "mae_t_ha"]].isna().any().any():
        raise AssertionError("Residual-model validation table must contain non-empty R2, RMSE, and MAE")
    nulls = data["event_null_baselines"]
    required_null_methods = {
        "Always drought",
        "Always heat",
        "Driver-frequency random",
        "Retrospective leave-one-event-year-out grouped SCAA",
    }
    if not required_null_methods.issubset(set(nulls["method"])):
        raise AssertionError("Event-year null baseline table is missing required methods")


def write_csv_and_tex(df: pd.DataFrame, csv_path: Path, tex_path: Path, caption: str, label: str, index: bool = False) -> None:
    df.to_csv(TABLE_CSV / csv_path.name, index=index)
    tex_path.write_text(latex_table(df, caption, label, index=index), encoding="utf-8")


def latex_escape(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def latex_table(df: pd.DataFrame, caption: str, label: str, index: bool = False) -> str:
    table = df.reset_index() if index else df.copy()
    cols = list(table.columns)
    alignment = "l" * len(cols)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(latex_escape(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(" & ".join(latex_escape(row[c]) for c in cols) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def write_event_null_table(df: pd.DataFrame) -> None:
    csv_path = TABLE_CSV / "table09_event_null_baselines.csv"
    tex_path = SUPPLEMENT / "tableS09_event_null_baselines.tex"
    df.to_csv(csv_path, index=False)
    cols = ["Method", "Expected match rate", "Median recovery", "n event rows"]
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Null baselines for event-year consistency checks.}",
        r"\label{tab:event_null_baselines}",
        r"\small",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccc@{}}",
        r"\toprule",
        " & ".join(latex_escape(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape(row[c]) for c in cols) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}"])
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def anomaly_keys(df: pd.DataFrame) -> set[tuple[str, str, int]]:
    return set(zip(df["crop"].astype(str), df["region"].astype(str), df["year"].astype(int)))


def score_linear_anomalies(frame: pd.DataFrame, threshold: float = -1.0) -> pd.DataFrame:
    pieces = []
    for (_, _), group in frame.groupby(["crop", "region"], sort=True):
        g = group.sort_values("year").copy()
        x = g["year"].to_numpy(dtype=float)
        y = g["yield_t_ha"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        trend = slope * x + intercept
        residual = y - trend
        std = np.std(residual, ddof=1)
        z = np.zeros_like(residual) if not np.isfinite(std) or std == 0 else residual / std
        g["trend_residual_z"] = z
        g["is_low_yield_anomaly"] = g["trend_residual_z"] < threshold
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def anomaly_membership_for_detrend(frame: pd.DataFrame, method: str) -> set[tuple[str, str, int]]:
    rows = []
    for (_, _), group in frame.groupby(["crop", "region"], sort=True):
        g = group.sort_values("year").copy()
        y = g["yield_t_ha"].to_numpy(dtype=float)
        x = g["year"].to_numpy(dtype=float)
        if method == "linear":
            coef = np.polyfit(x, y, 1)
            trend = np.polyval(coef, x)
        elif method == "quadratic":
            deg = 2 if len(g) >= 4 else 1
            coef = np.polyfit(x, y, deg)
            trend = np.polyval(coef, x)
        elif method == "centered_7yr_rolling":
            trend = (
                g["yield_t_ha"]
                .rolling(window=7, center=True, min_periods=3)
                .median()
                .bfill()
                .ffill()
                .to_numpy(dtype=float)
            )
        else:
            raise ValueError(f"Unknown detrending method: {method}")
        residual = y - trend
        std = np.std(residual, ddof=1)
        z = np.zeros_like(residual) if not np.isfinite(std) or std == 0 else residual / std
        g["is_low_yield_anomaly"] = z < -1.0
        rows.append(g[g["is_low_yield_anomaly"]])
    return anomaly_keys(pd.concat(rows, ignore_index=True)) if rows else set()


def build_threshold_sensitivity(frame: pd.DataFrame, temporal: pd.DataFrame) -> pd.DataFrame:
    if "trend_residual_z" not in temporal.columns:
        scored = score_linear_anomalies(frame)
        temporal = temporal.merge(
            scored[["crop", "region", "year", "trend_residual_z"]],
            on=["crop", "region", "year"],
            how="left",
        )
    rows = []
    for threshold in [-1.0, -1.5, -2.0]:
        subset = temporal[temporal["trend_residual_z"] < threshold].copy()
        top_groups = ", ".join(subset["driver_group"].value_counts().head(3).index.tolist()) if len(subset) else ""
        rows.append(
            {
                "threshold": f"z < {threshold:.1f}",
                "anomaly_count": int(len(subset)),
                "share_of_dataset": round(float(len(subset) / len(frame)), 3),
                "median_recoverable_fraction": "" if subset.empty else round(float(subset["recoverable_fraction"].median()), 3),
                "top_driver_groups": top_groups,
            }
        )
    return pd.DataFrame(rows)


def build_detrending_robustness(frame: pd.DataFrame) -> pd.DataFrame:
    linear = anomaly_membership_for_detrend(frame, "linear")
    rows = []
    for method in ["linear", "quadratic", "centered_7yr_rolling"]:
        keys = anomaly_membership_for_detrend(frame, method)
        overlap = len(keys & linear)
        union = len(keys | linear)
        rows.append(
            {
                "detrending_method": method,
                "anomaly_count": len(keys),
                "overlap_with_linear": overlap,
                "jaccard_with_linear": round(float(overlap / union), 3) if union else 1.0,
            }
        )
    return pd.DataFrame(rows)


def format_state_penalties_positive(value: object) -> str:
    text = str(value)

    def repl(match: re.Match[str]) -> str:
        return f"({abs(float(match.group(1))):.3f})"

    return re.sub(r"\((-?\d+(?:\.\d+)?)\)", repl, text)


def build_tables(data: dict[str, pd.DataFrame]) -> None:
    frame = data["frame"]
    anomalies = data["anomalies"]
    yield_metrics = data["yield_metrics"]
    scorecard = data["scorecard"]
    grouped = data["temporal_attr"]
    event = data["event_validation"]
    event_nulls = data["event_null_baselines"]
    residual_validation = data["residual_validation"]
    vulnerability = data["vulnerability"]
    warning = data["warning"]

    dataset_rows = []
    for crop, group in frame.groupby("crop"):
        dataset_rows.append(
            {
                "crop": crop,
                "observations": len(group),
                "states": group["region"].nunique(),
                "years": f"{int(group['year'].min())}-{int(group['year'].max())}",
                "windows": ", ".join(sorted(group["window"].unique())),
                "anomalies": int((anomalies["crop"] == crop).sum()),
            }
        )
    dataset_summary = pd.DataFrame(dataset_rows)
    write_csv_and_tex(
        dataset_summary,
        TABLES / "table01_dataset_summary.csv",
        TABLES / "table01_dataset_summary.tex",
        "Dataset summary by crop.",
        "tab:dataset_summary",
    )

    driver_df = pd.DataFrame(DRIVER_GROUPS)[["driver_group", "description"]]
    write_csv_and_tex(
        driver_df,
        TABLES / "table02_driver_groups.csv",
        TABLES / "table02_driver_groups.tex",
        "Extreme-weather driver groups used by grouped SCAA.",
        "tab:driver_groups",
    )
    driver_features_df = pd.DataFrame(DRIVER_GROUPS)[["driver_group", "features"]]
    write_csv_and_tex(
        driver_features_df,
        SUPPLEMENT / "tableS02_driver_group_features.csv",
        SUPPLEMENT / "tableS02_driver_group_features.tex",
        "Full feature list for each grouped-SCAA driver group.",
        "tab:driver_group_features",
    )

    metrics_rows = []
    for _, row in yield_metrics.iterrows():
        metrics_rows.append(
            {
                "evaluation": row["label"],
                "test_years": f"{int(row['test_start'])}-{int(row['test_end'])}",
                "n_test": int(row["n_test"]),
                "r2": round(float(row["r2"]), 3),
                "rmse_t_ha": round(float(row["rmse"]), 3),
            }
        )
    metrics_rows.append(
        {
            "evaluation": "low_yield_anomalies",
            "test_years": "1990-2025",
            "n_test": len(frame),
            "r2": "",
            "rmse_t_ha": f"{len(anomalies)} anomalies",
        }
    )
    model_table = pd.DataFrame(metrics_rows)
    write_csv_and_tex(
        model_table,
        TABLES / "table03_model_performance.csv",
        TABLES / "table03_model_performance.tex",
        "Forward-time yield performance and anomaly count.",
        "tab:model_performance",
    )

    residual_table = residual_validation.copy()
    protocol_labels = {
        "forward_time_train_1990_2015_test_2016_2025": ("Forward-time residual", "2016-2025"),
        "retrospective_leave_one_anomaly_year_out_all_rows": (
            "Leave-one-event-year-out residual",
            "All held-out rows",
        ),
        "retrospective_leave_one_anomaly_year_out_anomaly_rows": (
            "Leave-one-event-year-out residual",
            "Anomaly rows",
        ),
    }
    residual_table["held_out_unit"] = residual_table["protocol"].map(
        lambda key: protocol_labels.get(key, (key, ""))[1]
    )
    residual_table["protocol"] = residual_table["protocol"].map(
        lambda key: protocol_labels.get(key, (key, ""))[0]
    )
    residual_table["n_eval"] = residual_table["n_test"].astype(int)
    for col in ["r2", "rmse_t_ha", "mae_t_ha"]:
        residual_table[col] = residual_table[col].map(lambda x: round(float(x), 3))
    residual_table = residual_table[["protocol", "held_out_unit", "n_eval", "r2", "rmse_t_ha", "mae_t_ha"]]
    write_csv_and_tex(
        residual_table,
        TABLES / "table08_residual_model_validation.csv",
        TABLES / "table08_residual_model_validation.tex",
        "Residual-model validation for the attribution target.",
        "tab:residual_model_validation",
    )

    score_cols = [
        "method",
        "median_recoverable_fraction",
        "weather_driven_rate",
        "event_expected_match_rate",
    ]
    score_table = scorecard[score_cols].copy()
    score_table = score_table[score_table["method"].isin(METHOD_LABELS)].copy()
    score_table = score_table[score_table["median_recoverable_fraction"].notna()].copy()
    score_table["method"] = score_table["method"].map(METHOD_LABELS)
    for col in score_cols[1:]:
        score_table[col] = score_table[col].map(lambda x: "" if pd.isna(x) else round(float(x), 3))
    score_table = score_table.rename(columns={"weather_driven_rate": "high_recovery_rate"})
    write_csv_and_tex(
        score_table,
        TABLES / "table04_method_scorecard.csv",
        TABLES / "table04_method_scorecard.tex",
        "Comparison of attribution methods using recovery, high-recovery rate, and event-year agreement.",
        "tab:method_scorecard",
    )

    top_claims = grouped.sort_values(
        ["recoverable_fraction", "recovered_gap_t_ha"],
        ascending=[False, False],
    ).head(5)
    top_claims = top_claims[
        [
            "crop",
            "region",
            "year",
            "driver_group",
            "dominant_feature",
            "yield_gap_t_ha",
            "recovered_gap_t_ha",
            "recoverable_fraction",
        ]
    ].copy()
    for col in ["yield_gap_t_ha", "recovered_gap_t_ha", "recoverable_fraction"]:
        top_claims[col] = top_claims[col].round(3)
    write_csv_and_tex(
        top_claims,
        TABLES / "table05_top_event_claims.csv",
        TABLES / "table05_top_event_claims.tex",
        "Highest-recovery retrospective leave-one-event-year-out grouped-SCAA crop-region-year diagnostic cases.",
        "tab:top_claims",
    )

    temporal_event = event[event["method"] == "06_grouped_driver_scaa_temporal_holdout"].copy()
    event_source_table = pd.DataFrame(EVENT_EVIDENCE)
    write_csv_and_tex(
        event_source_table,
        SUPPLEMENT / "tableS08_event_evidence_sources.csv",
        SUPPLEMENT / "tableS08_event_evidence_sources.tex",
        "External evidence used to pre-specify expected event-year stress groups.",
        "tab:event_evidence_sources",
    )
    event_summary = (
        temporal_event.groupby(["method", "year"], as_index=False)
        .agg(
            n_events=("expected_match", "size"),
            expected_match_rate=("expected_match", "mean"),
            median_recoverable=("recoverable_fraction", "median"),
        )
        .sort_values(["method", "year"])
    )
    event_summary["expected_match_rate"] = event_summary["expected_match_rate"].round(3)
    event_summary["median_recoverable"] = event_summary["median_recoverable"].round(3)
    event_summary["method"] = event_summary["method"].map(METHOD_LABELS).fillna(event_summary["method"])
    write_csv_and_tex(
        event_summary,
        SUPPLEMENT / "tableS06_event_consistency_summary.csv",
        SUPPLEMENT / "tableS06_event_consistency_summary.tex",
        "Leave-one-event-year-out event-year consistency summary for 2012, 2021, and 2022.",
        "tab:event_consistency_summary",
    )

    null_table = event_nulls.copy()
    null_method_labels = {
        "Retrospective leave-one-event-year-out grouped SCAA": "Leave-one-event-year-out grouped SCAA",
    }
    null_table["method"] = null_table["method"].map(lambda x: null_method_labels.get(x, x))
    null_table["method"] = null_table["method"].str.replace(
        "Most frequent driver",
        "Most frequent event-year SCAA driver",
        regex=False,
    )
    null_table["expected_match_rate"] = null_table["expected_match_rate"].map(lambda x: f"{float(x):.3f}")
    null_table["median_recoverable_fraction"] = null_table["median_recoverable_fraction"].map(
        lambda x: "--" if pd.isna(x) or x == "" else f"{float(x):.3f}"
    )
    null_table["n_event_rows"] = null_table["n_event_rows"].astype(int)
    null_table = null_table.rename(
        columns={
            "method": "Method",
            "expected_match_rate": "Expected match rate",
            "median_recoverable_fraction": "Median recovery",
            "n_event_rows": "n event rows",
        }
    )[["Method", "Expected match rate", "Median recovery", "n event rows"]]
    write_event_null_table(null_table)

    vuln_table = vulnerability.sort_values("median_effect_t_ha").head(3)[
        ["crop", "driver_group", "median_effect_t_ha", "states_most_sensitive"]
    ].copy()
    vuln_table["median_yield_penalty_t_ha"] = (-vuln_table["median_effect_t_ha"]).round(3)
    vuln_table["states_largest_penalty"] = vuln_table["states_most_sensitive"].map(format_state_penalties_positive)
    vuln_table = vuln_table[
        ["crop", "driver_group", "median_yield_penalty_t_ha", "states_largest_penalty"]
    ]
    write_csv_and_tex(
        vuln_table,
        TABLES / "table07_crop_vulnerability.csv",
        TABLES / "table07_crop_vulnerability.tex",
        "Top crop-specific yield penalties under adverse observed weather extremes.",
        "tab:crop_vulnerability",
    )

    threshold_table = build_threshold_sensitivity(frame, grouped)
    write_csv_and_tex(
        threshold_table,
        SUPPLEMENT / "tableS03_anomaly_threshold_sensitivity.csv",
        SUPPLEMENT / "tableS03_anomaly_threshold_sensitivity.tex",
        "Sensitivity of leave-one-event-year-out SCAA summaries to the anomaly z-threshold.",
        "tab:threshold_sensitivity",
    )

    detrend_table = build_detrending_robustness(frame)
    write_csv_and_tex(
        detrend_table,
        SUPPLEMENT / "tableS04_detrending_robustness.csv",
        SUPPLEMENT / "tableS04_detrending_robustness.tex",
        "Robustness of low-yield anomaly membership to detrending choice.",
        "tab:detrending_robustness",
    )

    crop_state_rows = []
    for crop, group in frame.groupby("crop", sort=True):
        crop_state_rows.append(
            {
                "crop": crop,
                "observed_crop_state_pairs": int(group[["crop", "region"]].drop_duplicates().shape[0]),
                "observed_crop_state_year_rows": int(len(group)),
                "regions": ", ".join(sorted(group["region"].unique())),
            }
        )
    crop_state_table = pd.DataFrame(crop_state_rows)
    write_csv_and_tex(
        crop_state_table,
        SUPPLEMENT / "tableS05_observed_crop_state_pairs.csv",
        SUPPLEMENT / "tableS05_observed_crop_state_pairs.tex",
        "Observed crop-state support for vulnerability profiles.",
        "tab:observed_crop_state_pairs",
    )

    warning_table = warning[
        [
            "stage",
            "roc_auc",
            "average_precision",
            "brier_score",
            "top_10_precision",
            "top_20_precision",
            "threshold_from_calibration",
        ]
    ].copy()
    warning_table["stage"] = warning_table["stage"].map(
        {
            "early_third": "Early-season",
            "early_mid_two_thirds": "Early + mid-season",
        }
    ).fillna(warning_table["stage"])
    for col in warning_table.columns:
        if col != "stage":
            warning_table[col] = warning_table[col].map(lambda x: round(float(x), 3))
    write_csv_and_tex(
        warning_table,
        SUPPLEMENT / "tableS07_early_warning_metrics.csv",
        SUPPLEMENT / "tableS07_early_warning_metrics.tex",
        "Numeric early-warning performance corresponding to Figure 9.",
        "tab:early_warning_metrics",
    )

    ref_map = pd.DataFrame(REFERENCE_MAP, columns=["paper_section", "use", "bib_keys"])
    write_csv_and_tex(
        ref_map,
        SUPPLEMENT / "tableS01_reference_section_mapping.csv",
        SUPPLEMENT / "tableS01_reference_section_mapping.tex",
        "Reference mapping to manuscript sections.",
        "tab:reference_mapping",
    )


def fig_method_workflow() -> None:
    if not WORKFLOW_SOURCE.exists():
        raise FileNotFoundError(f"Missing workflow source image: {WORKFLOW_SOURCE}")
    shutil.copyfile(WORKFLOW_SOURCE, FIGURES / "fig01_method_workflow.png")


def fig_anomaly_timeline(anomalies: pd.DataFrame) -> None:
    counts = anomalies.groupby("year").size().reindex(range(1990, 2026), fill_value=0)
    colors = ["#c44747" if y in {2012, 2021, 2022} else "#4d7ea8" for y in counts.index]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.bar(counts.index, counts.values, color=colors)
    for y in [2012, 2021, 2022]:
        ax.text(y, counts.loc[y] + 0.5, str(y), ha="center", fontsize=9)
    ax.set_xlabel("Year")
    ax.set_ylabel("Anomaly count")
    ax.set_title("Detrended low-yield anomalies")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04_anomaly_timeline.png", dpi=220)
    plt.close(fig)


def fig_grouped_attribution(grouped: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].hist(grouped["recoverable_fraction"], bins=np.linspace(0, 1, 16), color="#517b9d", edgecolor="white")
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Recoverable fraction")
    axes[0].set_ylabel("Anomaly count")
    axes[0].set_title("Leave-one-event-year-out grouped-SCAA recovery")
    counts = grouped["driver_group"].value_counts().sort_values()
    counts.plot(kind="barh", ax=axes[1], color="#8b6f47")
    axes[1].set_xlabel("Attributed anomalies")
    axes[1].set_title("Dominant driver groups")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06_grouped_driver_attribution.png", dpi=220)
    plt.close(fig)


def fig_event_validation(event: pd.DataFrame) -> None:
    temporal = event[event["method"] == "06_grouped_driver_scaa_temporal_holdout"].copy()
    pivot = temporal.pivot_table(index="method", columns="year", values="expected_match", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(pivot.to_numpy(), cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i, j]:.0%}", ha="center", va="center", fontsize=9)
    ax.set_title("Event-year consistency with pre-specified stress groups")
    fig.colorbar(im, ax=ax, label="Expected-group match rate")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig08_event_validation.png", dpi=220)
    plt.close(fig)


def fig_warning(warning: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0))
    labels = warning["stage"].tolist()
    x = np.arange(len(labels))
    for ax, col, title, color in [
        (axes[0], "average_precision", "Average precision", "#4d7ea8"),
        (axes[1], "top_20_precision", "Top 20% precision", "#5f8f6b"),
        (axes[2], "brier_score", "Brier score", "#c47f47"),
    ]:
        ax.bar(x, warning[col], color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig09_early_warning.png", dpi=220)
    plt.close(fig)


def build_figures(data: dict[str, pd.DataFrame]) -> None:
    fig_method_workflow()
    fig_anomaly_timeline(data["anomalies"])
    fig_grouped_attribution(data["temporal_attr"])
    fig_event_validation(data["event_validation"])
    fig_warning(data["warning"])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(data: dict[str, pd.DataFrame]) -> None:
    files = [
        ROOT / "data" / "raw" / "us_yield_1989_2025_tha.csv",
        ROOT / "data" / "raw" / "nasa_power_daily.zip",
        ROOT / "data" / "processed" / "us_model_frame_hemisphere_aware_1990_2025.csv",
        ROOT / "outputs" / "yield_model_metrics.csv",
        ROOT / "outputs" / "low_yield_anomalies.csv",
        ROOT / "paper" / "DAP_new.pdf",
        ROOT / "paper" / "overleaf_zip" / "scaa_crop_yield_anomaly_attribution.zip",
    ]
    lines = [
        "# Data Manifest",
        "",
        "Generated by `python scripts/build_paper_assets.py`.",
        "",
        "## Dataset Checks",
        "",
        f"- Processed frame rows: {len(data['frame'])}",
        f"- Year range: {int(data['frame']['year'].min())}-{int(data['frame']['year'].max())}",
        f"- Crops: {', '.join(sorted(data['frame']['crop'].unique()))}",
        f"- Low-yield anomalies: {len(data['anomalies'])}",
        "",
        "## File Checksums",
        "",
        "| Path | Bytes | SHA256 |",
        "|---|---:|---|",
    ]
    for path in files:
        rel = path.relative_to(ROOT)
        lines.append(f"| `{rel}` | {path.stat().st_size} | `{sha256(path)}` |")
    (PAPER / "DATA_MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reproducibility() -> None:
    lines = [
        "# Reproducibility",
        "",
        "Run from the repository root.",
        "",
        "```powershell",
        "python -m venv .venv",
        ".\\.venv\\Scripts\\Activate.ps1",
        "pip install -r requirements.txt",
        "python scripts/run_all.py",
        "python scripts/run_improvement_experiments.py",
        "python scripts/build_paper_assets.py",
        "python scripts/package_overleaf.py",
        "```",
        "",
        "The paper uses `06_grouped_driver_scaa_temporal_holdout` as the implementation of retrospective leave-one-event-year-out grouped SCAA, keeps `02_grouped_driver_scaa` as an in-sample exploratory comparison, and uses `03_observed_analog_counterfactual` as a plausibility robustness check.",
        "",
        "The generated paper files are stored in `paper/latex_source/`; the Overleaf upload archive is stored in `paper/overleaf_zip/`.",
        "",
        "Local TeX compilation is optional. If a TeX distribution is unavailable, upload the Overleaf zip and compile there.",
        "",
        "Generated experiment workspaces such as `improve_target/` are intentionally omitted from the public branch and can be recreated with `python scripts/run_improvement_experiments.py`.",
    ]
    (PAPER / "REPRODUCIBILITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_supplement_tex() -> None:
    content = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage{caption}
\usepackage{orcidlink}
\usepackage{hyperref}
\hypersetup{colorlinks=true,allcolors=black}
\graphicspath{{figures/}}
\raggedbottom
\setlength{\textfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\floatsep}{8pt plus 2pt minus 2pt}
\setlength{\intextsep}{8pt plus 2pt minus 2pt}
\captionsetup{skip=4pt}
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.75}

\title{\textbf{Supplementary Material}\\Sparse Counterfactual Attribution of Crop-Yield Anomalies}
\author{
\begin{tabular}{c}
Tran Dai Phong Lam~\orcidlink{0009-0003-9320-8782}\thanks{Corresponding author.}, Thu Le~\orcidlink{0009-0008-3480-8311}\\
Nguyen Quoc Hung~\orcidlink{0009-0002-7538-7978}, Nguyen Trung Trinh~\orcidlink{0009-0003-5566-3469}\\[0.6em]
\small FPT University, Ho Chi Minh City, Vietnam\\
\small \href{mailto:phonglam2599@gmail.com}{\texttt{phonglam2599@gmail.com}},
\href{mailto:thulvn@fpt.edu.vn}{\texttt{thulvn@fpt.edu.vn}}\\
\small
\href{mailto:hungtvt2222@gmail.com}{\texttt{hungtvt2222@gmail.com}},
\href{mailto:trinhnguyen112355@gmail.com}{\texttt{trinhnguyen112355@gmail.com}}
\end{tabular}
}
\date{}

\begin{document}
\maketitle

\section{Driver Features and Robustness Tables}
\input{supplement/tableS02_driver_group_features.tex}
\input{supplement/tableS03_anomaly_threshold_sensitivity.tex}
\input{supplement/tableS04_detrending_robustness.tex}
\input{supplement/tableS05_observed_crop_state_pairs.tex}

\section{Event-Year Consistency Details}
\input{supplement/tableS08_event_evidence_sources.tex}
\input{supplement/tableS06_event_consistency_summary.tex}
\input{supplement/tableS09_event_null_baselines.tex}

\begin{figure}[!htbp]
  \centering
  \includegraphics[width=0.9\linewidth]{fig08_event_validation.png}
  \caption{Event-year consistency with pre-specified heat, drought, and moisture stress groups.}
  \label{fig:event_consistency_supp}
\end{figure}

\section{Early-Warning Extension}
\input{supplement/tableS07_early_warning_metrics.tex}

\begin{figure}[!htbp]
  \centering
  \includegraphics[width=\linewidth]{fig09_early_warning.png}
  \caption{Early-warning extension comparing early-season and early-plus-mid-season anomaly risk models.}
  \label{fig:warning_supp}
\end{figure}

\section{Reference Mapping}
\input{supplement/tableS01_reference_section_mapping.tex}

\end{document}
"""
    (LATEX / "supplement.tex").write_text(content, encoding="utf-8")


def assert_outputs() -> None:
    expected_figures = [
        "fig01_method_workflow.png",
        "fig04_anomaly_timeline.png",
        "fig06_grouped_driver_attribution.png",
        "fig08_event_validation.png",
        "fig09_early_warning.png",
    ]
    expected_tables = [
        "table01_dataset_summary.tex",
        "table02_driver_groups.tex",
        "table03_model_performance.tex",
        "table04_method_scorecard.tex",
        "table05_top_event_claims.tex",
        "table07_crop_vulnerability.tex",
        "table08_residual_model_validation.tex",
    ]
    expected_supplement = [
        "tableS01_reference_section_mapping.tex",
        "tableS02_driver_group_features.tex",
        "tableS03_anomaly_threshold_sensitivity.tex",
        "tableS04_detrending_robustness.tex",
        "tableS05_observed_crop_state_pairs.tex",
        "tableS06_event_consistency_summary.tex",
        "tableS07_early_warning_metrics.tex",
        "tableS08_event_evidence_sources.tex",
        "tableS09_event_null_baselines.tex",
    ]
    missing = [str(FIGURES / name) for name in expected_figures if not (FIGURES / name).exists()]
    missing += [str(TABLES / name) for name in expected_tables if not (TABLES / name).exists()]
    missing += [str(SUPPLEMENT / name) for name in expected_supplement if not (SUPPLEMENT / name).exists()]
    if not (LATEX / "supplement.tex").exists():
        missing.append(str(LATEX / "supplement.tex"))
    if missing:
        raise AssertionError(f"Missing paper assets: {missing}")
    for figure in expected_figures:
        img = plt.imread(FIGURES / figure)
        if img.size == 0 or float(img.std()) == 0.0:
            raise AssertionError(f"Blank figure: {figure}")


def main() -> None:
    ensure_dirs()
    data = read_inputs()
    validate_inputs(data)
    build_tables(data)
    build_figures(data)
    write_manifest(data)
    write_reproducibility()
    write_supplement_tex()
    assert_outputs()
    print("Paper assets built.")
    print(f"Figures: {FIGURES}")
    print(f"Tables: {TABLES}")


if __name__ == "__main__":
    main()
