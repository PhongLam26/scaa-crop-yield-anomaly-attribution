from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
LATEX = PAPER / "latex_source"
FIGURES = LATEX / "figures"
TABLES = LATEX / "tables"


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
    (PAPER / "reference_pack").mkdir(parents=True, exist_ok=True)
    (PAPER / "overleaf_zip").mkdir(parents=True, exist_ok=True)


def read_inputs() -> dict[str, pd.DataFrame]:
    return {
        "frame": pd.read_csv(ROOT / "data" / "processed" / "us_model_frame_hemisphere_aware_1990_2025.csv"),
        "yield_metrics": pd.read_csv(ROOT / "outputs" / "yield_model_metrics.csv"),
        "anomalies": pd.read_csv(ROOT / "outputs" / "low_yield_anomalies.csv"),
        "scorecard": pd.read_csv(ROOT / "improve_target" / "method_scorecard.csv"),
        "grouped_attr": pd.read_csv(ROOT / "improve_target" / "02_grouped_driver_scaa" / "outputs" / "grouped_driver_attributions.csv"),
        "temporal_attr": pd.read_csv(
            ROOT
            / "improve_target"
            / "06_grouped_driver_scaa_temporal_holdout"
            / "outputs"
            / "temporal_holdout_attributions.csv"
        ),
        "analog_attr": pd.read_csv(ROOT / "improve_target" / "03_observed_analog_counterfactual" / "outputs" / "observed_analog_attributions.csv"),
        "claims": pd.read_csv(ROOT / "improve_target" / "crop_driver_claims.csv"),
        "event_validation": pd.read_csv(ROOT / "improve_target" / "event_validation_2012_2021_2022.csv"),
        "vulnerability": pd.read_csv(ROOT / "improve_target" / "crop_specific_vulnerability_profiles.csv"),
        "warning": pd.read_csv(ROOT / "improve_target" / "05_early_mid_warning_improved" / "outputs" / "warning_metrics.csv"),
        "predictions": pd.read_csv(ROOT / "outputs" / "yield_predictions_2016_2021.csv"),
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


def write_csv_and_tex(df: pd.DataFrame, csv_path: Path, tex_path: Path, caption: str, label: str, index: bool = False) -> None:
    df.to_csv(csv_path, index=index)
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
        r"\begin{table}[H]",
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


def build_tables(data: dict[str, pd.DataFrame]) -> None:
    frame = data["frame"]
    anomalies = data["anomalies"]
    yield_metrics = data["yield_metrics"]
    scorecard = data["scorecard"]
    grouped = data["temporal_attr"]
    event = data["event_validation"]
    vulnerability = data["vulnerability"]

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
        TABLES / "tableS02_driver_group_features.csv",
        TABLES / "tableS02_driver_group_features.tex",
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

    score_cols = [
        "method",
        "median_recoverable_fraction",
        "weather_driven_rate",
        "event_expected_match_rate",
    ]
    score_table = scorecard[score_cols].copy()
    for col in score_cols[1:]:
        score_table[col] = score_table[col].map(lambda x: "" if pd.isna(x) else round(float(x), 3))
    write_csv_and_tex(
        score_table,
        TABLES / "table04_method_scorecard.csv",
        TABLES / "table04_method_scorecard.tex",
        "Method comparison by raw metrics; composite scores are not used in the manuscript.",
        "tab:method_scorecard",
    )

    top_claims = grouped.sort_values(["recoverable_fraction", "recovered_gap_t_ha"], ascending=False).head(5)
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
        "Top temporal-holdout grouped-SCAA crop-region-year attribution claims.",
        "tab:top_claims",
    )

    temporal_event = event[event["method"] == "06_grouped_driver_scaa_temporal_holdout"].copy()
    event_source_table = pd.DataFrame(EVENT_EVIDENCE)
    write_csv_and_tex(
        event_source_table,
        TABLES / "table06_event_evidence_sources.csv",
        TABLES / "table06_event_evidence_sources.tex",
        "External evidence used to pre-specify expected event-year stress groups.",
        "tab:event_evidence_sources",
    )
    write_csv_and_tex(
        event_source_table,
        TABLES / "table_event_evidence_sources.csv",
        TABLES / "table_event_evidence_sources.tex",
        "External evidence used to pre-specify expected event-year stress groups.",
        "tab:event_evidence_sources_alias",
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
    write_csv_and_tex(
        event_summary,
        TABLES / "tableS06_event_consistency_summary.csv",
        TABLES / "tableS06_event_consistency_summary.tex",
        "Temporal-holdout event-year consistency summary for 2012, 2021, and 2022.",
        "tab:event_consistency_summary",
    )

    vuln_table = vulnerability.sort_values("median_effect_t_ha").head(7)[
        ["crop", "driver_group", "median_effect_t_ha", "effect_direction", "states_most_sensitive"]
    ].copy()
    vuln_table["median_effect_t_ha"] = vuln_table["median_effect_t_ha"].round(3)
    write_csv_and_tex(
        vuln_table,
        TABLES / "table07_crop_vulnerability.csv",
        TABLES / "table07_crop_vulnerability.tex",
        "Crop-specific vulnerability profiles under adverse observed weather extremes.",
        "tab:crop_vulnerability",
    )

    threshold_table = build_threshold_sensitivity(frame, grouped)
    write_csv_and_tex(
        threshold_table,
        TABLES / "tableS03_anomaly_threshold_sensitivity.csv",
        TABLES / "tableS03_anomaly_threshold_sensitivity.tex",
        "Sensitivity of temporal-holdout SCAA summaries to the anomaly z-threshold.",
        "tab:threshold_sensitivity",
    )

    detrend_table = build_detrending_robustness(frame)
    write_csv_and_tex(
        detrend_table,
        TABLES / "tableS04_detrending_robustness.csv",
        TABLES / "tableS04_detrending_robustness.tex",
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
        TABLES / "tableS05_observed_crop_state_pairs.csv",
        TABLES / "tableS05_observed_crop_state_pairs.tex",
        "Observed crop-state support for vulnerability profiles.",
        "tab:observed_crop_state_pairs",
    )

    ref_map = pd.DataFrame(REFERENCE_MAP, columns=["paper_section", "use", "bib_keys"])
    write_csv_and_tex(
        ref_map,
        TABLES / "tableS01_reference_section_mapping.csv",
        TABLES / "tableS01_reference_section_mapping.tex",
        "Reference-pack mapping to manuscript sections.",
        "tab:reference_mapping",
    )


def fig_method_workflow() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0, 1)
    box_w = 0.15
    box_h = 0.22
    boxes = [
        ("Daily NASA POWER\nweather", 0.02, 0.58),
        ("Growing-season\nextreme features", 0.22, 0.58),
        ("USDA yield\nby crop-state-year", 0.02, 0.18),
        ("Detrend each\ncrop-state series", 0.22, 0.18),
        ("Low-yield\nanomaly events", 0.42, 0.18),
        ("Residual weather\nmodel", 0.42, 0.58),
        ("Grouped sparse\ncounterfactual", 0.64, 0.58),
        ("Crop-specific event\nclaim and recovery", 0.89, 0.38),
    ]
    for text, x, y in boxes:
        patch = FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.02", linewidth=1.3, edgecolor="#333333", facecolor="#f1f5f9")
        ax.add_patch(patch)
        ax.text(x + box_w / 2, y + box_h / 2, text, ha="center", va="center", fontsize=10)
    arrows = [
        ((0.17, 0.69), (0.22, 0.69)),
        ((0.17, 0.29), (0.22, 0.29)),
        ((0.37, 0.29), (0.42, 0.29)),
        ((0.37, 0.69), (0.42, 0.69)),
        ((0.57, 0.69), (0.64, 0.69)),
        ((0.57, 0.29), (0.66, 0.60)),
        ((0.79, 0.69), (0.865, 0.52)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=16, linewidth=1.2, color="#333333"))
    ax.text(
        0.5,
        0.03,
        "SCAA explains detrended low-yield events with sparse, feasible, physically grouped weather changes.",
        ha="center",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01_method_workflow.png", dpi=220)
    plt.close(fig)


def fig_data_coverage(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    crop_counts = frame["crop"].value_counts().sort_index()
    axes[0].bar(crop_counts.index, crop_counts.values, color="#4d7ea8")
    axes[0].set_title("Observations by crop")
    axes[0].set_ylabel("Crop-state-year rows")
    axes[0].grid(axis="y", alpha=0.25)

    pivot = frame.pivot_table(index="crop", columns="region", values="year", aggfunc="count").fillna(0)
    im = axes[1].imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu")
    axes[1].set_xticks(range(len(pivot.columns)))
    axes[1].set_xticklabels(pivot.columns, rotation=65, ha="right", fontsize=8)
    axes[1].set_yticks(range(len(pivot.index)))
    axes[1].set_yticklabels(pivot.index)
    axes[1].set_title("Crop coverage across states")
    fig.colorbar(im, ax=axes[1], label="Rows")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02_data_coverage.png", dpi=220)
    plt.close(fig)


def fig_prediction(predictions: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    ax.scatter(predictions["yield_t_ha"], predictions["predicted_yield_t_ha"], s=30, alpha=0.72, color="#4d7ea8")
    lo = min(predictions["yield_t_ha"].min(), predictions["predicted_yield_t_ha"].min())
    hi = max(predictions["yield_t_ha"].max(), predictions["predicted_yield_t_ha"].max())
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel("Observed yield (t/ha)")
    ax.set_ylabel("Predicted yield (t/ha)")
    ax.set_title("Forward-time test: 2016-2021")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03_yield_prediction.png", dpi=220)
    plt.close(fig)


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


def fig_method_scorecard(scorecard: pd.DataFrame) -> None:
    df = scorecard[["method", "median_recoverable_fraction", "weather_driven_rate", "event_expected_match_rate"]].copy()
    df = df.sort_values("median_recoverable_fraction")
    x = np.arange(len(df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x - width, df["median_recoverable_fraction"], width, label="Median recovery", color="#4d7ea8")
    ax.bar(x, df["weather_driven_rate"], width, label="Weather-driven rate", color="#5f8f6b")
    ax.bar(x + width, df["event_expected_match_rate"], width, label="Event-year consistency", color="#c47f47")
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate or fraction")
    ax.set_title("Method comparison by raw metrics")
    ax.legend(loc="upper left", ncols=3, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05_method_scorecard.png", dpi=220)
    plt.close(fig)


def fig_grouped_attribution(grouped: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].hist(grouped["recoverable_fraction"], bins=np.linspace(0, 1, 16), color="#517b9d", edgecolor="white")
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Recoverable fraction")
    axes[0].set_ylabel("Anomaly count")
    axes[0].set_title("Temporal-holdout grouped-SCAA recovery")
    counts = grouped["driver_group"].value_counts().sort_values()
    counts.plot(kind="barh", ax=axes[1], color="#8b6f47")
    axes[1].set_xlabel("Attributed anomalies")
    axes[1].set_title("Dominant driver groups")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06_grouped_driver_attribution.png", dpi=220)
    plt.close(fig)


def fig_vulnerability(vulnerability: pd.DataFrame) -> None:
    pivot = vulnerability.pivot(index="crop", columns="driver_group", values="median_effect_t_ha")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    vmax = max(abs(float(np.nanmin(pivot.to_numpy()))), abs(float(np.nanmax(pivot.to_numpy()))), 0.1)
    im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i, crop in enumerate(pivot.index):
        for j, group in enumerate(pivot.columns):
            value = pivot.loc[crop, group]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Crop-specific vulnerability under adverse observed extremes")
    fig.colorbar(im, ax=ax, label="Median residual effect (t/ha)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig07_crop_driver_vulnerability.png", dpi=220)
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
    fig_data_coverage(data["frame"])
    fig_prediction(data["predictions"])
    fig_anomaly_timeline(data["anomalies"])
    fig_method_scorecard(data["scorecard"])
    fig_grouped_attribution(data["temporal_attr"])
    fig_vulnerability(data["vulnerability"])
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
        ROOT / "improve_target" / "method_scorecard.csv",
        ROOT / "improve_target" / "crop_driver_claims.csv",
        ROOT
        / "improve_target"
        / "06_grouped_driver_scaa_temporal_holdout"
        / "outputs"
        / "temporal_holdout_attributions.csv",
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
        "The paper uses `06_grouped_driver_scaa_temporal_holdout` as the main attribution method, keeps `02_grouped_driver_scaa` as an in-sample exploratory comparison, and uses `03_observed_analog_counterfactual` as a plausibility robustness check.",
        "",
        "The generated paper files are stored in `paper/latex_source/`; the Overleaf upload archive is stored in `paper/overleaf_zip/`.",
        "",
        "Local TeX compilation is optional. If a TeX distribution is unavailable, upload the Overleaf zip and compile there.",
        "",
        "Reference metadata was extracted from `paper/reference_pack/crop_yield_anomaly_attribution_references.docx`; preprint and future-year entries are tracked in `paper/REFERENCE_AUDIT.md` rather than in submit-ready BibTeX notes.",
    ]
    (PAPER / "REPRODUCIBILITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_reference_pack() -> None:
    src = ROOT.parent / "crop_yield_anomaly_attribution_references.docx"
    dst = PAPER / "reference_pack" / "crop_yield_anomaly_attribution_references.docx"
    if src.exists():
        shutil.copy2(src, dst)


def write_reference_audit() -> None:
    lines = [
        "# Reference Audit",
        "",
        "The BibTeX file is derived from `paper/reference_pack/crop_yield_anomaly_attribution_references.docx`.",
        "",
        "## Core References Used In The Draft",
        "",
        "- Detrending and yield variability: Ray2015, Lu2017, Meng2024.",
        "- Extreme-weather yield loss: Lesk2016, Zampieri2017, Vogel2019, Heino2023, Sjulgard2023.",
        "- Yield prediction baselines: Paudel2021, Meroni2021, Khaki2019, LengHall2020.",
        "- Counterfactual explanation: Wachter2018, Mothilal2020, Ustun2019, Poyiadzi2020, Verma2024.",
        "- Event-attribution caution: Hannart2016, Otto2017, Oldenborgh2021.",
        "- External event-year evidence: NOAA2012Drought, DroughtGov2021NorthernPlains, NOAA2022Drought, NOAA2022AugustDrought.",
        "- Early warning and uncertainty: Anderson2024, Singh2024, Farag2025.",
        "",
        "## Submission Caution",
        "",
        "The reference pack contains 2025-2026 and preprint entries. Keep them for drafting, but verify DOI, volume, issue, and final publication status before formal submission.",
        "",
        "The submit-facing `paper/latex_source/references.bib` intentionally does not contain internal verification notes.",
    ]
    (PAPER / "REFERENCE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def assert_outputs() -> None:
    expected_figures = [
        "fig01_method_workflow.png",
        "fig02_data_coverage.png",
        "fig03_yield_prediction.png",
        "fig04_anomaly_timeline.png",
        "fig05_method_scorecard.png",
        "fig06_grouped_driver_attribution.png",
        "fig07_crop_driver_vulnerability.png",
        "fig08_event_validation.png",
        "fig09_early_warning.png",
    ]
    expected_tables = [
        "table01_dataset_summary.tex",
        "table02_driver_groups.tex",
        "table03_model_performance.tex",
        "table04_method_scorecard.tex",
        "table05_top_event_claims.tex",
        "table06_event_evidence_sources.tex",
        "table07_crop_vulnerability.tex",
        "tableS01_reference_section_mapping.tex",
        "tableS02_driver_group_features.tex",
        "tableS03_anomaly_threshold_sensitivity.tex",
        "tableS04_detrending_robustness.tex",
        "tableS05_observed_crop_state_pairs.tex",
        "tableS06_event_consistency_summary.tex",
    ]
    missing = [str(FIGURES / name) for name in expected_figures if not (FIGURES / name).exists()]
    missing += [str(TABLES / name) for name in expected_tables if not (TABLES / name).exists()]
    if missing:
        raise AssertionError(f"Missing paper assets: {missing}")
    for figure in expected_figures:
        img = plt.imread(FIGURES / figure)
        if img.size == 0 or float(img.std()) == 0.0:
            raise AssertionError(f"Blank figure: {figure}")


def main() -> None:
    ensure_dirs()
    copy_reference_pack()
    data = read_inputs()
    validate_inputs(data)
    build_tables(data)
    build_figures(data)
    write_manifest(data)
    write_reproducibility()
    write_reference_audit()
    assert_outputs()
    print("Paper assets built.")
    print(f"Figures: {FIGURES}")
    print(f"Tables: {TABLES}")


if __name__ == "__main__":
    main()
