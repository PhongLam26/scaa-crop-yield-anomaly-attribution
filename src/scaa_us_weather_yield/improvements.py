from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .pipeline import (
    ANOMALY_STD_DDOF,
    ANOMALY_Z_THRESHOLD,
    CAT_FEATURES,
    RANDOM_STATE,
    TARGET,
    WEATHER_DRIVEN_THRESHOLD,
    attribution_features,
    build_classifier,
    conformal_quantile,
    feasible_stats,
    load_frame,
    make_paths,
    r2_score_manual,
    safe_auc,
    validate_frame,
)


ROOT_NAME = "improve_target"
EVENT_YEARS = {2012, 2021, 2022}
EXPECTED_EVENT_GROUPS = {
    2012: {"heat", "drought"},
    2021: {"heat", "drought"},
    2022: {"heat", "drought", "excess_rain"},
}
METHODS = [
    "00_baseline_v1_raw_yield_scaa",
    "01_residual_target_scaa",
    "02_grouped_driver_scaa",
    "06_grouped_driver_scaa_temporal_holdout",
    "03_observed_analog_counterfactual",
    "04_crop_specific_vulnerability_profiles",
    "05_early_mid_warning_improved",
]
LEAKAGE_TOKENS = (
    "trend_",
    "is_low_yield_anomaly",
    "anomaly_label",
    "predicted_yield",
    "residual_observed",
)


@dataclass(frozen=True)
class MethodPaths:
    root: Path
    outputs: Path
    figures: Path
    results: Path
    method_note: Path


def method_paths(improve_root: Path, method: str) -> MethodPaths:
    root = improve_root / method
    outputs = root / "outputs"
    figures = root / "figures"
    outputs.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return MethodPaths(
        root=root,
        outputs=outputs,
        figures=figures,
        results=root / "RESULTS.md",
        method_note=root / "METHOD_NOTE.md",
    )


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def score_anomalies_no_write(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces: list[pd.DataFrame] = []
    for (crop, region), group in df.groupby(["crop", "region"], sort=True):
        g = group.sort_values("year").copy()
        x = g["year"].to_numpy(dtype=float)
        y = g[TARGET].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        trend = slope * x + intercept
        residual = y - trend
        std = np.std(residual, ddof=ANOMALY_STD_DDOF)
        z = np.zeros_like(residual) if not np.isfinite(std) or std == 0 else residual / std
        g["trend_yield_t_ha"] = trend
        g["trend_residual_t_ha"] = residual
        g["trend_residual_z"] = z
        g["is_low_yield_anomaly"] = g["trend_residual_z"] < ANOMALY_Z_THRESHOLD
        pieces.append(g)
    scored = pd.concat(pieces, ignore_index=True).sort_values(["year", "crop", "region"])
    anomalies = scored[scored["is_low_yield_anomaly"]].copy()
    return scored, anomalies


def full_weather_features(df: pd.DataFrame) -> list[str]:
    features = attribution_features(df)
    bad = [f for f in features if is_leakage_feature(f)]
    if bad:
        raise AssertionError(f"Leakage features detected: {bad}")
    return features


def stage_features(df: pd.DataFrame, suffix: str) -> list[str]:
    cols = [c for c in df.columns if c.endswith(suffix)]
    bad = [f for f in cols if is_leakage_feature(f)]
    if bad:
        raise AssertionError(f"Leakage stage features detected: {bad}")
    return cols


def is_leakage_feature(feature: str) -> bool:
    return any(token in feature for token in LEAKAGE_TOKENS)


def feature_group(feature: str) -> str:
    if feature in {"rain_sum", "rain_mean"}:
        return "drought"
    if "dry" in feature or "dry_spell" in feature:
        return "drought"
    if "heat" in feature or "heatwave" in feature or feature == "season_tmax_mean":
        return "heat"
    if feature in {"season_tmean_mean", "growing_degree_days_base5"}:
        return "heat"
    if "frost" in feature or "cold" in feature or feature == "min_tmin":
        return "frost_cold"
    if "heavy_rain" in feature or "wet_days" in feature or feature.startswith("max_") and "rain" in feature:
        return "excess_rain"
    if "radiation" in feature:
        return "radiation"
    if feature == "season_tmin_mean":
        return "frost_cold"
    return "other_weather"


def group_features(features: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "heat": [],
        "drought": [],
        "frost_cold": [],
        "excess_rain": [],
        "radiation": [],
    }
    for feature in features:
        group = feature_group(feature)
        if group in groups:
            groups[group].append(feature)
    return {k: v for k, v in groups.items() if v}


def group_label(group: str) -> str:
    return {
        "heat": "heat exposure",
        "drought": "drought/dry-spell stress",
        "frost_cold": "frost/cold stress",
        "excess_rain": "excess rainfall or wetness",
        "radiation": "radiation anomaly",
        "other_weather": "other weather stress",
    }.get(group, group)


def make_regressor(numeric: list[str], categorical: list[str]) -> Pipeline:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = ExtraTreesRegressor(
        n_estimators=350,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", pre), ("model", model)])


def make_classifier(numeric: list[str], categorical: list[str]) -> Pipeline:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = ExtraTreesClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", pre), ("model", model)])


def residual_columns(features: list[str]) -> tuple[list[str], list[str]]:
    return ["lat", "lon"] + features, CAT_FEATURES


def predict_one(model: Pipeline, row: pd.Series, numeric: list[str], categorical: list[str]) -> float:
    return float(model.predict(pd.DataFrame([row[numeric + categorical].to_dict()]))[0])


def predict_rows(model: Pipeline, rows: list[pd.Series], numeric: list[str], categorical: list[str]) -> np.ndarray:
    return model.predict(pd.DataFrame([r[numeric + categorical].to_dict() for r in rows]))


def regression_summary(y_true: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    return {
        "r2": r2_score_manual(y, p),
        "rmse_t_ha": float(math.sqrt(np.mean((y - p) ** 2))),
        "mae_t_ha": float(np.mean(np.abs(y - p))),
    }


def fit_residual_model(scored: pd.DataFrame, features: list[str]) -> tuple[Pipeline, list[str], list[str], dict[str, Any]]:
    numeric, categorical = residual_columns(features)
    train = scored[scored["year"] <= 2015].copy()
    test = scored[scored["year"] >= 2016].copy()
    model = make_regressor(numeric, categorical)
    model.fit(train[numeric + categorical], train["trend_residual_t_ha"])
    pred = model.predict(test[numeric + categorical])
    summary = regression_summary(test["trend_residual_t_ha"], pred)
    metrics = {
        "residual_r2_2016_2025": summary["r2"],
        "residual_rmse_2016_2025": summary["rmse_t_ha"],
        "residual_mae_2016_2025": summary["mae_t_ha"],
        "n_train": len(train),
        "n_test": len(test),
    }
    final = make_regressor(numeric, categorical)
    final.fit(scored[numeric + categorical], scored["trend_residual_t_ha"])
    return final, numeric, categorical, metrics


def standardized_delta(stats: pd.DataFrame, feature: str, from_value: float, to_value: float) -> float:
    if feature not in stats.index:
        return 0.0
    std = float(stats.loc[feature, "std"])
    if not np.isfinite(std) or std == 0:
        std = 1.0
    return abs(float(to_value) - float(from_value)) / std


def compact_changes(detail: list[dict[str, Any]]) -> str:
    return json.dumps(
        [
            {
                "feature": d["feature"],
                "from": round(float(d["from_value"]), 4),
                "to": round(float(d["to_value"]), 4),
                "standardized_delta": round(float(d["standardized_delta"]), 4),
                "prediction_gain_t_ha": round(float(d["prediction_gain_t_ha"]), 4),
            }
            for d in detail
        ]
    )


def claim_sentence(row: pd.Series) -> str:
    recovered = float(row["recovered_gap_t_ha"])
    fraction = float(row["recoverable_fraction"])
    gap = float(row["yield_gap_t_ha"])
    return (
        f"In {row['region']} {row['crop']} {int(row['year'])}, "
        f"{group_label(row['driver_group'])} was the dominant modelled driver; "
        f"moving {row['dominant_feature']} toward feasible normal levels recovered "
        f"{recovered:.3f} t/ha ({fraction:.1%}) of the {gap:.3f} t/ha detrended yield shortfall."
    )


def common_claim_columns(df: pd.DataFrame, method: str) -> pd.DataFrame:
    out = df.copy()
    out["method"] = method
    if "driver_group" not in out.columns:
        out["driver_group"] = out["dominant_feature"].map(feature_group)
    out["claim_sentence"] = out.apply(claim_sentence, axis=1)
    cols = [
        "method",
        "crop",
        "region",
        "year",
        "driver_group",
        "dominant_feature",
        "observed_yield_t_ha",
        "trend_yield_t_ha",
        "yield_gap_t_ha",
        "recovered_gap_t_ha",
        "recoverable_fraction",
        "claim_sentence",
    ]
    return out[cols]


def plot_recoverable(df: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(df["recoverable_fraction"], bins=np.linspace(0, 1, 16), color="#517b9d", edgecolor="white")
    ax.axvline(WEATHER_DRIVEN_THRESHOLD, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Recoverable fraction")
    ax.set_ylabel("Anomaly count")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_driver_counts(df: pd.DataFrame, path: Path, title: str, by_group: bool = True) -> None:
    col = "driver_group" if by_group else "dominant_feature"
    counts = df[col].value_counts().head(12).sort_values()
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    counts.plot(kind="barh", ax=ax, color="#8b6f47")
    ax.set_xlabel("Attributed anomaly count")
    ax.set_ylabel(col)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_crop_driver_heatmap(df: pd.DataFrame, path: Path, title: str) -> None:
    pivot = df.pivot_table(index="crop", columns="driver_group", values="recoverable_fraction", aggfunc="median")
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    im = ax.imshow(pivot.fillna(0).to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Median recoverable fraction")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_attribution_results(paths: MethodPaths, title: str, note: str, attr: pd.DataFrame, extra: dict[str, Any]) -> None:
    median_phi = float(attr["recoverable_fraction"].median()) if len(attr) else 0.0
    wd_rate = float((attr["recoverable_fraction"] >= WEATHER_DRIVEN_THRESHOLD).mean()) if len(attr) else 0.0
    top = attr["driver_group"].value_counts().head(8)
    lines = [
        f"# {title}",
        "",
        note,
        "",
        f"- Anomalies attributed: {len(attr)}",
        f"- Median recoverable fraction: {median_phi:.3f}",
        f"- High-recovery rate at phi >= {WEATHER_DRIVEN_THRESHOLD:.1f}: {wd_rate:.1%}",
    ]
    for key, value in extra.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.3f}")
        else:
            lines.append(f"- {key}: {value}")
    lines.extend(["", "## Top Driver Groups", ""])
    for group, count in top.items():
        lines.append(f"- {group}: {int(count)}")
    write_lines(paths.results, lines)


def write_method_note(paths: MethodPaths, title: str, bullets: list[str]) -> None:
    lines = [f"# {title}", ""]
    lines.extend([f"- {b}" for b in bullets])
    write_lines(paths.method_note, lines)


def run_baseline_v1(root: Path, improve_root: Path, scored: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = method_paths(improve_root, "00_baseline_v1_raw_yield_scaa")
    source_attr = root / "outputs" / "counterfactual_attributions.csv"
    if not source_attr.exists():
        raise AssertionError("Run python scripts/run_all.py before improvement experiments.")
    attr = pd.read_csv(source_attr)
    recovered = attr["model_gap_to_trend_t_ha"].clip(lower=0) * attr["weather_recoverable_fraction"]
    observed_gap = (attr["trend_yield_t_ha"] - attr["actual_yield_t_ha"]).clip(lower=0)
    recovered = np.minimum(recovered, observed_gap)
    out = pd.DataFrame(
        {
            "crop": attr["crop"],
            "region": attr["region"],
            "year": attr["year"],
            "dominant_feature": attr["dominant_driver"],
            "driver_group": attr["dominant_driver"].map(feature_group),
            "observed_yield_t_ha": attr["actual_yield_t_ha"],
            "trend_yield_t_ha": attr["trend_yield_t_ha"],
            "yield_gap_t_ha": observed_gap,
            "recovered_gap_t_ha": recovered,
            "recoverable_fraction": np.divide(
                recovered,
                observed_gap,
                out=np.zeros(len(recovered), dtype=float),
                where=observed_gap.to_numpy(dtype=float) > 0,
            ),
        }
    )
    out["weather_driven"] = out["recoverable_fraction"] >= WEATHER_DRIVEN_THRESHOLD
    out.to_csv(paths.outputs / "baseline_common_attributions.csv", index=False)
    common_claim_columns(out, "00_baseline_v1_raw_yield_scaa").to_csv(
        paths.outputs / "crop_driver_claims.csv", index=False
    )
    for name in [
        "recoverable_fraction_distribution.png",
        "dominant_driver_frequency.png",
        "anomaly_timeline.png",
    ]:
        src = root / "figures" / name
        if src.exists():
            shutil.copy2(src, paths.figures / name)
    plot_crop_driver_heatmap(out, paths.figures / "crop_driver_recoverability.png", "Baseline recoverability by crop and driver")
    write_method_note(
        paths,
        "Baseline V1 Raw-Yield SCAA",
        [
            "Uses the existing raw-yield model counterfactual output as a comparison point.",
            "This method is useful as a baseline, but should not be the main paper claim if residual or grouped methods are stronger.",
            "Driver labels are mapped from dominant weather features into physical driver groups.",
        ],
    )
    write_attribution_results(
        paths,
        "Baseline V1 Raw-Yield SCAA",
        "Baseline copied from the V1 raw-yield counterfactual attribution.",
        out,
        {"source": "outputs/counterfactual_attributions.csv"},
    )
    return out, attribution_summary("00_baseline_v1_raw_yield_scaa", out, 7, 7, 7)


def run_residual_sparse(
    improve_root: Path,
    scored: pd.DataFrame,
    anomalies: pd.DataFrame,
    model: Pipeline,
    numeric: list[str],
    categorical: list[str],
    features: list[str],
    residual_metrics: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = method_paths(improve_root, "01_residual_target_scaa")
    stats_map = feasible_stats(scored, features)
    records: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    quantile_cols = ["q0", "q10", "q25", "q50", "q75", "q90", "q100"]
    for _, row in anomalies.sort_values(["year", "crop", "region"]).iterrows():
        current = row.copy()
        stats = stats_map[(row["region"], row["window"])]
        base_pred = predict_one(model, current, numeric, categorical)
        needed = max(0.0, -base_pred)
        best_pred = base_pred
        selected: set[str] = set()
        row_details: list[dict[str, Any]] = []
        for step in range(1, 5):
            candidate_rows: list[pd.Series] = []
            metas: list[dict[str, Any]] = []
            for feature in features:
                if feature in selected or feature not in stats.index:
                    continue
                observed = float(current[feature])
                for q in quantile_cols:
                    value = float(stats.loc[feature, q])
                    if abs(value - observed) < 1e-12:
                        continue
                    candidate = current.copy()
                    candidate[feature] = value
                    cost = standardized_delta(stats, feature, observed, value)
                    if cost <= 0:
                        continue
                    candidate_rows.append(candidate[numeric + categorical])
                    metas.append(
                        {
                            "feature": feature,
                            "from_value": observed,
                            "to_value": value,
                            "standardized_delta": cost,
                        }
                    )
            if not candidate_rows:
                break
            preds = predict_rows(model, candidate_rows, numeric, categorical)
            best_i = None
            best_score = 0.0
            for i, pred in enumerate(preds):
                gain = float(pred - best_pred)
                if gain <= 1e-10:
                    continue
                score = gain / max(float(metas[i]["standardized_delta"]), 1e-9)
                if score > best_score:
                    best_score = score
                    best_i = i
            if best_i is None:
                break
            choice = metas[best_i]
            feature = choice["feature"]
            new_pred = float(preds[best_i])
            detail = {
                "crop": row["crop"],
                "region": row["region"],
                "year": int(row["year"]),
                "step": step,
                "feature": feature,
                "from_value": choice["from_value"],
                "to_value": choice["to_value"],
                "standardized_delta": choice["standardized_delta"],
                "prediction_before_t_ha": best_pred,
                "prediction_after_t_ha": new_pred,
                "prediction_gain_t_ha": new_pred - best_pred,
            }
            row_details.append(detail)
            details.append(detail)
            current[feature] = choice["to_value"]
            selected.add(feature)
            best_pred = new_pred
            if needed > 0 and (best_pred - base_pred) / needed >= 0.95:
                break

        observed_gap = max(0.0, float(row["trend_yield_t_ha"] - row[TARGET]))
        gain = max(0.0, best_pred - base_pred)
        recovered = min(gain, needed, observed_gap)
        fraction = 0.0 if observed_gap <= 0 else float(np.clip(recovered / observed_gap, 0.0, 1.0))
        dominant = (
            max(row_details, key=lambda d: abs(float(d["standardized_delta"])))["feature"]
            if row_details
            else "no_feasible_weather_gain"
        )
        records.append(
            {
                "crop": row["crop"],
                "region": row["region"],
                "year": int(row["year"]),
                "window": row["window"],
                "dominant_feature": dominant,
                "driver_group": feature_group(dominant),
                "observed_yield_t_ha": float(row[TARGET]),
                "trend_yield_t_ha": float(row["trend_yield_t_ha"]),
                "yield_gap_t_ha": observed_gap,
                "base_predicted_residual_t_ha": base_pred,
                "counterfactual_predicted_residual_t_ha": best_pred,
                "recovered_gap_t_ha": recovered,
                "recoverable_fraction": fraction,
                "n_changed_features": len(row_details),
                "changed_features_json": compact_changes(row_details),
            }
        )

    attr = pd.DataFrame(records)
    change_df = pd.DataFrame(details)
    attr.to_csv(paths.outputs / "residual_sparse_attributions.csv", index=False)
    change_df.to_csv(paths.outputs / "residual_sparse_feature_changes.csv", index=False)
    common_claim_columns(attr, "01_residual_target_scaa").to_csv(paths.outputs / "crop_driver_claims.csv", index=False)
    plot_recoverable(attr, paths.figures / "recoverable_fraction_distribution.png", "Residual-target sparse SCAA")
    plot_driver_counts(attr, paths.figures / "driver_group_frequency.png", "Residual-target driver groups")
    plot_crop_driver_heatmap(attr, paths.figures / "crop_driver_recoverability.png", "Residual SCAA by crop and driver")
    write_method_note(
        paths,
        "Residual-Target SCAA",
        [
            "Fits a weather-to-detrended-residual model instead of using raw yield for attribution.",
            "Counterfactual target is residual = 0, matching the advisor idea of explaining abnormal yield years.",
            "Sparse feature search changes up to four full-season weather indicators within observed region-window bounds.",
        ],
    )
    write_attribution_results(
        paths,
        "Residual-Target SCAA",
        "This is the closest implementation of detrend -> anomaly -> counterfactual weather attribution.",
        attr,
        residual_metrics,
    )
    return attr, attribution_summary("01_residual_target_scaa", attr, 10, 8, 8)


def apply_group_normal(row: pd.Series, stats: pd.DataFrame, group: list[str]) -> tuple[pd.Series, list[dict[str, Any]]]:
    candidate = row.copy()
    details: list[dict[str, Any]] = []
    for feature in group:
        if feature not in stats.index:
            continue
        old = float(row[feature])
        new = float(stats.loc[feature, "q50"])
        if abs(old - new) < 1e-12:
            continue
        candidate[feature] = new
        details.append(
            {
                "feature": feature,
                "from_value": old,
                "to_value": new,
                "standardized_delta": standardized_delta(stats, feature, old, new),
            }
        )
    return candidate, details


def run_grouped_scaa(
    improve_root: Path,
    scored: pd.DataFrame,
    anomalies: pd.DataFrame,
    model: Pipeline,
    numeric: list[str],
    categorical: list[str],
    features: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = method_paths(improve_root, "02_grouped_driver_scaa")
    groups = group_features(features)
    stats_map = feasible_stats(scored, features)
    records: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for _, row in anomalies.sort_values(["year", "crop", "region"]).iterrows():
        stats = stats_map[(row["region"], row["window"])]
        base_pred = predict_one(model, row, numeric, categorical)
        needed = max(0.0, -base_pred)
        candidates: list[pd.Series] = []
        metas: list[dict[str, Any]] = []
        for group_name, group_cols in groups.items():
            candidate, details = apply_group_normal(row, stats, group_cols)
            if not details:
                continue
            candidates.append(candidate[numeric + categorical])
            metas.append({"driver_group": group_name, "details": details})
        if candidates:
            preds = predict_rows(model, candidates, numeric, categorical)
            best_i = int(np.argmax(preds - base_pred))
            best_pred = float(preds[best_i])
            best_meta = metas[best_i]
            gain = max(0.0, best_pred - base_pred)
        else:
            best_pred = base_pred
            gain = 0.0
            best_meta = {"driver_group": "no_feasible_weather_gain", "details": []}
        observed_gap = max(0.0, float(row["trend_yield_t_ha"] - row[TARGET]))
        recovered = min(gain, needed, observed_gap)
        fraction = 0.0 if observed_gap <= 0 else float(np.clip(recovered / observed_gap, 0.0, 1.0))
        details = best_meta["details"]
        dominant = (
            max(details, key=lambda d: abs(float(d["standardized_delta"])))["feature"]
            if details
            else "no_feasible_weather_gain"
        )
        for d in details:
            detail_rows.append(
                {
                    "crop": row["crop"],
                    "region": row["region"],
                    "year": int(row["year"]),
                    "driver_group": best_meta["driver_group"],
                    **d,
                    "prediction_gain_t_ha": gain,
                }
            )
        records.append(
            {
                "crop": row["crop"],
                "region": row["region"],
                "year": int(row["year"]),
                "window": row["window"],
                "driver_group": best_meta["driver_group"],
                "dominant_feature": dominant,
                "observed_yield_t_ha": float(row[TARGET]),
                "trend_yield_t_ha": float(row["trend_yield_t_ha"]),
                "yield_gap_t_ha": observed_gap,
                "base_predicted_residual_t_ha": base_pred,
                "counterfactual_predicted_residual_t_ha": best_pred,
                "recovered_gap_t_ha": recovered,
                "recoverable_fraction": fraction,
                "n_changed_features": len(details),
                "changed_features_json": compact_changes(
                    [{**d, "prediction_gain_t_ha": gain} for d in details]
                ),
            }
        )
    attr = pd.DataFrame(records)
    details = pd.DataFrame(detail_rows)
    attr.to_csv(paths.outputs / "grouped_driver_attributions.csv", index=False)
    details.to_csv(paths.outputs / "grouped_driver_feature_changes.csv", index=False)
    common_claim_columns(attr, "02_grouped_driver_scaa").to_csv(paths.outputs / "crop_driver_claims.csv", index=False)
    plot_recoverable(attr, paths.figures / "recoverable_fraction_distribution.png", "Grouped-driver SCAA")
    plot_driver_counts(attr, paths.figures / "driver_group_frequency.png", "Grouped-driver frequency")
    plot_crop_driver_heatmap(attr, paths.figures / "crop_driver_recoverability.png", "Grouped SCAA by crop and driver")
    write_method_note(
        paths,
        "Grouped-Driver SCAA",
        [
            "Changes physical driver groups toward their observed region-window median instead of changing isolated features.",
            "Driver groups are heat, drought, frost/cold, excess rain, and radiation.",
            "This layer is meant to produce clearer paper claims such as drought stress harmed Wheat in a given state-year.",
        ],
    )
    write_attribution_results(
        paths,
        "Grouped-Driver SCAA",
        "This method trades feature-level sparsity for physically readable weather-driver groups.",
        attr,
        {"driver_groups": ", ".join(groups)},
    )
    return attr, attribution_summary("02_grouped_driver_scaa", attr, 9, 10, 9)


def run_grouped_scaa_temporal_holdout(
    improve_root: Path,
    scored: pd.DataFrame,
    anomalies: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    features: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = method_paths(improve_root, "06_grouped_driver_scaa_temporal_holdout")
    groups = group_features(features)
    stats_map = feasible_stats(scored, features)
    records: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    validation_rows: list[pd.DataFrame] = []

    forward_train = scored[scored["year"] <= 2015].copy()
    forward_test = scored[scored["year"] >= 2016].copy()
    forward_model = make_regressor(numeric, categorical)
    forward_model.fit(forward_train[numeric + categorical], forward_train["trend_residual_t_ha"])
    forward_pred = forward_model.predict(forward_test[numeric + categorical])
    forward_summary = regression_summary(forward_test["trend_residual_t_ha"], forward_pred)

    for heldout_year, event_rows in anomalies.sort_values(["year", "crop", "region"]).groupby("year", sort=True):
        train = scored[scored["year"] != heldout_year].copy()
        holdout_model = make_regressor(numeric, categorical)
        holdout_model.fit(train[numeric + categorical], train["trend_residual_t_ha"])
        heldout_rows = scored[scored["year"] == heldout_year].copy()
        heldout_rows["predicted_residual_t_ha"] = holdout_model.predict(heldout_rows[numeric + categorical])
        heldout_rows["heldout_year"] = int(heldout_year)
        heldout_rows["train_rows"] = int(len(train))
        validation_rows.append(heldout_rows)

        for _, row in event_rows.iterrows():
            stats = stats_map[(row["region"], row["window"])]
            base_pred = predict_one(holdout_model, row, numeric, categorical)
            needed = max(0.0, -base_pred)
            candidates: list[pd.Series] = []
            metas: list[dict[str, Any]] = []
            for group_name, group_cols in groups.items():
                candidate, details = apply_group_normal(row, stats, group_cols)
                if not details:
                    continue
                candidates.append(candidate[numeric + categorical])
                metas.append({"driver_group": group_name, "details": details})
            if candidates:
                preds = predict_rows(holdout_model, candidates, numeric, categorical)
                best_i = int(np.argmax(preds - base_pred))
                best_pred = float(preds[best_i])
                best_meta = metas[best_i]
                gain = max(0.0, best_pred - base_pred)
            else:
                best_pred = base_pred
                gain = 0.0
                best_meta = {"driver_group": "no_feasible_weather_gain", "details": []}

            observed_gap = max(0.0, float(row["trend_yield_t_ha"] - row[TARGET]))
            recovered = min(gain, needed, observed_gap)
            fraction = 0.0 if observed_gap <= 0 else float(np.clip(recovered / observed_gap, 0.0, 1.0))
            details = best_meta["details"]
            dominant = (
                max(details, key=lambda d: abs(float(d["standardized_delta"])))["feature"]
                if details
                else "no_feasible_weather_gain"
            )
            for d in details:
                detail_rows.append(
                    {
                        "crop": row["crop"],
                        "region": row["region"],
                        "year": int(row["year"]),
                        "heldout_year": int(heldout_year),
                        "driver_group": best_meta["driver_group"],
                        **d,
                        "prediction_gain_t_ha": gain,
                    }
                )
            records.append(
                {
                    "crop": row["crop"],
                    "region": row["region"],
                    "year": int(row["year"]),
                    "heldout_year": int(heldout_year),
                    "train_rows": int(len(train)),
                    "window": row["window"],
                    "trend_residual_z": float(row["trend_residual_z"]),
                    "driver_group": best_meta["driver_group"],
                    "dominant_feature": dominant,
                    "observed_yield_t_ha": float(row[TARGET]),
                    "trend_yield_t_ha": float(row["trend_yield_t_ha"]),
                    "yield_gap_t_ha": observed_gap,
                    "base_predicted_residual_t_ha": base_pred,
                    "counterfactual_predicted_residual_t_ha": best_pred,
                    "recovered_gap_t_ha": recovered,
                    "recoverable_fraction": fraction,
                    "n_changed_features": len(details),
                    "changed_features_json": compact_changes(
                        [{**d, "prediction_gain_t_ha": gain} for d in details]
                    ),
                }
            )

    attr = pd.DataFrame(records)
    details = pd.DataFrame(detail_rows)
    validation_predictions = pd.concat(validation_rows, ignore_index=True)
    all_holdout = regression_summary(
        validation_predictions["trend_residual_t_ha"],
        validation_predictions["predicted_residual_t_ha"],
    )
    anomaly_holdout = validation_predictions[validation_predictions["is_low_yield_anomaly"]].copy()
    anomaly_summary = regression_summary(
        anomaly_holdout["trend_residual_t_ha"],
        anomaly_holdout["predicted_residual_t_ha"],
    )
    residual_validation = pd.DataFrame(
        [
            {
                "model": "Residual ExtraTrees",
                "target": "trend_residual_t_ha",
                "protocol": "forward_time_train_1990_2015_test_2016_2025",
                "n_train": int(len(forward_train)),
                "n_test": int(len(forward_test)),
                **forward_summary,
                "interpretation": "Residual-model robustness under a forward-time split.",
            },
            {
                "model": "Residual ExtraTrees",
                "target": "trend_residual_t_ha",
                "protocol": "retrospective_leave_one_anomaly_year_out_all_rows",
                "n_train": f"{int(validation_predictions['train_rows'].min())}-{int(validation_predictions['train_rows'].max())}",
                "n_test": int(len(validation_predictions)),
                **all_holdout,
                "interpretation": "Same year-exclusion protocol as grouped SCAA, evaluated on all rows in held-out anomaly years.",
            },
            {
                "model": "Residual ExtraTrees",
                "target": "trend_residual_t_ha",
                "protocol": "retrospective_leave_one_anomaly_year_out_anomaly_rows",
                "n_train": f"{int(validation_predictions['train_rows'].min())}-{int(validation_predictions['train_rows'].max())}",
                "n_test": int(len(anomaly_holdout)),
                **anomaly_summary,
                "interpretation": "Same protocol evaluated only on low-yield anomaly rows that are explained by SCAA.",
            },
        ]
    )
    attr.to_csv(paths.outputs / "temporal_holdout_attributions.csv", index=False)
    details.to_csv(paths.outputs / "temporal_holdout_feature_changes.csv", index=False)
    residual_validation.to_csv(paths.outputs / "residual_model_validation.csv", index=False)
    validation_predictions[
        [
            "crop",
            "region",
            "year",
            "heldout_year",
            "train_rows",
            "trend_residual_t_ha",
            "predicted_residual_t_ha",
            "is_low_yield_anomaly",
        ]
    ].to_csv(paths.outputs / "residual_model_validation_predictions.csv", index=False)
    common_claim_columns(attr, "06_grouped_driver_scaa_temporal_holdout").to_csv(
        paths.outputs / "crop_driver_claims.csv", index=False
    )
    plot_recoverable(
        attr,
        paths.figures / "recoverable_fraction_distribution.png",
        "Temporal-holdout grouped SCAA",
    )
    plot_driver_counts(attr, paths.figures / "driver_group_frequency.png", "Temporal-holdout driver groups")
    plot_crop_driver_heatmap(
        attr,
        paths.figures / "crop_driver_recoverability.png",
        "Temporal-holdout SCAA by crop and driver",
    )
    write_method_note(
        paths,
        "Temporal-Holdout Grouped-Driver SCAA",
        [
            "For each anomaly year, the residual model is trained after excluding every row from that year.",
            "The residual target is raw detrended yield residual in t/ha; standardized residuals only screen anomalies.",
            "This is the main paper protocol because it avoids explaining event rows with a model fitted on the same year.",
        ],
    )
    write_attribution_results(
        paths,
        "Temporal-Holdout Grouped-Driver SCAA",
        "Main paper method: grouped SCAA with event-year holdout residual models.",
        attr,
        {
            "driver_groups": ", ".join(groups),
            "unique_heldout_years": int(attr["heldout_year"].nunique()),
            "residual_forward_r2": float(forward_summary["r2"]),
            "residual_leave_one_year_out_r2": float(all_holdout["r2"]),
        },
    )
    return attr, attribution_summary("06_grouped_driver_scaa_temporal_holdout", attr, 10, 10, 9)


def run_observed_analog(
    improve_root: Path,
    scored: pd.DataFrame,
    anomalies: pd.DataFrame,
    model: Pipeline,
    numeric: list[str],
    categorical: list[str],
    features: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = method_paths(improve_root, "03_observed_analog_counterfactual")
    stats_map = feasible_stats(scored, features)
    normal = scored[~scored["is_low_yield_anomaly"]].copy()
    records: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for _, row in anomalies.sort_values(["year", "crop", "region"]).iterrows():
        stats = stats_map[(row["region"], row["window"])]
        pool = normal[(normal["crop"] == row["crop"]) & (normal["region"] == row["region"])]
        fallback_used = False
        if pool.empty:
            pool = normal[(normal["region"] == row["region"]) & (normal["window"] == row["window"])]
            fallback_used = True
        base_pred = predict_one(model, row, numeric, categorical)
        needed = max(0.0, -base_pred)
        ranked: list[tuple[float, pd.Series]] = []
        for _, candidate in pool.iterrows():
            dist_terms = []
            for feature in features:
                if feature not in stats.index:
                    continue
                delta = standardized_delta(stats, feature, float(row[feature]), float(candidate[feature]))
                dist_terms.append(delta**2)
            distance = math.sqrt(float(np.mean(dist_terms))) if dist_terms else 0.0
            ranked.append((distance, candidate))
        ranked = sorted(ranked, key=lambda x: x[0])[:10]
        best_candidate = ranked[0][1] if ranked else row
        best_distance = ranked[0][0] if ranked else 0.0
        best_pred = base_pred
        best_gain = 0.0
        best_rank = 1
        for rank, (distance, analog) in enumerate(ranked, start=1):
            candidate_row = row.copy()
            for feature in features:
                candidate_row[feature] = analog[feature]
            pred = predict_one(model, candidate_row, numeric, categorical)
            gain = pred - base_pred
            score = gain / max(distance, 1e-6)
            if gain > best_gain and score > 0:
                best_candidate = analog
                best_distance = distance
                best_pred = pred
                best_gain = gain
                best_rank = rank
        group_delta: dict[str, float] = {}
        feature_delta: dict[str, float] = {}
        for feature in features:
            delta = standardized_delta(stats, feature, float(row[feature]), float(best_candidate[feature]))
            group_delta[feature_group(feature)] = group_delta.get(feature_group(feature), 0.0) + delta
            feature_delta[feature] = delta
            if delta > 0:
                details.append(
                    {
                        "crop": row["crop"],
                        "region": row["region"],
                        "year": int(row["year"]),
                        "analog_year": int(best_candidate["year"]),
                        "feature": feature,
                        "driver_group": feature_group(feature),
                        "from_value": float(row[feature]),
                        "to_value": float(best_candidate[feature]),
                        "standardized_delta": delta,
                        "prediction_gain_t_ha": best_gain,
                    }
                )
        driver_group = max(group_delta, key=group_delta.get) if group_delta else "no_analog"
        group_feature_delta = {
            feature: delta for feature, delta in feature_delta.items() if feature_group(feature) == driver_group
        }
        dominant_pool = group_feature_delta if group_feature_delta else feature_delta
        dominant = max(dominant_pool, key=dominant_pool.get) if dominant_pool else "no_analog"
        observed_gap = max(0.0, float(row["trend_yield_t_ha"] - row[TARGET]))
        recovered = min(max(0.0, best_pred - base_pred), needed, observed_gap)
        fraction = 0.0 if observed_gap <= 0 else float(np.clip(recovered / observed_gap, 0.0, 1.0))
        records.append(
            {
                "crop": row["crop"],
                "region": row["region"],
                "year": int(row["year"]),
                "window": row["window"],
                "driver_group": driver_group,
                "dominant_feature": dominant,
                "observed_yield_t_ha": float(row[TARGET]),
                "trend_yield_t_ha": float(row["trend_yield_t_ha"]),
                "yield_gap_t_ha": observed_gap,
                "base_predicted_residual_t_ha": base_pred,
                "counterfactual_predicted_residual_t_ha": best_pred,
                "recovered_gap_t_ha": recovered,
                "recoverable_fraction": fraction,
                "analog_year": int(best_candidate["year"]),
                "analog_distance": best_distance,
                "analog_rank_used": best_rank,
                "fallback_region_window_pool": fallback_used,
            }
        )
    attr = pd.DataFrame(records)
    detail_df = pd.DataFrame(details)
    attr.to_csv(paths.outputs / "observed_analog_attributions.csv", index=False)
    detail_df.to_csv(paths.outputs / "observed_analog_feature_changes.csv", index=False)
    common_claim_columns(attr, "03_observed_analog_counterfactual").to_csv(
        paths.outputs / "crop_driver_claims.csv", index=False
    )
    plot_recoverable(attr, paths.figures / "recoverable_fraction_distribution.png", "Observed-analog counterfactuals")
    plot_driver_counts(attr, paths.figures / "driver_group_frequency.png", "Observed-analog driver groups")
    plot_crop_driver_heatmap(attr, paths.figures / "crop_driver_recoverability.png", "Analog recoverability by crop and driver")
    write_method_note(
        paths,
        "Observed-Analog Counterfactual",
        [
            "Uses real non-anomalous seasons as counterfactual weather analogs.",
            "Primary pool is same crop and region; fallback is same region and growing-season window.",
            "This is the robustness check: the counterfactual weather state actually occurred historically.",
        ],
    )
    write_attribution_results(
        paths,
        "Observed-Analog Counterfactual",
        "This method uses observed normal seasons instead of synthetic quantile edits.",
        attr,
        {"fallback_rows": int(attr["fallback_region_window_pool"].sum())},
    )
    return attr, attribution_summary("03_observed_analog_counterfactual", attr, 8, 9, 10)


def adverse_value(feature: str, group: str, stats: pd.DataFrame, high: bool | None = None) -> float:
    if group == "drought":
        if feature in {"rain_sum", "rain_mean"} or feature == "wet_days_1mm":
            return float(stats.loc[feature, "q10"])
        return float(stats.loc[feature, "q90"])
    if group == "heat":
        return float(stats.loc[feature, "q90"])
    if group == "frost_cold":
        if feature in {"min_tmin", "season_tmin_mean"}:
            return float(stats.loc[feature, "q10"])
        return float(stats.loc[feature, "q90"])
    if group == "excess_rain":
        return float(stats.loc[feature, "q90"])
    if group == "radiation":
        q = "q90" if high else "q10"
        return float(stats.loc[feature, q])
    return float(stats.loc[feature, "q90"])


def run_vulnerability_profiles(
    improve_root: Path,
    scored: pd.DataFrame,
    model: Pipeline,
    numeric: list[str],
    categorical: list[str],
    features: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = method_paths(improve_root, "04_crop_specific_vulnerability_profiles")
    groups = group_features(features)
    stats_map = feasible_stats(scored, features)
    rows: list[dict[str, Any]] = []
    for _, row in scored.iterrows():
        stats = stats_map[(row["region"], row["window"])]
        base_pred = predict_one(model, row, numeric, categorical)
        for group_name, group_cols in groups.items():
            candidates: list[tuple[str, pd.Series]] = []
            direction_names = ["adverse"]
            if group_name == "radiation":
                direction_names = ["low_radiation", "high_radiation"]
            for direction in direction_names:
                candidate = row.copy()
                for feature in group_cols:
                    if feature in stats.index:
                        candidate[feature] = adverse_value(
                            feature,
                            group_name,
                            stats,
                            high=(direction == "high_radiation"),
                        )
                candidates.append((direction, candidate[numeric + categorical]))
            preds = predict_rows(model, [c[1] for c in candidates], numeric, categorical)
            worst_i = int(np.argmin(preds))
            effect = float(preds[worst_i] - base_pred)
            rows.append(
                {
                    "crop": row["crop"],
                    "region": row["region"],
                    "year": int(row["year"]),
                    "driver_group": group_name,
                    "direction": candidates[worst_i][0],
                    "base_predicted_residual_t_ha": base_pred,
                    "adverse_predicted_residual_t_ha": float(preds[worst_i]),
                    "effect_t_ha": effect,
                }
            )
    row_df = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for (crop, group), g in row_df.groupby(["crop", "driver_group"]):
        state_effects = g.groupby("region")["effect_t_ha"].median().sort_values()
        direction = g["direction"].mode().iloc[0]
        summaries.append(
            {
                "crop": crop,
                "driver_group": group,
                "median_effect_t_ha": float(g["effect_t_ha"].median()),
                "p25_effect_t_ha": float(g["effect_t_ha"].quantile(0.25)),
                "p75_effect_t_ha": float(g["effect_t_ha"].quantile(0.75)),
                "effect_direction": direction,
                "states_most_sensitive": "; ".join(
                    f"{state} ({value:.3f})" for state, value in state_effects.head(3).items()
                ),
                "claim_sentence": (
                    f"For {crop}, {group_label(group)} produced a median modelled residual change of "
                    f"{float(g['effect_t_ha'].median()):.3f} t/ha under adverse observed extremes; "
                    f"the most sensitive states were {', '.join(state_effects.head(3).index.astype(str))}."
                ),
            }
        )
    summary = pd.DataFrame(summaries)
    row_df.to_csv(paths.outputs / "crop_driver_vulnerability_rows.csv", index=False)
    summary.to_csv(paths.outputs / "crop_driver_vulnerability.csv", index=False)
    summary.to_csv(paths.outputs / "crop_vulnerability_claims.csv", index=False)
    plot_vulnerability(summary, paths.figures / "crop_driver_vulnerability_heatmap.png")
    write_method_note(
        paths,
        "Crop-Specific Vulnerability Profiles",
        [
            "Stress-tests each crop under adverse observed driver-group extremes.",
            "The output says which weather driver lowers which crop most strongly, instead of only saying yield decreases.",
            "This is a contribution table rather than an event-level anomaly attribution method.",
        ],
    )
    lines = [
        "# Crop-Specific Vulnerability Profiles",
        "",
        f"- Crop-driver rows: {len(summary)}",
        f"- Most negative median effect: {summary['median_effect_t_ha'].min():.3f} t/ha",
        "",
        "## Strongest Vulnerabilities",
        "",
    ]
    strongest = summary.sort_values("median_effect_t_ha").head(10)
    for _, r in strongest.iterrows():
        lines.append(
            f"- {r['crop']} / {r['driver_group']}: median {r['median_effect_t_ha']:.3f} t/ha; "
            f"states {r['states_most_sensitive']}"
        )
    write_lines(paths.results, lines)
    score = {
        "method": "04_crop_specific_vulnerability_profiles",
        "idea_alignment_score": 7,
        "crop_specificity_score": 10,
        "event_validation_score": 0,
        "recoverability_score": max(0, min(10, abs(float(summary["median_effect_t_ha"].min())) * 5)),
        "physical_plausibility_score": 9,
        "no_leakage_check": True,
        "median_recoverable_fraction": np.nan,
        "weather_driven_rate": np.nan,
        "event_expected_match_rate": np.nan,
        "total_score": np.nan,
    }
    score["total_score"] = weighted_score(score)
    return summary, score


def plot_vulnerability(summary: pd.DataFrame, path: Path) -> None:
    pivot = summary.pivot(index="crop", columns="driver_group", values="median_effect_t_ha")
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    vmax = max(abs(float(np.nanmin(pivot.to_numpy()))), abs(float(np.nanmax(pivot.to_numpy()))), 0.1)
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Crop-specific modelled residual change under adverse weather")
    fig.colorbar(im, ax=ax, label="Median effect (t/ha)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_warning_improved(improve_root: Path, scored: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = method_paths(improve_root, "05_early_mid_warning_improved")
    df = scored.copy()
    df["anomaly_label"] = df["is_low_yield_anomaly"].astype(int)
    stages = {
        "early_third": stage_features(df, "_early"),
        "early_mid_two_thirds": stage_features(df, "_early") + stage_features(df, "_mid"),
    }
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    crop_metrics: list[dict[str, Any]] = []
    for stage, features in stages.items():
        train = df[df["year"] <= 2015].copy()
        calibration = df[(df["year"] >= 2016) & (df["year"] <= 2018)].copy()
        test = df[df["year"] >= 2019].copy()
        numeric = ["year", "lat", "lon"] + features
        categorical = CAT_FEATURES
        model = make_classifier(numeric, categorical)
        columns = numeric + categorical
        model.fit(train[columns], train["anomaly_label"])
        calib_prob = model.predict_proba(calibration[columns])[:, 1]
        test_prob = model.predict_proba(test[columns])[:, 1]
        threshold = best_threshold(calibration["anomaly_label"].to_numpy(dtype=int), calib_prob)
        test_pred = (test_prob >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test["anomaly_label"], test_pred, average="binary", zero_division=0
        )
        q = conformal_quantile(np.abs(calibration["anomaly_label"].to_numpy(dtype=float) - calib_prob), alpha=0.1)
        metrics.append(
            {
                "stage": stage,
                "n_features": len(features),
                "n_train": len(train),
                "n_calibration": len(calibration),
                "n_test": len(test),
                "threshold_from_calibration": threshold,
                "roc_auc": safe_auc(test["anomaly_label"], test_prob),
                "average_precision": float(average_precision_score(test["anomaly_label"], test_prob)),
                "brier_score": float(brier_score_loss(test["anomaly_label"], test_prob)),
                "accuracy": float(accuracy_score(test["anomaly_label"], test_pred)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "top_10_precision": top_risk_precision(test["anomaly_label"], test_prob, 0.10),
                "top_20_precision": top_risk_precision(test["anomaly_label"], test_prob, 0.20),
                "conformal_abs_error_q90": q,
            }
        )
        pred = test[["country", "region", "crop", "year", "window", TARGET, "trend_yield_t_ha", "trend_residual_z", "anomaly_label"]].copy()
        pred["stage"] = stage
        pred["anomaly_probability"] = test_prob
        pred["predicted_anomaly"] = test_pred
        pred["threshold_from_calibration"] = threshold
        pred["probability_lower_conformal_style"] = np.clip(test_prob - q, 0.0, 1.0)
        pred["probability_upper_conformal_style"] = np.clip(test_prob + q, 0.0, 1.0)
        predictions.append(pred)
        for crop, g in pred.groupby("crop"):
            crop_metrics.append(
                {
                    "stage": stage,
                    "crop": crop,
                    "n_test": len(g),
                    "anomaly_rate": float(g["anomaly_label"].mean()),
                    "top_10_precision": top_risk_precision(g["anomaly_label"], g["anomaly_probability"], 0.10),
                    "top_20_precision": top_risk_precision(g["anomaly_label"], g["anomaly_probability"], 0.20),
                    "average_probability": float(g["anomaly_probability"].mean()),
                }
            )
    metrics_df = pd.DataFrame(metrics)
    predictions_df = pd.concat(predictions, ignore_index=True)
    crop_df = pd.DataFrame(crop_metrics)
    metrics_df.to_csv(paths.outputs / "warning_metrics.csv", index=False)
    predictions_df.to_csv(paths.outputs / "warning_predictions.csv", index=False)
    crop_df.to_csv(paths.outputs / "warning_crop_metrics.csv", index=False)
    plot_warning_improved(metrics_df, paths.figures / "warning_stage_comparison.png")
    write_method_note(
        paths,
        "Early/Mid-Season Warning Improved",
        [
            "Optimizes the binary warning threshold on the calibration period instead of forcing 0.5.",
            "Reports top-risk precision because warning systems usually act on the highest-risk cases.",
            "This is an application extension, not the main attribution contribution.",
        ],
    )
    lines = [
        "# Early/Mid-Season Warning Improved",
        "",
    ]
    for _, r in metrics_df.iterrows():
        lines.append(
            f"- {r['stage']}: AP={r['average_precision']:.3f}, AUC={r['roc_auc']:.3f}, "
            f"F1={r['f1']:.3f}, top10 precision={r['top_10_precision']:.3f}, "
            f"top20 precision={r['top_20_precision']:.3f}"
        )
    write_lines(paths.results, lines)
    best = metrics_df.sort_values("average_precision", ascending=False).iloc[0]
    score = {
        "method": "05_early_mid_warning_improved",
        "idea_alignment_score": 5,
        "crop_specificity_score": 6,
        "event_validation_score": 0,
        "recoverability_score": max(0, min(10, float(best["average_precision"]) * 20)),
        "physical_plausibility_score": 8,
        "no_leakage_check": True,
        "median_recoverable_fraction": np.nan,
        "weather_driven_rate": np.nan,
        "event_expected_match_rate": np.nan,
        "total_score": np.nan,
    }
    score["total_score"] = weighted_score(score)
    return metrics_df, score


def best_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        pred = (prob >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        if float(f1) > best_f1:
            best_f1 = float(f1)
            best_t = float(threshold)
    return best_t


def top_risk_precision(y_true: pd.Series | np.ndarray, prob: pd.Series | np.ndarray, frac: float) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    if len(y) == 0:
        return float("nan")
    k = max(1, int(math.ceil(len(y) * frac)))
    idx = np.argsort(-p)[:k]
    return float(y[idx].mean())


def plot_warning_improved(metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0))
    x = np.arange(len(metrics))
    labels = metrics["stage"].tolist()
    for ax, col, title, color in [
        (axes[0], "average_precision", "Average precision", "#4d7ea8"),
        (axes[1], "top_20_precision", "Top 20% precision", "#5f8f6b"),
        (axes[2], "brier_score", "Brier score", "#c47f47"),
    ]:
        ax.bar(x, metrics[col], color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def event_validation_rows(method: str, attr: pd.DataFrame) -> pd.DataFrame:
    rows = attr[attr["year"].isin(EVENT_YEARS)].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["method"] = method
    rows["expected_groups"] = rows["year"].map(lambda y: ",".join(sorted(EXPECTED_EVENT_GROUPS[int(y)])))
    rows["expected_match"] = rows.apply(
        lambda r: r["driver_group"] in EXPECTED_EVENT_GROUPS.get(int(r["year"]), set()), axis=1
    )
    return rows[
        [
            "method",
            "crop",
            "region",
            "year",
            "driver_group",
            "dominant_feature",
            "recoverable_fraction",
            "recovered_gap_t_ha",
            "yield_gap_t_ha",
            "expected_groups",
            "expected_match",
        ]
    ]


def build_event_consistency_null_baselines(event_validation: pd.DataFrame) -> pd.DataFrame:
    temporal = event_validation[event_validation["method"] == "06_grouped_driver_scaa_temporal_holdout"].copy()
    if temporal.empty:
        return pd.DataFrame()

    def expected_match_for_driver(driver: str) -> float:
        return float(
            temporal["year"].map(lambda y: driver in EXPECTED_EVENT_GROUPS.get(int(y), set())).mean()
        )

    driver_frequency = temporal["driver_group"].value_counts(normalize=True)
    most_frequent = str(driver_frequency.index[0])
    random_expected = []
    for year in temporal["year"]:
        expected = EXPECTED_EVENT_GROUPS.get(int(year), set())
        random_expected.append(float(sum(driver_frequency.get(group, 0.0) for group in expected)))

    rows = [
        {
            "method": "Always drought",
            "expected_match_rate": expected_match_for_driver("drought"),
            "median_recoverable_fraction": "",
            "n_event_rows": int(len(temporal)),
            "interpretation": "Trivial baseline; high values mean broad drought labels are easy to match.",
        },
        {
            "method": "Always heat",
            "expected_match_rate": expected_match_for_driver("heat"),
            "median_recoverable_fraction": "",
            "n_event_rows": int(len(temporal)),
            "interpretation": "Trivial baseline; high values mean broad heat labels are easy to match.",
        },
        {
            "method": f"Most frequent event-year SCAA driver ({most_frequent})",
            "expected_match_rate": expected_match_for_driver(most_frequent),
            "median_recoverable_fraction": "",
            "n_event_rows": int(len(temporal)),
            "interpretation": "Majority-driver baseline using the temporal-holdout SCAA event-year outputs.",
        },
        {
            "method": "Driver-frequency random",
            "expected_match_rate": float(np.mean(random_expected)),
            "median_recoverable_fraction": "",
            "n_event_rows": int(len(temporal)),
            "interpretation": "Expected match if drivers are sampled from the temporal-holdout event-year frequency distribution.",
        },
        {
            "method": "Retrospective leave-one-event-year-out grouped SCAA",
            "expected_match_rate": float(temporal["expected_match"].mean()),
            "median_recoverable_fraction": float(temporal["recoverable_fraction"].median()),
            "n_event_rows": int(len(temporal)),
            "interpretation": "Main diagnostic attribution method; evaluated against the same broad expected stress groups.",
        },
    ]
    return pd.DataFrame(rows)


def attribution_summary(
    method: str,
    attr: pd.DataFrame,
    idea_score: float,
    crop_score: float,
    plausibility_score: float,
) -> dict[str, Any]:
    event_rows = event_validation_rows(method, attr)
    event_match = float(event_rows["expected_match"].mean()) if len(event_rows) else 0.0
    median_phi = float(attr["recoverable_fraction"].median()) if len(attr) else 0.0
    wd_rate = float((attr["recoverable_fraction"] >= WEATHER_DRIVEN_THRESHOLD).mean()) if len(attr) else 0.0
    recoverability_score = max(0.0, min(10.0, 10.0 * (0.5 * median_phi + 0.5 * wd_rate)))
    score = {
        "method": method,
        "idea_alignment_score": idea_score,
        "crop_specificity_score": crop_score,
        "event_validation_score": 10.0 * event_match,
        "recoverability_score": recoverability_score,
        "physical_plausibility_score": plausibility_score,
        "no_leakage_check": True,
        "median_recoverable_fraction": median_phi,
        "weather_driven_rate": wd_rate,
        "event_expected_match_rate": event_match,
        "total_score": np.nan,
    }
    score["total_score"] = weighted_score(score)
    return score


def weighted_score(score: dict[str, Any]) -> float:
    if not bool(score["no_leakage_check"]):
        return 0.0
    return float(
        0.25 * score["idea_alignment_score"]
        + 0.25 * score["crop_specificity_score"]
        + 0.20 * score["event_validation_score"]
        + 0.20 * score["recoverability_score"]
        + 0.10 * score["physical_plausibility_score"]
    )


def assert_no_leakage_in_outputs(improve_root: Path) -> None:
    forbidden = {"trend_yield_t_ha", "trend_residual_t_ha", "trend_residual_z", "is_low_yield_anomaly"}
    for path in improve_root.glob("*/outputs/*.csv"):
        df = pd.read_csv(path, nrows=20)
        for col in ["dominant_feature", "dominant_driver"]:
            if col in df.columns:
                bad = set(df[col].dropna().astype(str)) & forbidden
                if bad:
                    raise AssertionError(f"Leakage-like dominant feature in {path}: {bad}")


def assert_figures_nonblank(improve_root: Path) -> None:
    import matplotlib.image as mpimg

    for method in METHODS:
        figures = list((improve_root / method / "figures").glob("*.png"))
        if not figures:
            raise AssertionError(f"No figures generated for {method}")
        for figure in figures:
            img = mpimg.imread(figure)
            if img.size == 0 or float(img.std()) == 0.0:
                raise AssertionError(f"Blank figure: {figure}")


def write_global_reports(
    improve_root: Path,
    scorecard: pd.DataFrame,
    all_claims: pd.DataFrame,
    event_validation: pd.DataFrame,
    vulnerability: pd.DataFrame,
) -> None:
    null_baselines = build_event_consistency_null_baselines(event_validation)
    scorecard.to_csv(improve_root / "method_scorecard.csv", index=False)
    all_claims.to_csv(improve_root / "crop_driver_claims.csv", index=False)
    event_validation.to_csv(improve_root / "event_validation_2012_2021_2022.csv", index=False)
    null_baselines.to_csv(improve_root / "event_consistency_null_baselines.csv", index=False)
    vulnerability.to_csv(improve_root / "crop_specific_vulnerability_profiles.csv", index=False)
    write_comparison_report(improve_root, scorecard, all_claims, event_validation, null_baselines, vulnerability)
    write_decision_report(improve_root, scorecard, all_claims, vulnerability)


def write_comparison_report(
    improve_root: Path,
    scorecard: pd.DataFrame,
    all_claims: pd.DataFrame,
    event_validation: pd.DataFrame,
    null_baselines: pd.DataFrame,
    vulnerability: pd.DataFrame,
) -> None:
    lines = [
        "# Improvement Experiment Comparison",
        "",
        "This report compares alternative ways to turn yield anomalies into crop-specific weather-driver claims.",
        "",
        "## Method Scorecard",
        "",
        "Note: `total_score` is kept for internal method triage only. The paper reports raw metrics instead.",
        "",
        "`06_grouped_driver_scaa_temporal_holdout` is the main submission method. `02_grouped_driver_scaa` is kept as an in-sample exploratory comparison, and `03_observed_analog_counterfactual` is a plausibility robustness check.",
        "",
        "| Method | Total | Idea | Crop-specific | Event | Recoverability | Plausibility | Median phi | High-recovery |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in scorecard.sort_values("total_score", ascending=False).iterrows():
        lines.append(
            f"| {r['method']} | {r['total_score']:.2f} | {r['idea_alignment_score']:.1f} | "
            f"{r['crop_specificity_score']:.1f} | {r['event_validation_score']:.1f} | "
            f"{r['recoverability_score']:.1f} | {r['physical_plausibility_score']:.1f} | "
            f"{fmt(r['median_recoverable_fraction'])} | {fmt(r['weather_driven_rate'])} |"
        )
    temporal_claims = all_claims[all_claims["method"] == "06_grouped_driver_scaa_temporal_holdout"]
    grouped_claims = all_claims[all_claims["method"] == "02_grouped_driver_scaa"]
    claim_source = temporal_claims if len(temporal_claims) else grouped_claims if len(grouped_claims) else all_claims
    lines.extend(["", "## Top Paper-Friendly Temporal-Holdout SCAA Claims", ""])
    for _, r in claim_source.sort_values(["recoverable_fraction", "recovered_gap_t_ha"], ascending=False).head(12).iterrows():
        lines.append(f"- {r['claim_sentence']}")
    if len(event_validation):
        lines.extend(["", "## Event-Year Consistency Check 2012/2021/2022", ""])
        match = event_validation.groupby("method")["expected_match"].mean().sort_values(ascending=False)
        for method, rate in match.items():
            lines.append(f"- {method}: {rate:.1%} of event-year attributions match expected heat/drought/moisture groups.")
    if len(null_baselines):
        lines.extend(["", "## Event-Year Null Baselines", ""])
        for _, r in null_baselines.iterrows():
            lines.append(f"- {r['method']}: {float(r['expected_match_rate']):.1%}.")
    lines.extend(["", "## Crop Vulnerability Highlights", ""])
    for _, r in vulnerability.sort_values("median_effect_t_ha").head(8).iterrows():
        lines.append(
            f"- {r['crop']} is most sensitive to {r['driver_group']}: "
            f"median effect {r['median_effect_t_ha']:.3f} t/ha; {r['states_most_sensitive']}."
        )
    write_lines(improve_root / "COMPARISON_REPORT.md", lines)


def write_decision_report(improve_root: Path, scorecard: pd.DataFrame, all_claims: pd.DataFrame, vulnerability: pd.DataFrame) -> None:
    main = scorecard[scorecard["method"] == "06_grouped_driver_scaa_temporal_holdout"].iloc[0]
    grouped = scorecard[scorecard["method"] == "02_grouped_driver_scaa"].iloc[0]
    analog = scorecard[scorecard["method"] == "03_observed_analog_counterfactual"].iloc[0]
    lines = [
        "# Paper Contribution Decision",
        "",
        f"Recommended main attribution method: `{main['method']}` because it excludes each event year from the residual-model fit before attribution.",
        "",
        "`03_observed_analog_counterfactual` can recover more because it replaces the full weather vector with a real normal season. Use it as robustness evidence, not as the main sparse attribution method.",
        "",
        "## Recommended Paper Framing",
        "",
        "- Main contribution: detrended anomaly attribution using event-year temporal-holdout residual models and grouped sparse counterfactual weather changes.",
        "- Explanation layer: report both dominant feature and physical driver group so each claim is crop-specific.",
        "- Robustness: use observed analog counterfactuals to show the weather replacement is historically plausible.",
        "- Supplementary contribution: crop-specific vulnerability profiles answer which weather stress lowers which crop.",
        "",
        "## Why This Is More Publishable Than V1 Alone",
        "",
        "- V1 proves the yield model and anomaly pipeline work, but its raw-yield attribution is conservative.",
        "- Residual and grouped methods target the actual object of interest: abnormal detrended yield shortfall.",
        "- Temporal holdout reduces in-sample attribution concerns before the paper makes event-level claims.",
        "- The global claim table states crop, region, year, driver group, dominant feature, and recovered t/ha.",
        "",
        "## Best Example Claims",
        "",
    ]
    main_claims = all_claims[all_claims["method"] == main["method"]]
    if main_claims.empty:
        main_claims = all_claims
    for _, r in main_claims.sort_values(["recoverable_fraction", "recovered_gap_t_ha"], ascending=False).head(10).iterrows():
        lines.append(f"- {r['claim_sentence']}")
    lines.extend(
        [
            "",
            "## Method Roles",
            "",
            f"- `06_grouped_driver_scaa_temporal_holdout`: main submission method; median phi {main['median_recoverable_fraction']:.3f}, high-recovery rate {main['weather_driven_rate']:.1%}.",
            f"- `02_grouped_driver_scaa`: in-sample exploratory grouped explanation; median phi {grouped['median_recoverable_fraction']:.3f}.",
            f"- `03_observed_analog_counterfactual`: robustness check; median phi {analog['median_recoverable_fraction']:.3f}.",
            "- `04_crop_specific_vulnerability_profiles`: use as a table answering which weather driver reduces each crop.",
            "- `05_early_mid_warning_improved`: keep as an operational extension, not the core paper contribution.",
            "",
            "## Crop-Specific Vulnerability Claims",
            "",
        ]
    )
    for _, r in vulnerability.sort_values("median_effect_t_ha").head(8).iterrows():
        lines.append(f"- {r['claim_sentence']}")
    write_lines(improve_root / "PAPER_CONTRIBUTION_DECISION.md", lines)


def fmt(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.3f}"


def run_improvement_suite(root: Path | None = None) -> None:
    project_root = root or Path(__file__).resolve().parents[2]
    paths = make_paths(project_root)
    df = load_frame(paths)
    validate_frame(df)
    scored, anomalies = score_anomalies_no_write(df)
    if len(anomalies) != 214:
        raise AssertionError(f"Expected 214 anomalies with current ddof=1 setting, found {len(anomalies)}")
    improve_root = project_root / ROOT_NAME
    improve_root.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        method_paths(improve_root, method)

    features = full_weather_features(scored)
    residual_model, numeric, categorical, residual_metrics = fit_residual_model(scored, features)

    all_scores: list[dict[str, Any]] = []
    claim_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []

    baseline_attr, score = run_baseline_v1(project_root, improve_root, scored)
    all_scores.append(score)
    claim_frames.append(common_claim_columns(baseline_attr, "00_baseline_v1_raw_yield_scaa"))
    event_frames.append(event_validation_rows("00_baseline_v1_raw_yield_scaa", baseline_attr))

    residual_attr, score = run_residual_sparse(
        improve_root, scored, anomalies, residual_model, numeric, categorical, features, residual_metrics
    )
    all_scores.append(score)
    claim_frames.append(common_claim_columns(residual_attr, "01_residual_target_scaa"))
    event_frames.append(event_validation_rows("01_residual_target_scaa", residual_attr))

    grouped_attr, score = run_grouped_scaa(improve_root, scored, anomalies, residual_model, numeric, categorical, features)
    all_scores.append(score)
    claim_frames.append(common_claim_columns(grouped_attr, "02_grouped_driver_scaa"))
    event_frames.append(event_validation_rows("02_grouped_driver_scaa", grouped_attr))

    temporal_attr, score = run_grouped_scaa_temporal_holdout(improve_root, scored, anomalies, numeric, categorical, features)
    all_scores.append(score)
    claim_frames.append(common_claim_columns(temporal_attr, "06_grouped_driver_scaa_temporal_holdout"))
    event_frames.append(event_validation_rows("06_grouped_driver_scaa_temporal_holdout", temporal_attr))

    analog_attr, score = run_observed_analog(improve_root, scored, anomalies, residual_model, numeric, categorical, features)
    all_scores.append(score)
    claim_frames.append(common_claim_columns(analog_attr, "03_observed_analog_counterfactual"))
    event_frames.append(event_validation_rows("03_observed_analog_counterfactual", analog_attr))

    vulnerability, score = run_vulnerability_profiles(improve_root, scored, residual_model, numeric, categorical, features)
    all_scores.append(score)

    _, score = run_warning_improved(improve_root, scored)
    all_scores.append(score)

    scorecard = pd.DataFrame(all_scores)
    all_claims = pd.concat(claim_frames, ignore_index=True)
    event_validation = pd.concat([f for f in event_frames if len(f)], ignore_index=True)
    write_global_reports(improve_root, scorecard, all_claims, event_validation, vulnerability)

    required_global = [
        improve_root / "COMPARISON_REPORT.md",
        improve_root / "PAPER_CONTRIBUTION_DECISION.md",
        improve_root / "crop_driver_claims.csv",
        improve_root / "event_validation_2012_2021_2022.csv",
        improve_root / "event_consistency_null_baselines.csv",
        improve_root / "method_scorecard.csv",
        improve_root
        / "06_grouped_driver_scaa_temporal_holdout"
        / "outputs"
        / "residual_model_validation.csv",
    ]
    missing = [str(p) for p in required_global if not p.exists()]
    if missing:
        raise AssertionError(f"Missing global improvement outputs: {missing}")
    if all_claims.empty or not all_claims["claim_sentence"].astype(str).str.contains("In ").any():
        raise AssertionError("crop_driver_claims.csv must contain paper-style claim sentences")
    if event_validation.empty:
        raise AssertionError("event_validation_2012_2021_2022.csv must contain event-year rows")
    assert_no_leakage_in_outputs(improve_root)
    assert_figures_nonblank(improve_root)

    print("Improvement suite complete.")
    print(f"Anomalies: {len(anomalies)}")
    print(f"Methods: {len(METHODS)}")
    print(f"Claims: {len(all_claims)}")
    print(f"Comparison report: {improve_root / 'COMPARISON_REPORT.md'}")
