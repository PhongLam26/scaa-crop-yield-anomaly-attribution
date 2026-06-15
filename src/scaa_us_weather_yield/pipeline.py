from __future__ import annotations

import json
import math
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
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "yield_t_ha"
CAT_FEATURES = ["region", "crop"]
NON_WEATHER_COLUMNS = {
    "country",
    "region",
    "crop",
    "year",
    "window",
    "lat",
    "lon",
    TARGET,
    "trend_yield_t_ha",
    "trend_residual_t_ha",
    "trend_residual_z",
    "is_low_yield_anomaly",
    "anomaly_label",
    "predicted_yield_t_ha",
    "residual_observed_minus_predicted",
}
ANOMALY_Z_THRESHOLD = -1.0
ANOMALY_STD_DDOF = 1
WEATHER_DRIVEN_THRESHOLD = 0.5
ATTRIBUTION_BUDGET = 4
RANDOM_STATE = 7


@dataclass(frozen=True)
class Paths:
    root: Path
    frame: Path
    outputs: Path
    figures: Path
    results_summary: Path


def make_paths(root: Path) -> Paths:
    return Paths(
        root=root,
        frame=root / "data" / "processed" / "us_model_frame_hemisphere_aware_1990_2025.csv",
        outputs=root / "outputs",
        figures=root / "figures",
        results_summary=root / "RESULTS_SUMMARY.md",
    )


def load_frame(paths: Paths) -> pd.DataFrame:
    df = pd.read_csv(paths.frame)
    validate_frame(df)
    return df


def validate_frame(df: pd.DataFrame) -> None:
    expected_crops = {"Barley", "Canola", "Oats", "Wheat"}
    if len(df) != 1257:
        raise AssertionError(f"Expected 1257 frame rows, found {len(df)}")
    if int(df["year"].min()) != 1990 or int(df["year"].max()) != 2025:
        raise AssertionError("Expected year range 1990-2025")
    crops = set(df["crop"].dropna().unique())
    if crops != expected_crops:
        raise AssertionError(f"Unexpected crops: {sorted(crops)}")
    if df[TARGET].isna().any():
        raise AssertionError("yield_t_ha contains missing values")


def weather_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_WEATHER_COLUMNS]


def model_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    weather = weather_columns(df)
    numeric = ["year", "lat", "lon"] + weather
    return weather, numeric, CAT_FEATURES


def build_regressor(df: pd.DataFrame) -> Pipeline:
    _, numeric, categorical = model_columns(df)
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
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    regressor = ExtraTreesRegressor(
        n_estimators=350,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", regressor)])


def build_classifier(df: pd.DataFrame, stage_features: list[str]) -> Pipeline:
    numeric = ["year", "lat", "lon"] + stage_features
    categorical = CAT_FEATURES
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
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    classifier = ExtraTreesClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", classifier)])


def r2_score_manual(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    denom = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - p) ** 2) / denom)


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))


def evaluate_yield_split(
    df: pd.DataFrame,
    train_end: int,
    test_start: int,
    test_end: int,
    label: str,
    paths: Paths,
) -> tuple[dict[str, Any], pd.DataFrame, Pipeline]:
    _, numeric, categorical = model_columns(df)
    train = df[df["year"] <= train_end].copy()
    test = df[(df["year"] >= test_start) & (df["year"] <= test_end)].copy()
    model = build_regressor(df)
    model.fit(train[numeric + categorical], train[TARGET])
    predictions = model.predict(test[numeric + categorical])
    test["predicted_yield_t_ha"] = predictions
    test["residual_observed_minus_predicted"] = test[TARGET] - test["predicted_yield_t_ha"]
    metrics = {
        "label": label,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "n_train": len(train),
        "n_test": len(test),
        "r2": r2_score_manual(test[TARGET], predictions),
        "rmse": rmse(test[TARGET], predictions),
    }
    out_name = f"yield_predictions_{test_start}_{test_end}.csv"
    test.to_csv(paths.outputs / out_name, index=False)
    return metrics, test, model


def train_yield_model(df: pd.DataFrame, paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    m1, pred_2016_2021, _ = evaluate_yield_split(
        df, 2015, 2016, 2021, "test_2016_2021", paths
    )
    m2, pred_2019_2025, _ = evaluate_yield_split(
        df, 2018, 2019, 2025, "test_2019_2025", paths
    )
    metrics.extend([m1, m2])

    _, numeric, categorical = model_columns(df)
    train = df[df["year"] <= 2015].copy()
    test = df[(df["year"] >= 2016) & (df["year"] <= 2021)].copy()
    model = build_regressor(df)
    model.fit(train[numeric + categorical], train[TARGET])
    test["predicted_yield_t_ha"] = model.predict(test[numeric + categorical])
    for window, group in test.groupby("window"):
        metrics.append(
            {
                "label": f"window_{window}_2016_2021",
                "train_end": 2015,
                "test_start": 2016,
                "test_end": 2021,
                "n_train": len(train),
                "n_test": len(group),
                "r2": r2_score_manual(group[TARGET], group["predicted_yield_t_ha"]),
                "rmse": rmse(group[TARGET], group["predicted_yield_t_ha"]),
            }
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(paths.outputs / "yield_model_metrics.csv", index=False)
    plot_yield_scatter(pred_2016_2021, paths)
    return metrics_df, pred_2019_2025


def detect_anomalies(df: pd.DataFrame, paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces: list[pd.DataFrame] = []
    for (crop, region), group in df.groupby(["crop", "region"], sort=True):
        g = group.sort_values("year").copy()
        x = g["year"].to_numpy(dtype=float)
        y = g[TARGET].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        trend = slope * x + intercept
        residual = y - trend
        std = np.std(residual, ddof=ANOMALY_STD_DDOF)
        if not np.isfinite(std) or std == 0:
            z = np.zeros_like(residual)
        else:
            z = residual / std
        g["trend_yield_t_ha"] = trend
        g["trend_residual_t_ha"] = residual
        g["trend_residual_z"] = z
        g["is_low_yield_anomaly"] = g["trend_residual_z"] < ANOMALY_Z_THRESHOLD
        pieces.append(g)

    scored = pd.concat(pieces, ignore_index=True).sort_values(["year", "crop", "region"])
    anomalies = scored[scored["is_low_yield_anomaly"]].copy()
    scored.to_csv(paths.outputs / "anomaly_scores_all_rows.csv", index=False)
    anomalies.to_csv(paths.outputs / "low_yield_anomalies.csv", index=False)
    plot_anomaly_timeline(anomalies, paths)
    return scored, anomalies


def attribution_features(df: pd.DataFrame) -> list[str]:
    stage_suffixes = ("_early", "_mid", "_late")
    return [
        c
        for c in weather_columns(df)
        if c != "season_days" and not c.endswith(stage_suffixes)
    ]


def feasible_stats(df: pd.DataFrame, features: list[str]) -> dict[tuple[str, str], pd.DataFrame]:
    stats: dict[tuple[str, str], pd.DataFrame] = {}
    quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    for key, group in df.groupby(["region", "window"]):
        rows = []
        for feature in features:
            values = group[feature].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            q = np.quantile(values, quantiles)
            std = float(np.std(values, ddof=ANOMALY_STD_DDOF))
            if not np.isfinite(std) or std == 0:
                std = float(np.std(df[feature].dropna().to_numpy(dtype=float), ddof=ANOMALY_STD_DDOF))
            if not np.isfinite(std) or std == 0:
                std = 1.0
            rows.append(
                {
                    "feature": feature,
                    "q0": q[0],
                    "q10": q[1],
                    "q25": q[2],
                    "q50": q[3],
                    "q75": q[4],
                    "q90": q[5],
                    "q100": q[6],
                    "std": std,
                }
            )
        stats[key] = pd.DataFrame(rows).set_index("feature")
    return stats


def predict_one(model: Pipeline, row: pd.Series, numeric: list[str], categorical: list[str]) -> float:
    frame = pd.DataFrame([row[numeric + categorical].to_dict()])
    return float(model.predict(frame)[0])


def attribute_one(
    model: Pipeline,
    row: pd.Series,
    numeric: list[str],
    categorical: list[str],
    features: list[str],
    stats: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    current = row.copy()
    base_pred = predict_one(model, current, numeric, categorical)
    target = float(row["trend_yield_t_ha"])
    needed = target - base_pred
    if needed <= 0:
        record = {
            "country": row["country"],
            "region": row["region"],
            "crop": row["crop"],
            "year": int(row["year"]),
            "window": row["window"],
            "actual_yield_t_ha": float(row[TARGET]),
            "trend_yield_t_ha": target,
            "model_predicted_yield_t_ha": base_pred,
            "counterfactual_predicted_yield_t_ha": base_pred,
            "model_gap_to_trend_t_ha": needed,
            "weather_recoverable_fraction": 0.0,
            "weather_driven": False,
            "dominant_driver": "model_gap_not_low",
            "n_changed_features": 0,
            "changed_features_json": "[]",
        }
        return record, []

    selected: set[str] = set()
    detail: list[dict[str, Any]] = []
    best_pred = base_pred
    for step in range(1, ATTRIBUTION_BUDGET + 1):
        candidates: list[pd.Series] = []
        candidate_meta: list[dict[str, Any]] = []
        for feature in features:
            if feature in selected or feature not in stats.index:
                continue
            observed = float(current[feature])
            row_stats = stats.loc[feature]
            candidate_values = [
                float(row_stats[q]) for q in ["q0", "q10", "q25", "q50", "q75", "q90", "q100"]
            ]
            for value in sorted(set(candidate_values)):
                if not np.isfinite(value) or abs(value - observed) < 1e-12:
                    continue
                candidate = current.copy()
                candidate[feature] = value
                cost = abs(value - observed) / float(row_stats["std"])
                if cost <= 0:
                    continue
                candidates.append(candidate[numeric + categorical])
                candidate_meta.append(
                    {
                        "feature": feature,
                        "from_value": observed,
                        "to_value": value,
                        "standardized_delta": cost,
                    }
                )
        if not candidates:
            break
        candidate_frame = pd.DataFrame([c.to_dict() for c in candidates])
        preds = model.predict(candidate_frame)
        best_index = None
        best_score = 0.0
        for i, pred in enumerate(preds):
            improvement = float(pred - best_pred)
            if improvement <= 1e-10:
                continue
            score = improvement / max(candidate_meta[i]["standardized_delta"], 1e-9)
            if score > best_score:
                best_score = score
                best_index = i
        if best_index is None:
            break
        choice = candidate_meta[best_index]
        feature = choice["feature"]
        current[feature] = choice["to_value"]
        selected.add(feature)
        new_pred = float(preds[best_index])
        detail.append(
            {
                "country": row["country"],
                "region": row["region"],
                "crop": row["crop"],
                "year": int(row["year"]),
                "window": row["window"],
                "step": step,
                "feature": feature,
                "from_value": choice["from_value"],
                "to_value": choice["to_value"],
                "delta": choice["to_value"] - choice["from_value"],
                "standardized_delta": choice["standardized_delta"],
                "prediction_before": best_pred,
                "prediction_after": new_pred,
                "prediction_gain": new_pred - best_pred,
            }
        )
        best_pred = new_pred
        if (best_pred - base_pred) / needed >= 0.95:
            break

    recoverable = float(np.clip((best_pred - base_pred) / needed, 0.0, 1.0))
    if detail:
        dominant = max(detail, key=lambda d: abs(float(d["standardized_delta"])))["feature"]
    else:
        dominant = "no_feasible_weather_gain"
    compact_changes = [
        {
            "feature": d["feature"],
            "from": round(float(d["from_value"]), 4),
            "to": round(float(d["to_value"]), 4),
            "standardized_delta": round(float(d["standardized_delta"]), 4),
            "prediction_gain": round(float(d["prediction_gain"]), 4),
        }
        for d in detail
    ]
    record = {
        "country": row["country"],
        "region": row["region"],
        "crop": row["crop"],
        "year": int(row["year"]),
        "window": row["window"],
        "actual_yield_t_ha": float(row[TARGET]),
        "trend_yield_t_ha": target,
        "model_predicted_yield_t_ha": base_pred,
        "counterfactual_predicted_yield_t_ha": best_pred,
        "model_gap_to_trend_t_ha": needed,
        "weather_recoverable_fraction": recoverable,
        "weather_driven": bool(recoverable >= WEATHER_DRIVEN_THRESHOLD),
        "dominant_driver": dominant,
        "n_changed_features": len(detail),
        "changed_features_json": json.dumps(compact_changes),
    }
    return record, detail


def attribute_counterfactuals(
    df_scored: pd.DataFrame,
    anomalies: pd.DataFrame,
    paths: Paths,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df_scored.copy()
    _, numeric, categorical = model_columns(df)
    final_model = build_regressor(df)
    final_model.fit(df[numeric + categorical], df[TARGET])
    features = attribution_features(df)
    stats_map = feasible_stats(df, features)
    records: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for _, row in anomalies.sort_values(["year", "crop", "region"]).iterrows():
        key = (row["region"], row["window"])
        record, detail = attribute_one(
            final_model,
            row,
            numeric,
            categorical,
            features,
            stats_map[key],
        )
        records.append(record)
        details.extend(detail)

    attribution = pd.DataFrame(records)
    change_details = pd.DataFrame(details)
    attribution.to_csv(paths.outputs / "counterfactual_attributions.csv", index=False)
    change_details.to_csv(paths.outputs / "counterfactual_feature_changes.csv", index=False)
    plot_recoverable_distribution(attribution, paths)
    plot_driver_frequency(attribution, paths)
    return attribution, change_details


def conformal_quantile(scores: np.ndarray, alpha: float = 0.1) -> float:
    clean = np.sort(np.asarray(scores, dtype=float))
    n = len(clean)
    if n == 0:
        return 1.0
    rank = int(math.ceil((n + 1) * (1 - alpha)))
    rank = min(max(rank, 1), n)
    return float(clean[rank - 1])


def safe_auc(y_true: pd.Series, scores: np.ndarray) -> float:
    if len(set(y_true.astype(int))) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def early_warning_baseline(df_scored: pd.DataFrame, paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df_scored.copy()
    df["anomaly_label"] = df["is_low_yield_anomaly"].astype(int)
    early = [c for c in weather_columns(df) if c.endswith("_early")]
    mid = [c for c in weather_columns(df) if c.endswith("_mid")]
    stages = {
        "early_third": early,
        "early_mid_two_thirds": early + mid,
    }
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for stage, features in stages.items():
        if not features:
            raise AssertionError(f"No features found for {stage}")
        train = df[df["year"] <= 2015].copy()
        calibration = df[(df["year"] >= 2016) & (df["year"] <= 2018)].copy()
        test = df[df["year"] >= 2019].copy()
        model = build_classifier(df, features)
        columns = ["year", "lat", "lon"] + features + CAT_FEATURES
        model.fit(train[columns], train["anomaly_label"])
        calib_prob = model.predict_proba(calibration[columns])[:, 1]
        calib_scores = np.abs(calibration["anomaly_label"].to_numpy(dtype=float) - calib_prob)
        q = conformal_quantile(calib_scores, alpha=0.1)
        test_prob = model.predict_proba(test[columns])[:, 1]
        test_pred = (test_prob >= 0.5).astype(int)
        lower = np.clip(test_prob - q, 0.0, 1.0)
        upper = np.clip(test_prob + q, 0.0, 1.0)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test["anomaly_label"],
            test_pred,
            average="binary",
            zero_division=0,
        )
        metrics.append(
            {
                "stage": stage,
                "n_features": len(features),
                "n_train": len(train),
                "n_calibration": len(calibration),
                "n_test": len(test),
                "anomaly_rate_test": float(test["anomaly_label"].mean()),
                "roc_auc": safe_auc(test["anomaly_label"], test_prob),
                "average_precision": float(average_precision_score(test["anomaly_label"], test_prob)),
                "brier_score": float(brier_score_loss(test["anomaly_label"], test_prob)),
                "accuracy_at_0_5": float(accuracy_score(test["anomaly_label"], test_pred)),
                "precision_at_0_5": float(precision),
                "recall_at_0_5": float(recall),
                "f1_at_0_5": float(f1),
                "conformal_abs_error_q90": q,
                "label_coverage_by_probability_interval": float(
                    ((test["anomaly_label"] >= lower) & (test["anomaly_label"] <= upper)).mean()
                ),
            }
        )
        pred = test[
            [
                "country",
                "region",
                "crop",
                "year",
                "window",
                TARGET,
                "trend_yield_t_ha",
                "trend_residual_z",
                "anomaly_label",
            ]
        ].copy()
        pred["stage"] = stage
        pred["anomaly_probability"] = test_prob
        pred["probability_lower_conformal_style"] = lower
        pred["probability_upper_conformal_style"] = upper
        predictions.append(pred)

    metrics_df = pd.DataFrame(metrics)
    predictions_df = pd.concat(predictions, ignore_index=True)
    metrics_df.to_csv(paths.outputs / "early_warning_metrics.csv", index=False)
    predictions_df.to_csv(paths.outputs / "early_warning_predictions.csv", index=False)
    plot_early_warning(metrics_df, paths)
    return metrics_df, predictions_df


def plot_yield_scatter(predictions: pd.DataFrame, paths: Paths) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.scatter(predictions[TARGET], predictions["predicted_yield_t_ha"], alpha=0.72, s=28)
    lo = min(predictions[TARGET].min(), predictions["predicted_yield_t_ha"].min())
    hi = max(predictions[TARGET].max(), predictions["predicted_yield_t_ha"].max())
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.1)
    ax.set_xlabel("Observed yield (t/ha)")
    ax.set_ylabel("Predicted yield (t/ha)")
    ax.set_title("Forward-time yield prediction, test 2016-2021")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths.figures / "yield_observed_vs_predicted_2016_2021.png", dpi=180)
    plt.close(fig)


def plot_anomaly_timeline(anomalies: pd.DataFrame, paths: Paths) -> None:
    counts = anomalies.groupby("year").size().reindex(range(1990, 2026), fill_value=0)
    colors = ["#c44747" if y in {2012, 2021, 2022} else "#4d7ea8" for y in counts.index]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_xlabel("Year")
    ax.set_ylabel("Low-yield anomaly count")
    ax.set_title("Detrended low-yield anomalies by year")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths.figures / "anomaly_timeline.png", dpi=180)
    plt.close(fig)


def plot_recoverable_distribution(attribution: pd.DataFrame, paths: Paths) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(attribution["weather_recoverable_fraction"], bins=np.linspace(0, 1, 16), color="#5f8f6b", edgecolor="white")
    ax.axvline(WEATHER_DRIVEN_THRESHOLD, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Weather-recoverable fraction")
    ax.set_ylabel("Anomaly count")
    ax.set_title("Counterfactual weather recoverability")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths.figures / "recoverable_fraction_distribution.png", dpi=180)
    plt.close(fig)


def plot_driver_frequency(attribution: pd.DataFrame, paths: Paths) -> None:
    exclude = {"model_gap_not_low", "no_feasible_weather_gain", "none"}
    counts = attribution[~attribution["dominant_driver"].isin(exclude)]["dominant_driver"].value_counts().head(12)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    if len(counts) > 0:
        counts.sort_values().plot(kind="barh", ax=ax, color="#8b6f47")
    ax.set_xlabel("Attributed anomaly count")
    ax.set_ylabel("Dominant driver")
    ax.set_title("Most frequent dominant weather drivers")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths.figures / "dominant_driver_frequency.png", dpi=180)
    plt.close(fig)


def plot_early_warning(metrics: pd.DataFrame, paths: Paths) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    stages = metrics["stage"].tolist()
    axes[0].bar(stages, metrics["average_precision"], color="#4d7ea8")
    axes[0].set_title("Average precision")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=18)
    axes[1].bar(stages, metrics["brier_score"], color="#c47f47")
    axes[1].set_title("Brier score")
    axes[1].set_ylim(0, max(0.25, float(metrics["brier_score"].max()) * 1.2))
    axes[1].tick_params(axis="x", rotation=18)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Early-warning baseline: early vs early+mid season")
    fig.tight_layout()
    fig.savefig(paths.figures / "early_mid_warning_comparison.png", dpi=180)
    plt.close(fig)


def write_results_summary(
    df: pd.DataFrame,
    yield_metrics: pd.DataFrame,
    anomalies: pd.DataFrame,
    attribution: pd.DataFrame,
    early_metrics: pd.DataFrame,
    paths: Paths,
) -> None:
    main_2016 = yield_metrics[yield_metrics["label"] == "test_2016_2021"].iloc[0]
    main_2019 = yield_metrics[yield_metrics["label"] == "test_2019_2025"].iloc[0]
    anomaly_count = int(len(anomalies))
    anomaly_rate = anomaly_count / len(df)
    weather_driven_rate = float(attribution["weather_driven"].mean()) if len(attribution) else 0.0
    median_recoverable = float(attribution["weather_recoverable_fraction"].median()) if len(attribution) else 0.0
    top_years = anomalies.groupby("year").size().sort_values(ascending=False).head(8)
    top_drivers = (
        attribution[~attribution["dominant_driver"].isin({"model_gap_not_low", "no_feasible_weather_gain", "none"})][
            "dominant_driver"
        ]
        .value_counts()
        .head(8)
    )

    lines = [
        "# Results Summary",
        "",
        "This file is generated by `python scripts/run_all.py`.",
        "",
        "## Yield Model Reproduction",
        "",
        f"- Frame: {len(df):,} crop-region-year rows, {df['year'].min()}-{df['year'].max()}.",
        f"- Test 2016-2021: R2={main_2016['r2']:.3f}, RMSE={main_2016['rmse']:.3f}, n={int(main_2016['n_test'])}.",
        f"- Test 2019-2025: R2={main_2019['r2']:.3f}, RMSE={main_2019['rmse']:.3f}, n={int(main_2019['n_test'])}.",
        "",
        "## Detrended Anomalies",
        "",
        f"- Low-yield anomalies: {anomaly_count:,} of {len(df):,} rows ({anomaly_rate:.1%}) using z < {ANOMALY_Z_THRESHOLD:g}.",
        f"- Trend standardization uses sample residual standard deviation within each crop-region series (ddof={ANOMALY_STD_DDOF}).",
        "- Top anomaly years:",
    ]
    for year, count in top_years.items():
        lines.append(f"  - {int(year)}: {int(count)} anomalies")
    lines.extend(
        [
            "",
            "## Sparse Counterfactual Attribution",
            "",
            f"- Median weather-recoverable fraction: {median_recoverable:.3f}.",
            f"- Weather-driven anomalies at threshold {WEATHER_DRIVEN_THRESHOLD:.1f}: {weather_driven_rate:.1%}.",
            "- Most frequent dominant drivers:",
        ]
    )
    for driver, count in top_drivers.items():
        lines.append(f"  - {driver}: {int(count)}")
    lines.extend(["", "## Early-Warning Baseline", ""])
    for _, row in early_metrics.iterrows():
        lines.append(
            "- "
            f"{row['stage']}: AP={row['average_precision']:.3f}, "
            f"ROC_AUC={row['roc_auc']:.3f}, "
            f"Brier={row['brier_score']:.3f}, "
            f"F1@0.5={row['f1_at_0_5']:.3f}, "
            f"q90 interval radius={row['conformal_abs_error_q90']:.3f}."
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `outputs/yield_model_metrics.csv`",
            "- `outputs/anomaly_scores_all_rows.csv` and `outputs/low_yield_anomalies.csv`",
            "- `outputs/counterfactual_attributions.csv` and `outputs/counterfactual_feature_changes.csv`",
            "- `outputs/early_warning_metrics.csv` and `outputs/early_warning_predictions.csv`",
            "- `figures/*.png`",
            "",
            "All attribution results are model-internal predictive explanations, not causal effects.",
            "",
        ]
    )
    paths.results_summary.write_text("\n".join(lines), encoding="utf-8")


def run_all(root: Path | None = None) -> None:
    project_root = root or Path(__file__).resolve().parents[2]
    paths = make_paths(project_root)
    paths.outputs.mkdir(parents=True, exist_ok=True)
    paths.figures.mkdir(parents=True, exist_ok=True)

    df = load_frame(paths)
    yield_metrics, _ = train_yield_model(df, paths)
    scored, anomalies = detect_anomalies(df, paths)
    attribution, _ = attribute_counterfactuals(scored, anomalies, paths)
    early_metrics, _ = early_warning_baseline(scored, paths)
    write_results_summary(df, yield_metrics, anomalies, attribution, early_metrics, paths)

    required_outputs = [
        paths.outputs / "yield_model_metrics.csv",
        paths.outputs / "anomaly_scores_all_rows.csv",
        paths.outputs / "low_yield_anomalies.csv",
        paths.outputs / "counterfactual_attributions.csv",
        paths.outputs / "counterfactual_feature_changes.csv",
        paths.outputs / "early_warning_metrics.csv",
        paths.outputs / "early_warning_predictions.csv",
        paths.figures / "yield_observed_vs_predicted_2016_2021.png",
        paths.figures / "anomaly_timeline.png",
        paths.figures / "recoverable_fraction_distribution.png",
        paths.figures / "dominant_driver_frequency.png",
        paths.figures / "early_mid_warning_comparison.png",
        paths.results_summary,
    ]
    missing = [str(p) for p in required_outputs if not p.exists()]
    if missing:
        raise AssertionError(f"Missing expected outputs: {missing}")

    main = yield_metrics.set_index("label")
    if abs(float(main.loc["test_2016_2021", "r2"]) - 0.809) > 0.015:
        raise AssertionError("2016-2021 R2 drifted beyond tolerance")
    if abs(float(main.loc["test_2019_2025", "r2"]) - 0.808) > 0.015:
        raise AssertionError("2019-2025 R2 drifted beyond tolerance")
    for column in ["trend_residual_z", "trend_yield_t_ha", "trend_residual_t_ha", "is_low_yield_anomaly"]:
        if column not in scored.columns:
            raise AssertionError(f"Missing anomaly column: {column}")
    if not attribution["weather_recoverable_fraction"].between(0, 1).all():
        raise AssertionError("Recoverable fractions must be in [0, 1]")
    recovered = attribution["weather_recoverable_fraction"] > 0
    if recovered.any() and attribution.loc[recovered, "dominant_driver"].eq("").any():
        raise AssertionError("Recovered anomalies must have a dominant driver")

    print("SCAA prototype run complete.")
    print(f"Rows: {len(df)}")
    print(
        "Yield R2: "
        f"2016-2021={main.loc['test_2016_2021', 'r2']:.3f}, "
        f"2019-2025={main.loc['test_2019_2025', 'r2']:.3f}"
    )
    print(f"Low-yield anomalies: {len(anomalies)}")
    print(f"Attribution rows: {len(attribution)}")
    print(f"Results summary: {paths.results_summary}")
