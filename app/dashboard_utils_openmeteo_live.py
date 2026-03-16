from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import pickle

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HORIZON_STEPS = {
    "Next 3 hours": 1,
    "Next 6 hours": 2,
    "Next 12 hours": 4,
    "Next 1 day": 8,
    "Next 2 days": 16,
}

TORONTO_LAT = 43.6532
TORONTO_LON = -79.3832
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
LIVE_WEATHER_FIELDS = [
    "temperature",
    "rain",
    "snow",
    "wind_speed",
    "relative_humidity",
    "visibility",
]


@dataclass
class AppPaths:
    repo_root: Path
    forecast_source: Path
    geojson: Path
    models_dir: Path


def _find_repo_root(app_dir: Path) -> Path:
    """Support running the dashboard either from the repo root or a subfolder."""
    candidates = [app_dir, app_dir.parent, app_dir.parent.parent]
    for root in candidates:
        if (root / "data").exists() or (root / "models").exists():
            return root
    return app_dir.parent


def normalize_hood_code(value: object) -> str:
    """Normalize neighbourhood codes so joins work for values like 41 vs 041."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    try:
        num = int(float(text))
        return f"{num:03d}"
    except Exception:
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            return digits.zfill(3)
        return text


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def resolve_paths(base_dir: Optional[Path] = None) -> AppPaths:
    app_dir = base_dir or Path(__file__).resolve().parent
    repo_root = _find_repo_root(app_dir)
    processed = repo_root / "data" / "processed"
    raw = repo_root / "data" / "data_raw"

    forecast_candidates = [
        processed / "supervised_hood_3h_multiclass.xlsx",
        processed / "supervised_hood_3h_multiclass.csv",
        repo_root / "supervised_hood_3h_multiclass.xlsx",
        repo_root / "supervised_hood_3h_multiclass.csv",
        processed / "dashboard_hood_3h_weather.csv",
        repo_root / "dashboard_hood_3h_weather.csv",
    ]
    geo_candidates = [
        raw / "Neighbourhoods.geojson",
        raw / "Neighborhoods.geojson",
        repo_root / "Neighbourhoods.geojson",
        repo_root / "Neighborhoods.geojson",
    ]

    forecast_source = next((p for p in forecast_candidates if p.exists()), forecast_candidates[0])
    geojson = next((p for p in geo_candidates if p.exists()), geo_candidates[0])
    models_dir = repo_root / "models"
    return AppPaths(repo_root=repo_root, forecast_source=forecast_source, geojson=geojson, models_dir=models_dir)


class DemoRiskModel:
    def __init__(self) -> None:
        self.feature_columns: List[str] = []
        self.model: Optional[Pipeline] = None

    def fit(self, df: pd.DataFrame) -> "DemoRiskModel":
        work = df.copy()
        if "y_class" not in work.columns:
            if "collisions" in work.columns:
                work["y_class"] = np.where(work["collisions"] >= 2, 2, np.where(work["collisions"] == 1, 1, 0))
            else:
                work["y_class"] = 0

        features = [
            c
            for c in [
                "HOOD_158_CODE",
                "block_hour",
                "dow_num",
                "month_num",
                "is_weekend",
                "temperature",
                "rain",
                "snow",
                "wind_speed",
                "relative_humidity",
                "visibility",
            ]
            if c in work.columns
        ]
        if not features:
            raise ValueError("DemoRiskModel could not find any usable feature columns.")

        self.feature_columns = features
        X = work[features].copy()
        y = work["y_class"].astype(int)

        cat_cols = [c for c in ["HOOD_158_CODE"] if c in features]
        num_cols = [c for c in features if c not in cat_cols]
        preprocess = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    num_cols,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    cat_cols,
                ),
            ],
            remainder="drop",
        )
        clf = LogisticRegression(max_iter=1200, class_weight="balanced", solver="lbfgs", random_state=42)
        self.model = Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("DemoRiskModel must be fitted before prediction.")
        return self.model.predict_proba(X[self.feature_columns].copy())


def load_geojson(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Forecast source not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if "time_3h" in df.columns:
        df["time_3h"] = pd.to_datetime(df["time_3h"], errors="coerce")
    if "HOOD_158_CODE" in df.columns:
        df["HOOD_158_CODE"] = pd.to_numeric(df["HOOD_158_CODE"], errors="coerce").fillna(-1).astype(int)
    if "collisions" not in df.columns and "y_class" in df.columns:
        df["collisions"] = df["y_class"].map({0: 0, 1: 1, 2: 2}).fillna(0)
    return df


def load_best_bundle(models_dir: Path) -> Tuple[Optional[dict], Optional[Path]]:
    if not models_dir.exists():
        return None, None

    preferred = [
        "stacking_3model_dashboard_bundle.pkl",
        "stacking_4model_dashboard_bundle.pkl",
        "13_stacking_oof_dashboard_bundle.pkl",
    ]
    candidates: List[Path] = []
    for name in preferred:
        p = models_dir / name
        if p.exists():
            candidates.append(p)
    candidates.extend(sorted(models_dir.glob("*dashboard_bundle*.pkl")))

    seen = set()
    unique_candidates: List[Path] = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique_candidates.append(p)

    for p in unique_candidates:
        try:
            bundle = joblib.load(p)
            if isinstance(bundle, dict) and "meta_model" in bundle:
                return bundle, p
        except Exception:
            pass
        try:
            with p.open("rb") as f:
                bundle = pickle.load(f)
            if isinstance(bundle, dict) and "meta_model" in bundle:
                return bundle, p
        except Exception:
            continue
    return None, None


def derive_temporal_fields(ts: pd.Timestamp) -> Dict[str, object]:
    dow_num = int(ts.dayofweek)
    return {
        "time_3h": ts,
        "block_hour": int(ts.hour),
        "OCC_DOW": ts.day_name(),
        "OCC_MONTH": ts.month_name(),
        "dow_num": dow_num,
        "month_num": int(ts.month),
        "is_weekend": int(dow_num >= 5),
        "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
        "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow_num / 7),
        "dow_cos": np.cos(2 * np.pi * dow_num / 7),
        "month_sin": np.sin(2 * np.pi * ts.month / 12),
        "month_cos": np.cos(2 * np.pi * ts.month / 12),
    }


def geo_lookup_table(geojson: dict) -> pd.DataFrame:
    rows = []
    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        rows.append(
            {
                "HOOD_158_CODE": normalize_hood_code(props.get("AREA_SHORT_CODE")),
                "hood_name": props.get("AREA_NAME", "Unknown"),
                "hood_desc": props.get("AREA_DESC", ""),
            }
        )
    return pd.DataFrame(rows)


def build_latest_template(history: pd.DataFrame, geojson: Optional[dict] = None) -> pd.DataFrame:
    work = history.copy()
    if "time_3h" in work.columns:
        work = work.dropna(subset=["time_3h"])
        latest_idx = work.sort_values("time_3h").groupby("HOOD_158_CODE")["time_3h"].idxmax()
        latest = work.loc[latest_idx].sort_values("HOOD_158_CODE").reset_index(drop=True)
    else:
        latest = work.drop_duplicates(subset=["HOOD_158_CODE"], keep="last").sort_values("HOOD_158_CODE").reset_index(drop=True)

    if geojson is None:
        return latest

    geo_codes = geo_lookup_table(geojson)["HOOD_158_CODE"].tolist()
    present = {normalize_hood_code(v) for v in latest["HOOD_158_CODE"].tolist()}
    missing = [code for code in geo_codes if code not in present]
    if not missing:
        return latest

    fallback: Dict[str, object] = {}
    for col in latest.columns:
        if col == "HOOD_158_CODE":
            continue
        if pd.api.types.is_numeric_dtype(latest[col]):
            fallback[col] = float(latest[col].median()) if latest[col].notna().any() else 0.0
        else:
            mode = latest[col].mode(dropna=True)
            fallback[col] = mode.iloc[0] if len(mode) else None

    filler_rows = []
    latest_time = latest["time_3h"].max() if "time_3h" in latest.columns else pd.Timestamp.now().floor("3h")
    for code in missing:
        row = {"HOOD_158_CODE": int(code)}
        row.update(fallback)
        if "time_3h" in latest.columns:
            row["time_3h"] = latest_time
        filler_rows.append(row)

    latest = pd.concat([latest, pd.DataFrame(filler_rows)], ignore_index=True)
    latest = latest.sort_values("HOOD_158_CODE").reset_index(drop=True)
    return latest


def bundle_feature_fill_values(history: pd.DataFrame, feature_columns: List[str]) -> Tuple[Dict[str, float], Dict[str, object]]:
    numeric_fill: Dict[str, float] = {}
    other_fill: Dict[str, object] = {}
    for col in feature_columns:
        if col in history.columns:
            if pd.api.types.is_numeric_dtype(history[col]):
                numeric_fill[col] = float(history[col].median()) if history[col].notna().any() else 0.0
            else:
                mode = history[col].mode(dropna=True)
                other_fill[col] = mode.iloc[0] if len(mode) else "Unknown"
        else:
            numeric_fill[col] = 0.0
    return numeric_fill, other_fill


def fetch_openmeteo_hourly(forecast_start: pd.Timestamp, hours_ahead: int = 48) -> Tuple[pd.DataFrame, dict]:
    """Fetch Toronto hourly weather from Open-Meteo and standardize names/units."""
    start_date = forecast_start.normalize().date().isoformat()
    end_date = (forecast_start + pd.Timedelta(hours=max(hours_ahead, 48))).normalize().date().isoformat()
    params = {
        "latitude": TORONTO_LAT,
        "longitude": TORONTO_LON,
        "hourly": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "rain",
                "snowfall",
                "wind_speed_10m",
                "visibility",
            ]
        ),
        "timezone": "America/Toronto",
        "start_date": start_date,
        "end_date": end_date,
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "temperature_unit": "celsius",
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    hourly = payload.get("hourly", {})
    df = pd.DataFrame(hourly)
    if df.empty:
        raise ValueError("Open-Meteo returned no hourly rows.")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    rename = {
        "temperature_2m": "temperature",
        "relative_humidity_2m": "relative_humidity",
        "snowfall": "snow",
        "wind_speed_10m": "wind_speed",
    }
    df = df.rename(columns=rename)
    for col in ["temperature", "relative_humidity", "rain", "snow", "wind_speed", "visibility"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["visibility"] = df["visibility"] / 1000.0  # meters -> km

    meta = {
        "source": "Open-Meteo",
        "latitude": payload.get("latitude", TORONTO_LAT),
        "longitude": payload.get("longitude", TORONTO_LON),
        "timezone": payload.get("timezone", "America/Toronto"),
    }
    return df.sort_values("time").reset_index(drop=True), meta


def get_live_inputs_for_start(forecast_start: pd.Timestamp) -> Tuple[Dict[str, float], dict, pd.DataFrame]:
    wx_df, meta = fetch_openmeteo_hourly(forecast_start)
    rounded = forecast_start.floor("h")
    nearest_idx = (wx_df["time"] - rounded).abs().idxmin()
    row = wx_df.loc[nearest_idx]
    inputs = {
        "temperature": safe_float(row.get("temperature"), 0.0),
        "rain": safe_float(row.get("rain"), 0.0),
        "snow": safe_float(row.get("snow"), 0.0),
        "wind_speed": safe_float(row.get("wind_speed"), 0.0),
        "relative_humidity": safe_float(row.get("relative_humidity"), 0.0),
        "visibility": safe_float(row.get("visibility"), 10.0),
    }
    return inputs, meta, wx_df


def weather_inputs_for_timestamp(
    weather_forecast: Optional[pd.DataFrame],
    ts: pd.Timestamp,
    fallback_inputs: Dict[str, float],
) -> Dict[str, float]:
    inputs = {k: safe_float(v, 0.0) for k, v in fallback_inputs.items()}
    if weather_forecast is None or weather_forecast.empty or "time" not in weather_forecast.columns:
        return inputs

    nearest_idx = (weather_forecast["time"] - ts.floor("h")).abs().idxmin()
    row = weather_forecast.loc[nearest_idx]
    for field in LIVE_WEATHER_FIELDS:
        default = inputs.get(field, 0.0 if field != "visibility" else 10.0)
        inputs[field] = safe_float(row.get(field), default)
    return inputs


def make_future_feature_frame(
    history: pd.DataFrame,
    forecast_start: pd.Timestamp,
    horizon_steps: int,
    user_inputs: Dict[str, float],
    feature_columns: Optional[List[str]] = None,
    geojson: Optional[dict] = None,
    weather_forecast: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    latest = build_latest_template(history, geojson=geojson)
    numeric_fill, other_fill = bundle_feature_fill_values(history, feature_columns or list(latest.columns))

    frames = []
    for step in range(1, horizon_steps + 1):
        ts = forecast_start + pd.Timedelta(hours=3 * step)
        frame = latest.copy()
        for k, v in derive_temporal_fields(ts).items():
            frame[k] = v

        step_inputs = weather_inputs_for_timestamp(weather_forecast, ts, user_inputs)
        for k, v in step_inputs.items():
            frame[k] = v

        frame["forecast_step"] = step
        frame["forecast_time"] = ts
        frames.append(frame)

    future = pd.concat(frames, ignore_index=True)

    if feature_columns is not None:
        for col in feature_columns:
            if col not in future.columns:
                if col in other_fill:
                    future[col] = other_fill[col]
                else:
                    future[col] = numeric_fill.get(col, 0.0)
        future = future[[c for c in feature_columns] + ["forecast_step", "forecast_time"]]

    return future


def _predict_base_model_proba(model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(X), dtype=float)


def _standardize_multiclass_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2:
        raise ValueError(f"Expected a 2D probability array, got shape {proba.shape}")
    if proba.shape[1] == 3:
        return proba
    if proba.shape[1] == 2:
        pad = np.zeros((proba.shape[0], 1), dtype=float)
        return np.hstack([proba, pad])
    raise ValueError(f"Expected 2 or 3 probability columns, got {proba.shape[1]}")


def predict_with_bundle(bundle: dict, future_features: pd.DataFrame) -> pd.DataFrame:
    feature_columns = list(bundle.get("feature_columns", []))
    base_model_order = list(bundle.get("base_model_order", []))
    base_models = bundle.get("base_models", {})
    meta_model = bundle.get("meta_model")

    if meta_model is None:
        raise ValueError("Bundle is missing 'meta_model'.")

    X = future_features[feature_columns].copy() if feature_columns else future_features.copy()

    meta_parts = []
    for model_name in base_model_order:
        if model_name not in base_models:
            raise KeyError(f"Base model '{model_name}' listed in bundle order but missing in bundle['base_models'].")
        model = base_models[model_name]
        meta_parts.append(_predict_base_model_proba(model, X))

    if not meta_parts:
        raise ValueError("Bundle did not provide any base-model probabilities.")

    X_meta = np.hstack(meta_parts)
    final_proba = _standardize_multiclass_proba(np.asarray(meta_model.predict_proba(X_meta), dtype=float))
    pred_class = final_proba.argmax(axis=1)

    out = future_features[["forecast_step", "forecast_time"]].copy()
    hood_source = X["HOOD_158_CODE"] if "HOOD_158_CODE" in X.columns else future_features["HOOD_158_CODE"]
    out["HOOD_158_CODE"] = hood_source.map(normalize_hood_code)
    out["pred_class"] = pred_class.astype(int)
    out["p0"] = final_proba[:, 0]
    out["p1"] = final_proba[:, 1]
    out["p2"] = final_proba[:, 2]
    out["risk_score"] = 0.5 * out["p1"] + 1.0 * out["p2"] + out["pred_class"] * 0.25
    return out


def predict_with_fallback(model: DemoRiskModel, future_features: pd.DataFrame) -> pd.DataFrame:
    proba = _standardize_multiclass_proba(model.predict_proba(future_features))
    pred_class = proba.argmax(axis=1)
    out = future_features[["forecast_step", "forecast_time", "HOOD_158_CODE"]].copy()
    out["HOOD_158_CODE"] = out["HOOD_158_CODE"].map(normalize_hood_code)
    out["pred_class"] = pred_class.astype(int)
    out["p0"] = proba[:, 0]
    out["p1"] = proba[:, 1]
    out["p2"] = proba[:, 2]
    out["risk_score"] = 0.5 * out["p1"] + 1.0 * out["p2"] + out["pred_class"] * 0.25
    return out


def aggregate_horizon_predictions(pred_df: pd.DataFrame, horizon_label: str | None = None) -> pd.DataFrame:
    grouped = pred_df.groupby("HOOD_158_CODE", as_index=False).agg(
        risk_score=("risk_score", "max"),
        pred_class=("pred_class", "max"),
        p0=("p0", "mean"),
        p1=("p1", "mean"),
        p2=("p2", "mean"),
        forecast_end=("forecast_time", "max"),
    )

    def bucket(score: float, pred_class: int) -> str:
        if pred_class >= 2 or score >= 1.35:
            return "High"
        if pred_class >= 1 or score >= 0.75:
            return "Medium"
        return "Low"

    grouped["risk_bucket"] = [bucket(s, c) for s, c in zip(grouped["risk_score"], grouped["pred_class"])]
    grouped["HOOD_158_CODE"] = grouped["HOOD_158_CODE"].map(normalize_hood_code)
    grouped["forecast_end"] = pd.to_datetime(grouped["forecast_end"], errors="coerce")
    if horizon_label is not None:
        grouped["horizon"] = horizon_label
    return grouped


def merge_geo_labels(pred_agg: pd.DataFrame, geojson: dict) -> pd.DataFrame:
    lookup = geo_lookup_table(geojson).copy()
    pred_work = pred_agg.copy()

    lookup["HOOD_158_CODE"] = lookup["HOOD_158_CODE"].map(normalize_hood_code)
    pred_work["HOOD_158_CODE"] = pred_work["HOOD_158_CODE"].map(normalize_hood_code)

    out = lookup.merge(pred_work, on="HOOD_158_CODE", how="left")
    out["pred_class"] = out["pred_class"].fillna(0).astype(int)
    out["p0"] = out["p0"].fillna(1.0)
    out["p1"] = out["p1"].fillna(0.0)
    out["p2"] = out["p2"].fillna(0.0)
    out["risk_score"] = out["risk_score"].fillna(0.0)
    out["risk_bucket"] = out["risk_bucket"].fillna("Low")
    out["forecast_end"] = pd.to_datetime(out["forecast_end"], errors="coerce")
    out["forecast_end"] = out["forecast_end"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("N/A")
    if "horizon" in out.columns:
        out["horizon"] = out["horizon"].fillna("N/A")
    return out


def top_hotspots(pred_agg: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    cols = ["HOOD_158_CODE", "hood_name", "risk_bucket", "pred_class", "risk_score", "p2", "forecast_end"]
    return (
        pred_agg.sort_values(["pred_class", "risk_score", "p2"], ascending=[False, False, False])
        .head(n)[cols]
        .reset_index(drop=True)
    )


def recent_history_for_hood(history: pd.DataFrame, hood_code: str, periods: int = 50) -> pd.DataFrame:
    hood_int = int(normalize_hood_code(hood_code))
    out = history.loc[history["HOOD_158_CODE"] == hood_int].sort_values("time_3h").tail(periods).copy()
    return out
