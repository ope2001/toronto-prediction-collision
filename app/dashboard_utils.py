from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HORIZON_STEPS = {
    "Next 3 hours": 1,
    "Next 6 hours": 2,
    "Next 12 hours": 4,
    "Next 1 day": 8,
    "Next 2 days": 16,
}

@dataclass
class AppPaths:
    repo_root: Path
    forecast_source: Path
    geojson: Path
    models_dir: Path


def resolve_paths(base_dir: Optional[Path] = None) -> AppPaths:
    app_dir = base_dir or Path(__file__).resolve().parent
    repo_root = app_dir.parent
    processed = repo_root / "data" / "processed"
    raw = repo_root / "data" / "data_raw"
    candidates = [
        processed / "supervised_hood_3h_multiclass.xlsx",
        processed / "supervised_hood_3h_multiclass.csv",
        repo_root / "supervised_hood_3h_multiclass.xlsx",
        repo_root / "supervised_hood_3h_multiclass.csv",
        processed / "dashboard_hood_3h_weather.csv",
    ]
    forecast_source = next((p for p in candidates if p.exists()), processed / "dashboard_hood_3h_weather.csv")
    return AppPaths(repo_root=repo_root, forecast_source=forecast_source, geojson=raw / "Neighbourhoods.geojson", models_dir=repo_root / "models")


def load_geojson(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def geo_lookup_table(geojson: dict) -> pd.DataFrame:
    rows = []
    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        rows.append({
            "HOOD_158_CODE": str(props.get("AREA_SHORT_CODE")),
            "hood_name": props.get("AREA_NAME", "Unknown"),
            "hood_desc": props.get("AREA_DESC", ""),
        })
    return pd.DataFrame(rows)


def load_history(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    if "time_3h" in df.columns:
        df["time_3h"] = pd.to_datetime(df["time_3h"])
    if "HOOD_158_CODE" in df.columns:
        df["HOOD_158_CODE"] = pd.to_numeric(df["HOOD_158_CODE"], errors="coerce").fillna(-1).astype(int)
    if "collisions" not in df.columns and "y_class" in df.columns:
        df["collisions"] = df["y_class"].map({0: 0, 1: 1, 2: 2}).fillna(0)
    return df


def load_best_bundle(models_dir: Path):
    preferred = [
        "stacking_3model_dashboard_bundle.pkl",
        "stacking_4model_dashboard_bundle.pkl",
        "13_stacking_oof_dashboard_bundle.pkl",
    ]
    candidates = []
    for name in preferred:
        p = models_dir / name
        if p.exists():
            candidates.append(p)
    candidates.extend(sorted(models_dir.glob("*dashboard_bundle*.pkl")))
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        try:
            obj = joblib.load(p)
            if isinstance(obj, dict) and "meta_model" in obj:
                return obj, p
        except Exception:
            pass
        try:
            with p.open("rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and "meta_model" in obj:
                return obj, p
        except Exception:
            pass
    return None, None


def derive_temporal_fields(ts: pd.Timestamp) -> Dict[str, float]:
    block_hour = int(ts.hour // 3 * 3)
    dow_num = int(ts.dayofweek)
    month_num = int(ts.month)
    is_weekend = int(dow_num >= 5)
    out = {
        "block_hour": block_hour,
        "dow_num": dow_num,
        "month_num": month_num,
        "is_weekend": is_weekend,
        "hour_sin": np.sin(2 * np.pi * block_hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * block_hour / 24.0),
        "dow_sin": np.sin(2 * np.pi * dow_num / 7.0),
        "dow_cos": np.cos(2 * np.pi * dow_num / 7.0),
        "month_sin": np.sin(2 * np.pi * month_num / 12.0),
        "month_cos": np.cos(2 * np.pi * month_num / 12.0),
    }
    return out


def build_latest_template(history: pd.DataFrame, geojson: Optional[dict] = None) -> pd.DataFrame:
    latest_idx = history.sort_values("time_3h").groupby("HOOD_158_CODE")["time_3h"].idxmax()
    latest = history.loc[latest_idx].sort_values("HOOD_158_CODE").reset_index(drop=True)

    if geojson is None:
        return latest

    geo_codes = geo_lookup_table(geojson)["HOOD_158_CODE"].astype(int).sort_values().tolist()
    present = set(latest["HOOD_158_CODE"].astype(int).tolist())
    missing = [c for c in geo_codes if c not in present]
    if not missing:
        return latest

    # global fallback row using medians/modes from latest available neighborhood rows
    fallback = {}
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
    numeric_fill, other_fill = {}, {}
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


def make_future_feature_frame(history: pd.DataFrame, forecast_start: pd.Timestamp, horizon_steps: int, user_inputs: Dict[str, float], feature_columns: Optional[List[str]] = None, geojson: Optional[dict] = None) -> pd.DataFrame:
    latest = build_latest_template(history, geojson=geojson)
    numeric_fill, other_fill = bundle_feature_fill_values(history, feature_columns or list(latest.columns))
    frames = []
    for step in range(1, horizon_steps + 1):
        ts = forecast_start + pd.Timedelta(hours=3 * step)
        frame = latest.copy()
        for k, v in derive_temporal_fields(ts).items():
            frame[k] = v
        for k, v in user_inputs.items():
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


def predict_with_bundle(bundle: dict, future_features: pd.DataFrame) -> pd.DataFrame:
    feature_columns = list(bundle.get("feature_columns", []))
    base_model_order = list(bundle.get("base_model_order", []))
    base_models = bundle.get("base_models", {})
    meta_model = bundle.get("meta_model")
    X = future_features[feature_columns].copy() if feature_columns else future_features.copy()
    meta_parts = []
    for model_name in base_model_order:
        model = base_models[model_name]
        meta_parts.append(_predict_base_model_proba(model, X))
    X_meta = np.hstack(meta_parts)
    final_proba = np.asarray(meta_model.predict_proba(X_meta), dtype=float)
    pred_class = final_proba.argmax(axis=1)
    out = future_features[["forecast_step", "forecast_time"]].copy()
    out["HOOD_158_CODE"] = (X["HOOD_158_CODE"] if "HOOD_158_CODE" in X.columns else future_features["HOOD_158_CODE"]).astype(str)
    out["pred_class"] = pred_class
    out["p0"] = final_proba[:, 0]
    out["p1"] = final_proba[:, 1]
    out["p2"] = final_proba[:, 2]
    out["risk_score"] = 0.5 * out["p1"] + 1.0 * out["p2"] + out["pred_class"] * 0.25
    return out


class DemoRiskModel:
    def __init__(self) -> None:
        self.feature_columns: List[str] = []
        self.model: Optional[Pipeline] = None
        self.class_labels = [0, 1, 2]

    def fit(self, df: pd.DataFrame) -> "DemoRiskModel":
        work = df.copy()
        if "y_class" not in work.columns:
            if "collisions" in work.columns:
                work["y_class"] = np.where(work["collisions"] >= 2, 2, np.where(work["collisions"] == 1, 1, 0))
            else:
                work["y_class"] = 0
        features = [c for c in ["HOOD_158_CODE", "block_hour", "dow_num", "month_num", "is_weekend", "temperature", "rain", "snow", "wind_speed", "relative_humidity"] if c in work.columns]
        self.feature_columns = features
        num_cols = [c for c in features if c != "HOOD_158_CODE"]
        cat_cols = [c for c in features if c == "HOOD_158_CODE"]
        preprocess = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ])
        clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        self.model = Pipeline([("prep", preprocess), ("clf", clf)])
        self.model.fit(work[features], work["y_class"])
        return self

    def predict_proba(self, future_features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict_proba(future_features[self.feature_columns]), dtype=float)


def predict_with_fallback(model: DemoRiskModel, future_features: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict_proba(future_features)
    pred_class = proba.argmax(axis=1)
    out = future_features[["forecast_step", "forecast_time", "HOOD_158_CODE"]].copy()
    out["HOOD_158_CODE"] = out["HOOD_158_CODE"].astype(str)
    out["pred_class"] = pred_class
    out["p0"] = proba[:, 0]
    out["p1"] = proba[:, 1]
    out["p2"] = proba[:, 2]
    out["risk_score"] = 0.5 * out["p1"] + 1.0 * out["p2"] + out["pred_class"] * 0.25
    return out


def aggregate_horizon_predictions(pred_df: pd.DataFrame, horizon_label: str | None = None) -> pd.DataFrame:
    agg = pred_df.groupby("HOOD_158_CODE", as_index=False).agg(
        pred_class=("pred_class", "max"),
        p0=("p0", "mean"),
        p1=("p1", "mean"),
        p2=("p2", "mean"),
        risk_score=("risk_score", "mean"),
        forecast_end=("forecast_time", "max"),
    )

    agg["risk_bucket"] = np.select(
        [agg["pred_class"] >= 2, agg["pred_class"] == 1],
        ["High", "Medium"],
        default="Low",
    )

    agg["HOOD_158_CODE"] = agg["HOOD_158_CODE"].astype(str)

    if horizon_label is not None:
        agg["horizon"] = horizon_label

    return agg


def merge_geo_labels(pred_agg: pd.DataFrame, geojson: dict) -> pd.DataFrame:
    lookup = geo_lookup_table(geojson)
    out = lookup.merge(pred_agg.copy(), on="HOOD_158_CODE", how="left")
    # Fill any still-missing predictions with low-risk defaults so every neighbourhood is shown
    out["pred_class"] = out["pred_class"].fillna(0).astype(int)
    out["p0"] = out["p0"].fillna(1.0)
    out["p1"] = out["p1"].fillna(0.0)
    out["p2"] = out["p2"].fillna(0.0)
    out["risk_score"] = out["risk_score"].fillna(0.0)
    out["risk_bucket"] = out["risk_bucket"].fillna("Low")
    return out


def top_hotspots(pred_agg: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    cols = ["HOOD_158_CODE", "hood_name", "risk_bucket", "pred_class", "risk_score", "p2"]
    return pred_agg.sort_values(["pred_class", "risk_score", "p2"], ascending=[False, False, False]).head(n)[cols].reset_index(drop=True)


def recent_history_for_hood(history: pd.DataFrame, hood_code: str, periods: int = 50) -> pd.DataFrame:
    hood_int = int(hood_code)
    return history.loc[history["HOOD_158_CODE"] == hood_int].sort_values("time_3h").tail(periods).copy()
