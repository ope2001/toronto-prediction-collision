from __future__ import annotations

from pathlib import Path
import sys
from datetime import datetime, time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from dashboard_utils import (
    HORIZON_STEPS,
    DemoRiskModel,
    aggregate_horizon_predictions,
    geo_lookup_table,
    load_best_bundle,
    load_geojson,
    load_history,
    make_future_feature_frame,
    merge_geo_labels,
    predict_with_bundle,
    predict_with_fallback,
    recent_history_for_hood,
    resolve_paths,
    top_hotspots,
)

st.set_page_config(page_title="Toronto Collision Risk Forecast", page_icon="🚦", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        padding: 1rem 1.25rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 45%, #14b8a6 100%);
        color: white;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1 {margin: 0; font-size: 2.1rem;}
    .hero p {margin: 0.35rem 0 0; opacity: 0.95;}
    .small-note {color: #475569; font-size: 0.9rem;}
    .data-src {padding: 0.5rem 0.75rem; border-radius: 12px; background:#eff6ff; border:1px solid #bfdbfe; margin-bottom:1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_static_assets():
    paths = resolve_paths(APP_DIR)
    history = load_history(paths.forecast_source)

    st.caption(f"Forecast source: {paths.forecast_source.name}")
    geojson = load_geojson(paths.geojson)
    return paths, history, geojson


@st.cache_resource(show_spinner=False)
def get_model_assets(history_hash: int):
    paths = resolve_paths(APP_DIR)
    bundle, bundle_path = load_best_bundle(paths.models_dir)
    if bundle is not None:
        return {"mode": "bundle", "bundle": bundle, "bundle_path": bundle_path}

    demo = DemoRiskModel().fit(load_history(paths.forecast_source))
    return {"mode": "demo", "bundle": demo, "bundle_path": None}


def combine_date_time(d: datetime.date, t: time) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(d, t)).floor("3h")


def main() -> None:
    paths, history, geojson = get_static_assets()
    model_assets = get_model_assets(len(history))

    st.markdown("""
    <style>
    .dashboard-title {
        background: #4F8EF7;   /* lighter blue */
        color: white;
        padding: 18px 24px;
        border-radius: 12px;
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
    '   <div class="dashboard-title">Toronto Traffic Collision Prediction Dashboard</div>',
        unsafe_allow_html=True
    )
    

    st.markdown(
        f"<div class='data-src'><b>Forecast source:</b> {paths.forecast_source.name} &nbsp; | &nbsp; "
        f"<b>Neighbourhood file:</b> {paths.geojson.name}</div>",
        unsafe_allow_html=True,
    
    )
    if model_assets["mode"] == "bundle":
        model_name = model_assets["bundle"].get("model_family", "saved_bundle")
    else:
        model_name = "fallback_demo_model"

    st.info(
        f"Model in use: {model_name} | Forecast source: {paths.forecast_source.name} | "
        f"Neighbourhood polygons: {paths.geojson.name}"
    )

    now_floor = pd.Timestamp.now().floor("3h")

    with st.sidebar:
        st.header("Forecast Controls")
        forecast_date = st.date_input(
            "Forecast date",
            value=now_floor.date(),
            help="Select the forecast start date.",
        )
        forecast_time = st.time_input(
            "Forecast time",
            value=now_floor.time(),
            step=10800,
            help="Use 3-hour blocks to match the trained model.",
        )
        forecast_start_ts = combine_date_time(forecast_date, forecast_time)
        st.caption(f"Rounded forecast start: {forecast_start_ts}")

        horizons = st.multiselect(
            "Forecast horizons",
            options=list(HORIZON_STEPS.keys()),
            default=["Next 3 hours", "Next 12 hours", "Next 1 day"],
        )

        st.subheader("Live sensor inputs")
        temperature = st.slider("Temperature (°C)", -20.0, 35.0, 8.0, 0.5)
        rain = st.slider("Rain (mm)", 0.0, 30.0, 0.0, 0.1)
        snow = st.slider("Snow (cm/mm equiv.)", 0.0, 25.0, 0.0, 0.1)
        wind_speed = st.slider("Wind speed (km/h)", 0.0, 80.0, 15.0, 0.5)
        humidity = st.slider("Relative humidity (%)", 20, 100, 72)
        visibility = st.slider("Visibility (km)", 0.0, 25.0, 12.0, 0.1)

        st.markdown("<div class='small-note'>Only a compact operational feature set is exposed to keep the dashboard clean. Other required features are derived from time or filled from recent historical medians.</div>", unsafe_allow_html=True)

        hood_lookup = geo_lookup_table(geojson)[["HOOD_158_CODE", "hood_name"]].copy()
        hood_lookup["HOOD_158_CODE"] = hood_lookup["HOOD_158_CODE"].astype(str)
        hood_lookup["label"] = hood_lookup["hood_name"] + " (" + hood_lookup["HOOD_158_CODE"] + ")"
        hood_lookup = hood_lookup.sort_values("hood_name").reset_index(drop=True)

        selected_label = st.selectbox(
            "Neighbourhood to inspect",
            options=hood_lookup["label"].tolist(),
            index=0,
        )
        selected_hood = hood_lookup.loc[hood_lookup["label"] == selected_label, "HOOD_158_CODE"].iloc[0]

        if model_assets["mode"] == "bundle":
            st.success(f"Loaded dashboard bundle: {model_assets['bundle_path'].name}")
        else:
            st.warning(f"No saved dashboard bundle could be loaded from {paths.models_dir}. Running with a fallback demo model trained from the available forecasting dataset.")

    if not horizons:
        st.info("Select at least one forecast horizon from the sidebar.")
        st.stop()

    user_inputs = {
        "temperature": temperature,
        "rain": rain,
        "snow": snow,
        "wind_speed": wind_speed,
        "relative_humidity": float(humidity),
        "visibility": visibility,
    }

    all_horizon_outputs = {}
    feature_columns = None
    if model_assets["mode"] == "bundle":
        feature_columns = list(model_assets["bundle"].get("feature_columns", []))

    for horizon_label in horizons:
        steps = HORIZON_STEPS[horizon_label]
        future_features = make_future_feature_frame(
            history=history,
            forecast_start=forecast_start_ts,
            horizon_steps=steps,
            user_inputs=user_inputs,
            feature_columns=feature_columns,
            geojson=geojson,
        )

        if model_assets["mode"] == "bundle":
            pred_df = predict_with_bundle(model_assets["bundle"], future_features)
        else:
            pred_df = predict_with_fallback(model_assets["bundle"], future_features)

        agg_df = aggregate_horizon_predictions(pred_df, horizon_label)
        agg_df = merge_geo_labels(agg_df, geojson)
        all_horizon_outputs[horizon_label] = {"step_level": pred_df, "aggregated": agg_df}

    tabs = st.tabs(horizons)
    color_map = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}

    for tab, horizon_label in zip(tabs, horizons):
        with tab:
            agg_df = all_horizon_outputs[horizon_label]["aggregated"].copy()
            top_df = top_hotspots(agg_df, n=10)

            high_count = int((agg_df["risk_bucket"] == "High").sum())
            med_count = int((agg_df["risk_bucket"] == "Medium").sum())
            avg_score = float(agg_df["risk_score"].mean())

            c1, c2, c3 = st.columns(3)
            c1.metric("High-risk neighbourhoods", high_count)
            c2.metric("Medium-risk neighbourhoods", med_count)
            c3.metric("Average risk score", f"{avg_score:.2f}")

            fig = px.choropleth_mapbox(
                agg_df,
                geojson=geojson,
                locations="HOOD_158_CODE",
                featureidkey="properties.AREA_SHORT_CODE",
                color="risk_bucket",
                category_orders={"risk_bucket": ["Low", "Medium", "High"]},
                color_discrete_map=color_map,
                hover_name="hood_name",
                hover_data={
                    "HOOD_158_CODE": True,
                    "risk_bucket": True,
                    "pred_class": True,
                    "risk_score": ':.2f',
                    "p2": ':.2f',
                    "forecast_end": True,
                },
                mapbox_style="carto-positron",
                zoom=9.1,
                center={"lat": 43.72, "lon": -79.38},
                opacity=0.72,
                height=620,
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="Risk level")

            left, right = st.columns([1.7, 1])
            with left:
                st.subheader(f"Hotspot map — {horizon_label}")
                st.plotly_chart(fig, use_container_width=True)
            with right:
                st.subheader("Top hotspots")
                top_display = top_df.rename(
                    columns={
                        "HOOD_158_CODE": "Hood",
                        "hood_name": "Neighbourhood",
                        "pred_class": "Risk class",
                        "risk_score": "Risk score",
                        "p2": "P(class 2)",
                    }
                )
                st.dataframe(top_display, use_container_width=True, hide_index=True)

                st.download_button(
                    label="Download hotspot table (CSV)",
                    data=top_display.to_csv(index=False).encode("utf-8"),
                    file_name=f"hotspots_{horizon_label.lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                )

    st.divider()
    st.subheader("Selected neighbourhood detail")
    history_df = recent_history_for_hood(history, selected_hood, periods=60)
    detail_cols = st.columns([1, 1.4])

    with detail_cols[0]:
        hood_all = []
        for horizon_label in horizons:
            agg_df = all_horizon_outputs[horizon_label]["aggregated"]
            row = agg_df.loc[agg_df["HOOD_158_CODE"] == str(selected_hood)].copy()
            if len(row):
                hood_all.append(row.iloc[0][["horizon", "hood_name", "risk_bucket", "pred_class", "risk_score", "p0", "p1", "p2"]])
        if hood_all:
            hood_detail = pd.DataFrame(hood_all)
            st.dataframe(hood_detail, use_container_width=True, hide_index=True)
        else:
            st.info("No forecast row found for the selected neighbourhood.")

    with detail_cols[1]:
        if len(history_df):
            hist_fig = px.line(
                history_df,
                x="time_3h",
                y="collisions",
                title=f"Recent historical collisions for neighbourhood {selected_hood}",
                markers=True,
            )
            hist_fig.update_layout(margin=dict(l=0, r=0, t=42, b=0))
            st.plotly_chart(hist_fig, use_container_width=True)
        else:
            st.info("No recent historical rows available for the selected neighbourhood.")

    st.divider()
    st.caption(
        "Deployment note: the dashboard exposes only a compact set of operational inputs. Time-based fields are derived automatically, and non-exposed features are filled from recent historical medians. If a saved dashboard bundle is available in models/, it is used; otherwise the app falls back to a lightweight demo model trained from the available forecasting dataset."
    )


if __name__ == "__main__":
    main()
