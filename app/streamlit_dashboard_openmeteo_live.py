
from __future__ import annotations

from pathlib import Path
import sys
from datetime import datetime, time

import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from dashboard_utils_openmeteo_live import (
    HORIZON_STEPS,
    DemoRiskModel,
    aggregate_horizon_predictions,
    geo_lookup_table,
    get_live_inputs_for_start,
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
        padding: 1.15rem 1.35rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #020617 0%, #0f172a 28%, #1d4ed8 62%, #0f766e 100%);
        color: #ffffff;
        box-shadow: 0 16px 36px rgba(2, 6, 23, 0.28);
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero h1 {
        margin: 0;
        font-size: 2.15rem;
        color: #ffffff !important;
        text-shadow: 0 2px 8px rgba(2,6,23,0.35);
    }
    .hero p {
        margin: 0.45rem 0 0;
        color: #f8fafc !important;
        opacity: 1;
    }
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

    st.markdown(
        """
        <div class="hero">
            <h1>Toronto Collision Risk Forecast Dashboard</h1>
            <p>
                Live-weather deployment demo for neighbourhood-level collision risk forecasting.
                Toronto weather is auto-fetched and fed into the trained model to produce hotspot forecasts for multiple horizons.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='data-src'><b>Forecast source:</b> {paths.forecast_source.name} &nbsp; | &nbsp; "
        f"<b>Neighbourhood file:</b> {paths.geojson.name}</div>",
        unsafe_allow_html=True,
    )

    now_floor = pd.Timestamp.now().floor("3h")

    with st.sidebar:
        st.header("Forecast Controls")
        forecast_date = st.date_input("Forecast date", value=now_floor.date(), help="Select the forecast start date.")
        forecast_time = st.time_input("Forecast time", value=now_floor.time(), step=10800, help="Use 3-hour blocks to match the trained model.")
        forecast_start_ts = combine_date_time(forecast_date, forecast_time)
        st.caption(f"Rounded forecast start: {forecast_start_ts}")

        horizons = st.multiselect(
            "Forecast horizons",
            options=list(HORIZON_STEPS.keys()),
            default=["Next 3 hours", "Next 12 hours", "Next 1 day"],
        )

        st.subheader("Weather source")
        weather_mode = st.radio(
            "Input mode",
            options=["Live Open-Meteo", "Manual override"],
            index=0,
            help="Use live Toronto weather automatically, or manually override the compact sensor inputs.",
        )

    live_meta = None
    live_df = None
    live_inputs = {
        "temperature": 8.0,
        "rain": 0.0,
        "snow": 0.0,
        "wind_speed": 15.0,
        "relative_humidity": 72.0,
        "visibility": 12.0,
    }

    if weather_mode == "Live Open-Meteo":
        try:
            live_inputs, live_meta, live_df = get_live_inputs_for_start(forecast_start_ts)
        except Exception as e:
            st.warning(f"Live weather fetch failed. Falling back to manual defaults. Details: {e}")
            weather_mode = "Manual override"

    with st.sidebar:
        st.subheader("Live sensor inputs")
        if weather_mode == "Live Open-Meteo":
            st.success("Live Toronto weather loaded from Open-Meteo.")
            st.caption("These values can still be adjusted for what-if analysis.")

        temperature = st.slider("Temperature (°C)", -20.0, 35.0, float(live_inputs["temperature"]), 0.5)
        rain = st.slider("Rain (mm)", 0.0, 30.0, float(live_inputs["rain"]), 0.1)
        snow = st.slider("Snow (cm/mm equiv.)", 0.0, 25.0, float(live_inputs["snow"]), 0.1)
        wind_speed = st.slider("Wind speed (km/h)", 0.0, 80.0, float(live_inputs["wind_speed"]), 0.5)
        humidity = st.slider("Relative humidity (%)", 20, 100, int(round(live_inputs["relative_humidity"])))
        visibility = st.slider("Visibility (km)", 0.0, 25.0, float(live_inputs["visibility"]), 0.1)

        st.markdown(
            "<div class='small-note'>Only a compact operational feature set is exposed. Time features are derived automatically and remaining model inputs are filled from recent history so the interface stays clean.</div>",
            unsafe_allow_html=True,
        )

        hood_lookup = geo_lookup_table(geojson)[["HOOD_158_CODE", "hood_name"]].copy()
        hood_lookup["HOOD_158_CODE"] = hood_lookup["HOOD_158_CODE"].astype(str)
        hood_lookup["label"] = hood_lookup["hood_name"] + " (" + hood_lookup["HOOD_158_CODE"] + ")"
        hood_lookup = hood_lookup.sort_values("hood_name").reset_index(drop=True)

        selected_label = st.selectbox("Neighbourhood to inspect", options=hood_lookup["label"].tolist(), index=0)
        selected_hood = hood_lookup.loc[hood_lookup["label"] == selected_label, "HOOD_158_CODE"].iloc[0]

        if model_assets["mode"] == "bundle":
            st.success(f"Loaded dashboard bundle: {model_assets['bundle_path'].name}")
        else:
            st.warning(f"No saved dashboard bundle could be loaded from {paths.models_dir}. Running with a fallback demo model trained from the available forecasting dataset.")

    if not horizons:
        st.info("Select at least one forecast horizon from the sidebar.")
        st.stop()

    user_inputs = {
        "temperature": float(temperature),
        "rain": float(rain),
        "snow": float(snow),
        "wind_speed": float(wind_speed),
        "relative_humidity": float(humidity),
        "visibility": float(visibility),
    }

    model_name = model_assets["bundle"].get("model_family", "saved_bundle") if model_assets["mode"] == "bundle" else "fallback_demo_model"
    source_note = "Open-Meteo live feed" if weather_mode == "Live Open-Meteo" else "Manual override"
    st.info(f"Model in use: {model_name} | Forecast source: {paths.forecast_source.name} | Weather input mode: {source_note}")
    if live_meta is not None:
        st.caption(f"Live weather coordinates: {live_meta['latitude']:.4f}, {live_meta['longitude']:.4f} | Timezone: {live_meta['timezone']}")

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
                top_display = top_df.rename(columns={
                    "HOOD_158_CODE": "Hood",
                    "hood_name": "Neighbourhood",
                    "pred_class": "Risk class",
                    "risk_score": "Risk score",
                    "p2": "P(class 2)",
                })
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
        st.markdown(f"### Recent historical collisions for neighbourhood {selected_hood}")
        if history_df.empty or "time_3h" not in history_df.columns or "collisions" not in history_df.columns:
            st.info("No recent history available for the selected neighbourhood.")
        else:
            hist_fig = px.line(history_df, x="time_3h", y="collisions", markers=True)
            hist_fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(hist_fig, use_container_width=True)

    if live_df is not None:
        st.divider()
        st.subheader("Live weather preview (Toronto) — next 24 hours")
        preview = live_df[[c for c in ["time", "temperature", "rain", "snow", "wind_speed", "relative_humidity", "visibility"] if c in live_df.columns]].head(24).copy()
        st.dataframe(preview, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
