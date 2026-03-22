from __future__ import annotations

from pathlib import Path
import sys
from datetime import datetime, time
import math
import re
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from pulp import (
        LpBinary,
        LpInteger,
        LpMaximize,
        LpProblem,
        LpStatus,
        LpVariable,
        PULP_CBC_CMD,
        lpSum,
        value,
    )
except Exception as e:
    raise ImportError(
        "PuLP is required for the optimization layer. Install it with: pip install pulp"
    ) from e

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))
PREFERRED_BUNDLE_FILES = [
    "stacking_4model_dashboard_bundle.pkl",
    "14_stacking_4model_dashboard_bundle.pkl",
]
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
    normalize_hood_code,
    predict_with_bundle,
    predict_with_fallback,
    recent_history_for_hood,
    resolve_paths,
    top_hotspots,
)

# ======================================================================
# 1) Division mapping and resources
# ======================================================================
DIVISION_NEIGHBORHOODS = {
    14: [
        {"Neighborhood Name": "South Parkdale", "Hood ID": 85},
        {"Neighborhood Name": "Trinity-Bellwoods", "Hood ID": 81},
        {"Neighborhood Name": "Wellington Place", "Hood ID": 164},
        {"Neighborhood Name": "Kensington-Chinatown", "Hood ID": 78},
        {"Neighborhood Name": "Annex", "Hood ID": 95},
        {"Neighborhood Name": "University", "Hood ID": 79},
        {"Neighborhood Name": "West Queen West", "Hood ID": 162},
        {"Neighborhood Name": "Palmerston-Little Italy", "Hood ID": 80},
        {"Neighborhood Name": "Dufferin Grove", "Hood ID": 83},
        {"Neighborhood Name": "Little Portugal", "Hood ID": 84},
        {"Neighborhood Name": "Fort York-Liberty Village", "Hood ID": 163},
        {"Neighborhood Name": "Dovercourt Village", "Hood ID": 172},
        {"Neighborhood Name": "Harbourfront-CityPlace", "Hood ID": 165},
    ],
    32: [
        {"Neighborhood Name": "York University Heights", "Hood ID": 27},
        {"Neighborhood Name": "Lansing-Westgate", "Hood ID": 38},
        {"Neighborhood Name": "Yorkdale-Glen Park", "Hood ID": 31},
        {"Neighborhood Name": "St.Andrew-Windfields", "Hood ID": 40},
        {"Neighborhood Name": "Westminster-Branson", "Hood ID": 35},
        {"Neighborhood Name": "Clanton Park", "Hood ID": 33},
        {"Neighborhood Name": "Newtonbrook West", "Hood ID": 36},
        {"Neighborhood Name": "Englemount-Lawrence", "Hood ID": 32},
        {"Neighborhood Name": "Bathurst Manor", "Hood ID": 34},
        {"Neighborhood Name": "Yonge-Doris", "Hood ID": 151},
        {"Neighborhood Name": "Willowdale West", "Hood ID": 37},
        {"Neighborhood Name": "Bedford Park-Nortown", "Hood ID": 39},
        {"Neighborhood Name": "Bridle Path-Sunnybrook-York Mills", "Hood ID": 41},
        {"Neighborhood Name": "Lawrence Park North", "Hood ID": 105},
        {"Neighborhood Name": "Newtonbrook East", "Hood ID": 50},
        {"Neighborhood Name": "East Willowdale", "Hood ID": 152},
        {"Neighborhood Name": "Avondale", "Hood ID": 153},
        {"Neighborhood Name": "Downsview", "Hood ID": 155},
    ],
    43: [
        {"Neighborhood Name": "Scarborough Village", "Hood ID": 139},
        {"Neighborhood Name": "Centennial Scarborough", "Hood ID": 133},
        {"Neighborhood Name": "Cliffcrest", "Hood ID": 123},
        {"Neighborhood Name": "Guildwood", "Hood ID": 140},
        {"Neighborhood Name": "West Hill", "Hood ID": 136},
        {"Neighborhood Name": "Highland Creek", "Hood ID": 134},
        {"Neighborhood Name": "Eglinton East", "Hood ID": 138},
        {"Neighborhood Name": "Bendale-Glen Andrew", "Hood ID": 156},
        {"Neighborhood Name": "West Rouge", "Hood ID": 143},
        {"Neighborhood Name": "Woburn North", "Hood ID": 142},
        {"Neighborhood Name": "Morningside", "Hood ID": 135},
        {"Neighborhood Name": "Golfdale-Cedarbrae-Woburn", "Hood ID": 141},
        {"Neighborhood Name": "Morningside Heights", "Hood ID": 144},
        {"Neighborhood Name": "Bendale South", "Hood ID": 157},
    ],
    55: [
        {"Neighborhood Name": "The Beaches", "Hood ID": 63},
        {"Neighborhood Name": "Danforth East Yor", "Hood ID": 59},
        {"Neighborhood Name": "Danforth", "Hood ID": 66},
        {"Neighborhood Name": "South Riverdale", "Hood ID": 70},
        {"Neighborhood Name": "Taylor-Massey", "Hood ID": 61},
        {"Neighborhood Name": "Flemingdon Park", "Hood ID": 44},
        {"Neighborhood Name": "Broadview North", "Hood ID": 57},
        {"Neighborhood Name": "North Riverdale", "Hood ID": 68},
        {"Neighborhood Name": "Greenwood-Coxwell", "Hood ID": 65},
        {"Neighborhood Name": "Victoria Village", "Hood ID": 43},
        {"Neighborhood Name": "O'Connor-Parkview", "Hood ID": 54},
        {"Neighborhood Name": "Old East York", "Hood ID": 58},
        {"Neighborhood Name": "Blake-Jones", "Hood ID": 69},
        {"Neighborhood Name": "East End-Danforth", "Hood ID": 62},
        {"Neighborhood Name": "Playter Estates-Danforth", "Hood ID": 67},
        {"Neighborhood Name": "Woodbine-Lumsden", "Hood ID": 60},
        {"Neighborhood Name": "Woodbine Corridor", "Hood ID": 64},
        {"Neighborhood Name": "Leaside-Bennington", "Hood ID": 56},
    ],
    11: [
        {"Neighborhood Name": "South Parkdale", "Hood ID": 85},
        {"Neighborhood Name": "Junction Area", "Hood ID": 90},
        {"Neighborhood Name": "Runnymede-Bloor West Village", "Hood ID": 89},
        {"Neighborhood Name": "Roncesvalles", "Hood ID": 86},
        {"Neighborhood Name": "Junction-Wallace Emerson", "Hood ID": 171},
        {"Neighborhood Name": "High Park North", "Hood ID": 88},
        {"Neighborhood Name": "High Park-Swansea", "Hood ID": 87},
        {"Neighborhood Name": "Weston-Pelham Park", "Hood ID": 91},
        {"Neighborhood Name": "Lambton Baby Point", "Hood ID": 114},
        {"Neighborhood Name": "Rockcliffe-Smythe", "Hood ID": 111},
        {"Neighborhood Name": "Dufferin Grove", "Hood ID": 83},
        {"Neighborhood Name": "Little Portugal", "Hood ID": 84},
    ],
    22: [
        {"Neighborhood Name": "Stonegate-Queensway", "Hood ID": 16},
        {"Neighborhood Name": "Islington City Centre-West", "Hood ID": 158},
        {"Neighborhood Name": "Princess-Rosethorn", "Hood ID": 10},
        {"Neighborhood Name": "Etobicoke West Mall", "Hood ID": 13},
        {"Neighborhood Name": "Kingsway South", "Hood ID": 15},
        {"Neighborhood Name": "Humber Heights-Westmount", "Hood ID": 8},
        {"Neighborhood Name": "Edenbridge-Humber Valley", "Hood ID": 9},
        {"Neighborhood Name": "Eringate-Centennial-West Deane", "Hood ID": 11},
        {"Neighborhood Name": "Alderwood", "Hood ID": 20},
        {"Neighborhood Name": "New Toronto", "Hood ID": 18},
        {"Neighborhood Name": "Long Branch", "Hood ID": 19},
        {"Neighborhood Name": "Markland Wood", "Hood ID": 12},
        {"Neighborhood Name": "Mimico-Queensway", "Hood ID": 160},
        {"Neighborhood Name": "Etobicoke City Centre", "Hood ID": 159},
        {"Neighborhood Name": "Humber Bay Shores", "Hood ID": 161},
    ],
    33: [
        {"Neighborhood Name": "St.Andrew-Windfields", "Hood ID": 40},
        {"Neighborhood Name": "Victoria Village", "Hood ID": 43},
        {"Neighborhood Name": "Bayview Woods-Steeles", "Hood ID": 49},
        {"Neighborhood Name": "Henry Farm", "Hood ID": 53},
        {"Neighborhood Name": "Hillcrest Village", "Hood ID": 48},
        {"Neighborhood Name": "Banbury-Don Mills", "Hood ID": 42},
        {"Neighborhood Name": "Parkwoods-O'Connor Hills", "Hood ID": 149},
        {"Neighborhood Name": "Bayview Village", "Hood ID": 52},
        {"Neighborhood Name": "Bridle Path-Sunnybrook-York Mills", "Hood ID": 41},
        {"Neighborhood Name": "Don Valley Village", "Hood ID": 47},
        {"Neighborhood Name": "Pleasant View", "Hood ID": 46},
        {"Neighborhood Name": "Leaside-Bennington", "Hood ID": 56},
        {"Neighborhood Name": "Fenside-Parkwoods", "Hood ID": 150},
        {"Neighborhood Name": "Community Safety Hub", "Hood ID": None},
    ],
    51: [
        {"Neighborhood Name": "South Riverdale", "Hood ID": 70},
        {"Neighborhood Name": "Church-Wellesley", "Hood ID": 167},
        {"Neighborhood Name": "Wellington Place", "Hood ID": 164},
        {"Neighborhood Name": "North St.James Town", "Hood ID": 74},
        {"Neighborhood Name": "Regent Park", "Hood ID": 72},
        {"Neighborhood Name": "Cabbagetown-South St.James Town", "Hood ID": 71},
        {"Neighborhood Name": "Moss Park", "Hood ID": 73},
        {"Neighborhood Name": "St.Lawrence-East Bayfront-The Islands", "Hood ID": 166},
        {"Neighborhood Name": "Downtown Yonge East", "Hood ID": 168},
    ],
    12: [
        {"Neighborhood Name": "Rustic", "Hood ID": 28},
        {"Neighborhood Name": "Junction Area", "Hood ID": 90},
        {"Neighborhood Name": "Keelesdale-Eglinton West", "Hood ID": 110},
        {"Neighborhood Name": "Mount Dennis", "Hood ID": 115},
        {"Neighborhood Name": "Beechborough-Greenbrook", "Hood ID": 112},
        {"Neighborhood Name": "Weston-Pelham Park", "Hood ID": 91},
        {"Neighborhood Name": "Pelmo Park-Humberlea", "Hood ID": 23},
        {"Neighborhood Name": "Rockcliffe-Smythe", "Hood ID": 111},
        {"Neighborhood Name": "Weston", "Hood ID": 113},
        {"Neighborhood Name": "Brookhaven-Amesbury", "Hood ID": 30},
        {"Neighborhood Name": "Maple Leaf", "Hood ID": 29},
    ],
    23: [
        {"Neighborhood Name": "Thistletown-Beaumond Heights", "Hood ID": 3},
        {"Neighborhood Name": "Humbermede", "Hood ID": 22},
        {"Neighborhood Name": "West Humber-Clairville", "Hood ID": 1},
        {"Neighborhood Name": "Kingsview Village-The Westway", "Hood ID": 6},
        {"Neighborhood Name": "Humber Heights-Westmount", "Hood ID": 8},
        {"Neighborhood Name": "Edenbridge-Humber Valley", "Hood ID": 9},
        {"Neighborhood Name": "Elms-Old Rexdale", "Hood ID": 5},
        {"Neighborhood Name": "Eringate-Centennial-West Deane", "Hood ID": 11},
        {"Neighborhood Name": "Mount Olive-Silverstone-Jamestown", "Hood ID": 2},
        {"Neighborhood Name": "Rexdale-Kipling", "Hood ID": 4},
        {"Neighborhood Name": "Willowridge-Martingrove-Richview", "Hood ID": 7},
    ],
    41: [
        {"Neighborhood Name": "Clairlea-Birchmount", "Hood ID": 120},
        {"Neighborhood Name": "Cliffcrest", "Hood ID": 123},
        {"Neighborhood Name": "Ionview", "Hood ID": 125},
        {"Neighborhood Name": "Kennedy Park", "Hood ID": 124},
        {"Neighborhood Name": "Dorset Park", "Hood ID": 126},
        {"Neighborhood Name": "Oakridge", "Hood ID": 121},
        {"Neighborhood Name": "Wexford/Maryvale", "Hood ID": 119},
        {"Neighborhood Name": "Eglinton East", "Hood ID": 138},
        {"Neighborhood Name": "Bendale-Glen Andrew", "Hood ID": 156},
        {"Neighborhood Name": "Birchcliffe-Cliffside", "Hood ID": 122},
        {"Neighborhood Name": "Bendale South", "Hood ID": 157},
    ],
    52: [
        {"Neighborhood Name": "Wellington Place", "Hood ID": 164},
        {"Neighborhood Name": "Kensington-Chinatown", "Hood ID": 78},
        {"Neighborhood Name": "University", "Hood ID": 79},
        {"Neighborhood Name": "Bay-Cloverhill", "Hood ID": 169},
        {"Neighborhood Name": "Yonge-Bay Corridor", "Hood ID": 170},
        {"Neighborhood Name": "Harbourfront-CityPlace", "Hood ID": 165},
        {"Neighborhood Name": "St.Lawrence-East Bayfront-The Islands", "Hood ID": 166},
    ],
    13: [
        {"Neighborhood Name": "Yorkdale-Glen Park", "Hood ID": 31},
        {"Neighborhood Name": "Humewood-Cedarvale", "Hood ID": 106},
        {"Neighborhood Name": "Corso Italia-Davenport", "Hood ID": 92},
        {"Neighborhood Name": "Forest Hill North", "Hood ID": 102},
        {"Neighborhood Name": "Casa Loma", "Hood ID": 96},
        {"Neighborhood Name": "Forest Hill South", "Hood ID": 101},
        {"Neighborhood Name": "Caledonia-Fairbank", "Hood ID": 109},
        {"Neighborhood Name": "Junction-Wallace Emerson", "Hood ID": 171},
        {"Neighborhood Name": "Oakwood Village", "Hood ID": 107},
        {"Neighborhood Name": "Englemount-Lawrence", "Hood ID": 32},
        {"Neighborhood Name": "Wychwood", "Hood ID": 94},
        {"Neighborhood Name": "Briar Hill-Belgravia", "Hood ID": 108},
    ],
    31: [
        {"Neighborhood Name": "York University Heights", "Hood ID": 27},
        {"Neighborhood Name": "Humber Summit", "Hood ID": 21},
        {"Neighborhood Name": "Humbermede", "Hood ID": 22},
        {"Neighborhood Name": "Glenfield-Jane Heights", "Hood ID": 25},
        {"Neighborhood Name": "Oakdale-Beverley Heights", "Hood ID": 154},
        {"Neighborhood Name": "Black Creek", "Hood ID": 24},
        {"Neighborhood Name": "Pelmo Park-Humberlea", "Hood ID": 23},
        {"Neighborhood Name": "Downsview", "Hood ID": 155},
    ],
    42: [
        {"Neighborhood Name": "Tam O'Shanter-Sullivan", "Hood ID": 118},
        {"Neighborhood Name": "Centennial Scarborough", "Hood ID": 133},
        {"Neighborhood Name": "Agincourt North", "Hood ID": 129},
        {"Neighborhood Name": "Agincourt South-Malvern West", "Hood ID": 128},
        {"Neighborhood Name": "L'Amoreaux West", "Hood ID": 147},
        {"Neighborhood Name": "West Rouge", "Hood ID": 143},
        {"Neighborhood Name": "Malvern West", "Hood ID": 145},
        {"Neighborhood Name": "Steeles", "Hood ID": 116},
        {"Neighborhood Name": "Milliken", "Hood ID": 130},
        {"Neighborhood Name": "Malvern East", "Hood ID": 146},
        {"Neighborhood Name": "East L'Amoreaux", "Hood ID": 148},
    ],
    53: [
        {"Neighborhood Name": "Yonge-St.Clair", "Hood ID": 97},
        {"Neighborhood Name": "Thorncliffe Park", "Hood ID": 55},
        {"Neighborhood Name": "Broadview North", "Hood ID": 57},
        {"Neighborhood Name": "Forest Hill North", "Hood ID": 102},
        {"Neighborhood Name": "Casa Loma", "Hood ID": 96},
        {"Neighborhood Name": "Forest Hill South", "Hood ID": 101},
        {"Neighborhood Name": "Annex", "Hood ID": 95},
        {"Neighborhood Name": "Rosedale-Moore Park", "Hood ID": 98},
        {"Neighborhood Name": "Mount Pleasant East", "Hood ID": 99},
        {"Neighborhood Name": "North Toronto", "Hood ID": 173},
        {"Neighborhood Name": "Bedford Park-Nortown", "Hood ID": 39},
        {"Neighborhood Name": "Bridle Path-Sunnybrook-York Mills", "Hood ID": 41},
        {"Neighborhood Name": "Lawrence Park South", "Hood ID": 103},
        {"Neighborhood Name": "Yonge-Eglinton", "Hood ID": 100},
        {"Neighborhood Name": "Leaside-Bennington", "Hood ID": 56},
        {"Neighborhood Name": "South Eglinton-Davisville", "Hood ID": 174},
    ],
}

DIVISION_RESOURCE_DATA = [
    {"Division": 11, "Neighborhood Count": 12, "Police Officers": 286, "Marked Vehicles": 42, "Bicycles": 23, "Motorcycles": 2},
    {"Division": 12, "Neighborhood Count": 11, "Police Officers": 303, "Marked Vehicles": 44, "Bicycles": 24, "Motorcycles": 2},
    {"Division": 13, "Neighborhood Count": 12, "Police Officers": 303, "Marked Vehicles": 44, "Bicycles": 24, "Motorcycles": 2},
    {"Division": 14, "Neighborhood Count": 13, "Police Officers": 303, "Marked Vehicles": 44, "Bicycles": 24, "Motorcycles": 2},
    {"Division": 22, "Neighborhood Count": 15, "Police Officers": 454, "Marked Vehicles": 67, "Bicycles": 36, "Motorcycles": 3},
    {"Division": 23, "Neighborhood Count": 11, "Police Officers": 302, "Marked Vehicles": 44, "Bicycles": 24, "Motorcycles": 2},
    {"Division": 31, "Neighborhood Count": 8,  "Police Officers": 202, "Marked Vehicles": 30, "Bicycles": 16, "Motorcycles": 1},
    {"Division": 32, "Neighborhood Count": 18, "Police Officers": 487, "Marked Vehicles": 72, "Bicycles": 38, "Motorcycles": 3},
    {"Division": 33, "Neighborhood Count": 14, "Police Officers": 403, "Marked Vehicles": 59, "Bicycles": 32, "Motorcycles": 2},
    {"Division": 41, "Neighborhood Count": 11, "Police Officers": 303, "Marked Vehicles": 44, "Bicycles": 24, "Motorcycles": 2},
    {"Division": 42, "Neighborhood Count": 11, "Police Officers": 336, "Marked Vehicles": 49, "Bicycles": 27, "Motorcycles": 2},
    {"Division": 43, "Neighborhood Count": 14, "Police Officers": 370, "Marked Vehicles": 54, "Bicycles": 29, "Motorcycles": 2},
    {"Division": 51, "Neighborhood Count": 9,  "Police Officers": 252, "Marked Vehicles": 37, "Bicycles": 20, "Motorcycles": 3},
    {"Division": 53, "Neighborhood Count": 16, "Police Officers": 403, "Marked Vehicles": 59, "Bicycles": 32, "Motorcycles": 3},
    {"Division": 55, "Neighborhood Count": 18, "Police Officers": 538, "Marked Vehicles": 79, "Bicycles": 42, "Motorcycles": 3},
    {"Division": 52, "Neighborhood Count": 7,  "Police Officers": 151, "Marked Vehicles": 22, "Bicycles": 12, "Motorcycles": 1},
]

CLASS_TO_WEIGHT = {0: 1.0, 1: 3.0, 2: 6.0}


def normalize_name(text: object) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


@st.cache_data(show_spinner=False)
def build_division_mapping_df() -> pd.DataFrame:
    rows = []
    for div, hoods in DIVISION_NEIGHBORHOODS.items():
        for hood in hoods:
            rows.append(
                {
                    "Division": int(div),
                    "Neighborhood Name": hood["Neighborhood Name"],
                    "Hood ID": hood["Hood ID"],
                    "Neighborhood Name_norm": normalize_name(hood["Neighborhood Name"]),
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_resource_df() -> pd.DataFrame:
    return pd.DataFrame(DIVISION_RESOURCE_DATA).copy()


# ======================================================================
# 2) Dynamic MILP helper
# ======================================================================
def run_dynamic_milp(
    agg_df: pd.DataFrame,
    officer_share: float = 0.30,
    vehicle_duty_share: float = 0.80,
    alpha: float = 1.0,
    beta: float = 0.7,
    gamma: float = 0.4,
    delta: float = 0.5,
    max_cars_per_neighborhood: int = 4,
    max_bikes_per_neighborhood: int = 2,
    max_motorcycles_per_neighborhood: int = 1,
    max_total_officers_per_neighborhood: int = 20,
    min_active_share: float = 0.25,
    enforce_spread: bool = True,
):
    mapping_df = build_division_mapping_df()
    resource_df = build_resource_df()

    scenario_df = agg_df[["HOOD_158_CODE", "hood_name", "pred_class"]].copy()
    scenario_df["Hood ID"] = pd.to_numeric(scenario_df["HOOD_158_CODE"], errors="coerce")
    scenario_df["Neighborhood Name_norm"] = scenario_df["hood_name"].map(normalize_name)
    scenario_df["Risk Class"] = pd.to_numeric(scenario_df["pred_class"], errors="coerce").fillna(0).astype(int)
    scenario_df["Risk Weight"] = scenario_df["Risk Class"].map(CLASS_TO_WEIGHT).fillna(1.0)

    merged = mapping_df.merge(
        scenario_df[["Hood ID", "Neighborhood Name_norm", "Risk Class", "Risk Weight"]].drop_duplicates(subset=["Hood ID"]),
        on="Hood ID",
        how="left",
    )

    missing = merged["Risk Weight"].isna()
    if missing.any():
        fill_name = mapping_df.loc[missing, ["Division", "Neighborhood Name", "Hood ID", "Neighborhood Name_norm"]].merge(
            scenario_df[["Neighborhood Name_norm", "Risk Class", "Risk Weight"]].drop_duplicates(subset=["Neighborhood Name_norm"]),
            on="Neighborhood Name_norm",
            how="left",
        )
        merged.loc[missing, "Risk Class"] = fill_name["Risk Class"].values
        merged.loc[missing, "Risk Weight"] = fill_name["Risk Weight"].values

    merged["Risk Class"] = merged["Risk Class"].fillna(0).astype(int)
    merged["Risk Weight"] = merged["Risk Weight"].fillna(1.0)
    merged = merged.merge(resource_df, on="Division", how="left")

    D = sorted(resource_df["Division"].tolist())
    N_by_D = {
        d: [(int(r["Division"]), str(r["Neighborhood Name"])) for _, r in merged[merged["Division"] == d].iterrows()]
        for d in D
    }
    neigh_keys = [k for items in N_by_D.values() for k in items]

    risk_weight = {}
    hood_lookup = {}
    risk_class = {}
    for _, r in merged.iterrows():
        key = (int(r["Division"]), str(r["Neighborhood Name"]))
        risk_weight[key] = float(r["Risk Weight"])
        risk_class[key] = int(r["Risk Class"])
        hood_lookup[key] = r["Hood ID"]

    officers_available = {
        int(r["Division"]): max(1, int(math.floor(float(r["Police Officers"]) * officer_share)))
        for _, r in resource_df.iterrows()
    }
    cars_available = {
        int(r["Division"]): max(0, int(math.floor(float(r["Marked Vehicles"]) * vehicle_duty_share)))
        for _, r in resource_df.iterrows()
    }
    bikes_available = {
        int(r["Division"]): max(0, int(math.floor(float(r["Bicycles"]) * vehicle_duty_share)))
        for _, r in resource_df.iterrows()
    }
    motorcycles_available = {
        int(r["Division"]): max(0, int(math.floor(float(r["Motorcycles"]) * vehicle_duty_share)))
        for _, r in resource_df.iterrows()
    }
    min_active_by_div = {
        d: max(1, int(math.ceil(min_active_share * len(N_by_D[d])))) if len(N_by_D[d]) else 0
        for d in D
    }

    model = LpProblem("Dynamic_Dashboard_MILP", LpMaximize)

    x_car = LpVariable.dicts("x_car", neigh_keys, lowBound=0, cat=LpInteger)
    x_bike = LpVariable.dicts("x_bike", neigh_keys, lowBound=0, cat=LpInteger)
    x_moto = LpVariable.dicts("x_moto", neigh_keys, lowBound=0, cat=LpInteger)
    p_car = LpVariable.dicts("p_car", neigh_keys, lowBound=0, cat=LpInteger)
    p_bike = LpVariable.dicts("p_bike", neigh_keys, lowBound=0, cat=LpInteger)
    p_moto = LpVariable.dicts("p_moto", neigh_keys, lowBound=0, cat=LpInteger)
    y = LpVariable.dicts("y_active", neigh_keys, lowBound=0, upBound=1, cat=LpBinary)

    model += lpSum(
        risk_weight[k]
        * (
            alpha * (p_car[k] + p_bike[k] + p_moto[k])
            + beta * x_car[k]
            + gamma * x_bike[k]
            + delta * x_moto[k]
        )
        for k in neigh_keys
    )

    for d in D:
        keys = N_by_D[d]
        model += lpSum(p_car[k] + p_bike[k] + p_moto[k] for k in keys) <= officers_available[d]
        model += lpSum(x_car[k] for k in keys) <= cars_available[d]
        model += lpSum(x_bike[k] for k in keys) <= bikes_available[d]
        model += lpSum(x_moto[k] for k in keys) <= motorcycles_available[d]
        if enforce_spread and keys:
            model += lpSum(y[k] for k in keys) >= min_active_by_div[d]

    for k in neigh_keys:
        model += p_car[k] >= x_car[k]
        model += p_car[k] <= 5 * x_car[k]
        model += p_bike[k] == x_bike[k]
        model += p_moto[k] >= x_moto[k]
        model += p_moto[k] <= 2 * x_moto[k]

        model += x_car[k] <= max_cars_per_neighborhood * y[k]
        model += x_bike[k] <= max_bikes_per_neighborhood * y[k]
        model += x_moto[k] <= max_motorcycles_per_neighborhood * y[k]
        model += p_car[k] + p_bike[k] + p_moto[k] <= max_total_officers_per_neighborhood * y[k]
        model += p_car[k] + p_bike[k] + p_moto[k] >= y[k]

    solver = PULP_CBC_CMD(msg=False)
    model.solve(solver)

    rows = []
    for k in neigh_keys:
        d, neighborhood = k
        rows.append(
            {
                "Division": d,
                "Neighborhood Name": neighborhood,
                "Hood ID": hood_lookup.get(k),
                "Risk Class": risk_class[k],
                "Risk Weight": risk_weight[k],
                "Active": int(round((y[k].value() or 0))),
                "Allocated Marked Vehicles": int(round((x_car[k].value() or 0))),
                "Allocated Bicycles": int(round((x_bike[k].value() or 0))),
                "Allocated Motorcycles": int(round((x_moto[k].value() or 0))),
                "Allocated Officers in Cars": int(round((p_car[k].value() or 0))),
                "Allocated Officers on Bicycles": int(round((p_bike[k].value() or 0))),
                "Allocated Officers on Motorcycles": int(round((p_moto[k].value() or 0))),
            }
        )

    allocation_df = pd.DataFrame(rows)
    allocation_df["Allocated Total Officers"] = (
        allocation_df["Allocated Officers in Cars"]
        + allocation_df["Allocated Officers on Bicycles"]
        + allocation_df["Allocated Officers on Motorcycles"]
    )
    allocation_df["Allocated Total Patrol Units"] = (
        allocation_df["Allocated Marked Vehicles"]
        + allocation_df["Allocated Bicycles"]
        + allocation_df["Allocated Motorcycles"]
    )

    division_summary = (
        allocation_df.groupby("Division", as_index=False)
        .agg(
            neighborhoods=("Neighborhood Name", "count"),
            class0=("Risk Class", lambda s: int((s == 0).sum())),
            class1=("Risk Class", lambda s: int((s == 1).sum())),
            class2=("Risk Class", lambda s: int((s == 2).sum())),
            assigned_officers=("Allocated Total Officers", "sum"),
            assigned_cars=("Allocated Marked Vehicles", "sum"),
            assigned_bikes=("Allocated Bicycles", "sum"),
            assigned_motorcycles=("Allocated Motorcycles", "sum"),
        )
        .merge(resource_df, on="Division", how="left")
    )

    division_summary["Officers Available This Shift"] = division_summary["Police Officers"].mul(officer_share).apply(math.floor).clip(lower=1)
    division_summary["Cars Available This Shift"] = division_summary["Marked Vehicles"].mul(vehicle_duty_share).apply(math.floor)
    division_summary["Bikes Available This Shift"] = division_summary["Bicycles"].mul(vehicle_duty_share).apply(math.floor)
    division_summary["Motorcycles Available This Shift"] = division_summary["Motorcycles"].mul(vehicle_duty_share).apply(math.floor)

    division_summary["Officer Balance OK"] = division_summary["assigned_officers"] == division_summary["Officers Available This Shift"]
    division_summary["Car Balance OK"] = division_summary["assigned_cars"] <= division_summary["Cars Available This Shift"]
    division_summary["Bike Balance OK"] = division_summary["assigned_bikes"] <= division_summary["Bikes Available This Shift"]
    division_summary["Motorcycle Balance OK"] = division_summary["assigned_motorcycles"] <= division_summary["Motorcycles Available This Shift"]

    meta = {
        "solver_status": LpStatus[model.status],
        "objective_value": value(model.objective),
        "officer_share": officer_share,
        "vehicle_duty_share": vehicle_duty_share,
    }
    return allocation_df, division_summary, meta


# ======================================================================
# 3) Forecast dashboard UI
# ======================================================================
st.set_page_config(page_title="Toronto Collision Risk Forecast + MILP", page_icon="🚦", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        padding: 1.15rem 1.35rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 52%, #0f766e 100%);
        color: #ffffff;
        box-shadow: 0 16px 36px rgba(2, 6, 23, 0.22);
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero h1 {
        margin: 0;
        font-size: 2.05rem;
        color: #ffffff !important;
        text-shadow: 0 2px 8px rgba(2,6,23,0.28);
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
    signature = f"{paths.forecast_source}:{paths.forecast_source.stat().st_mtime_ns}:{paths.geojson}:{paths.geojson.stat().st_mtime_ns}"
    return paths, history, geojson, signature


@st.cache_resource(show_spinner=False)
def get_model_assets(history_signature: str):
    _ = history_signature
    paths = resolve_paths(APP_DIR)

    # 1) First try the exact Notebook 14 deployment bundle
    for fname in PREFERRED_BUNDLE_FILES:
        candidate = paths.models_dir / fname
        if candidate.exists():
            try:
                bundle = joblib.load(candidate)
                return {"mode": "bundle", "bundle": bundle, "bundle_path": candidate}
            except Exception:
                pass

    # 2) If not found, fall back to the app's generic bundle loader
    bundle, bundle_path = load_best_bundle(paths.models_dir)
    if bundle is not None:
        return {"mode": "bundle", "bundle": bundle, "bundle_path": bundle_path}

    # 3) Final safety fallback: demo model
    demo = DemoRiskModel().fit(load_history(paths.forecast_source))
    return {"mode": "demo", "bundle": demo, "bundle_path": None}

@st.cache_data(show_spinner=False, ttl=1800)
def get_live_weather_cached(forecast_start_iso: str):
    forecast_start = pd.Timestamp(forecast_start_iso)
    return get_live_inputs_for_start(forecast_start)


def combine_date_time(d: datetime.date, t: time) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(d, t)).floor("3h")


def main() -> None:
    paths, history, geojson, history_signature = get_static_assets()
    model_assets = get_model_assets(history_signature)

    st.markdown(
        """
        <div class="hero">
            <h1>Toronto Collision Risk Forecast + Dynamic MILP Allocation</h1>
            <p>
                Live-weather collision forecasting with immediate division-to-neighbourhood patrol allocation.
                Every new weather scenario updates both the forecast and the optimization output.
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
        forecast_date = st.date_input("Forecast date", value=now_floor.date())
        forecast_time = st.time_input("Forecast time", value=now_floor.time(), step=10800)
        forecast_start_ts = combine_date_time(forecast_date, forecast_time)
        st.caption(f"Rounded forecast start: {forecast_start_ts}")

        horizons = st.multiselect(
            "Forecast horizons",
            options=list(HORIZON_STEPS.keys()),
            default=["Next 3 hours", "Next 12 hours", "Next 1 day"],
        )

        st.subheader("Weather source")
        weather_mode = st.radio("Input mode", options=["Live Open-Meteo", "Manual override"], index=0)

        st.subheader("Optimization Controls")
        officer_share = st.slider("Officer availability share", 0.10, 1.00, 0.30, 0.05)
        vehicle_duty_share = st.slider("Vehicle duty share", 0.50, 1.00, 0.80, 0.05)
        enforce_spread = st.checkbox("Enforce spread across neighborhoods", value=True)
        min_active_share = st.slider("Minimum active neighborhood share", 0.10, 0.80, 0.25, 0.05)
        max_cars_per_neighborhood = st.slider("Max cars per neighborhood", 1, 8, 4)
        max_bikes_per_neighborhood = st.slider("Max bikes per neighborhood", 0, 4, 2)
        max_motorcycles_per_neighborhood = st.slider("Max motorcycles per neighborhood", 0, 3, 1)
        max_total_officers_per_neighborhood = st.slider("Max officers per neighborhood", 5, 40, 20)

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
    weather_fetch_error = None

    if weather_mode == "Live Open-Meteo":
        try:
            live_inputs, live_meta, live_df = get_live_weather_cached(forecast_start_ts.isoformat())
        except Exception as e:
            weather_fetch_error = str(e)
            weather_mode = "Manual override"

    with st.sidebar:
        st.subheader("Live sensor inputs")
        if weather_mode == "Live Open-Meteo":
            st.success("Live Toronto weather loaded from Open-Meteo.")
        elif weather_fetch_error:
            st.warning(f"Live weather fetch failed. Using manual defaults instead. Details: {weather_fetch_error}")

        temperature = st.slider("Temperature (°C)", -20.0, 35.0, float(live_inputs["temperature"]), 0.5)
        rain = st.slider("Rain (mm)", 0.0, 30.0, float(live_inputs["rain"]), 0.1)
        snow = st.slider("Snow (cm/mm equiv.)", 0.0, 25.0, float(live_inputs["snow"]), 0.1)
        wind_speed = st.slider("Wind speed (km/h)", 0.0, 80.0, float(live_inputs["wind_speed"]), 0.5)
        humidity = st.slider("Relative humidity (%)", 20, 100, int(round(live_inputs["relative_humidity"])))
        visibility = st.slider("Visibility (km)", 0.0, 25.0, float(live_inputs["visibility"]), 0.1)

        st.markdown(
            "<div class='small-note'>Forecast classes are generated live from the selected scenario, then passed directly into the MILP. No static class CSV is used.</div>",
            unsafe_allow_html=True,
        )

        hood_lookup = geo_lookup_table(geojson)[["HOOD_158_CODE", "hood_name"]].copy()
        hood_lookup["HOOD_158_CODE"] = hood_lookup["HOOD_158_CODE"].map(normalize_hood_code)
        hood_lookup["label"] = hood_lookup["hood_name"] + " (" + hood_lookup["HOOD_158_CODE"] + ")"
        hood_lookup = hood_lookup.sort_values("hood_name").reset_index(drop=True)

        selected_label = st.selectbox("Neighbourhood to inspect", options=hood_lookup["label"].tolist(), index=0)
        selected_hood = hood_lookup.loc[hood_lookup["label"] == selected_label, "HOOD_158_CODE"].iloc[0]

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

    if model_assets["mode"] == "bundle":
        model_name = model_assets["bundle"].get("model_family", "saved_bundle")
        bundle_file = Path(model_assets["bundle_path"]).name if model_assets["bundle_path"] else "unknown_bundle"
        model_name = f"{model_name} ({bundle_file})"
    else:
        model_name = "fallback_demo_model"
    source_note = "Open-Meteo live feed" if weather_mode == "Live Open-Meteo" else "Manual override"
    st.info(f"Model in use: {model_name} | Forecast source: {paths.forecast_source.name} | Weather input mode: {source_note}")

    all_horizon_outputs = {}
    feature_columns = list(model_assets["bundle"].get("feature_columns", [])) if model_assets["mode"] == "bundle" else None
    weather_forecast = live_df if weather_mode == "Live Open-Meteo" else None

    for horizon_label in horizons:
        steps = HORIZON_STEPS[horizon_label]
        future_features = make_future_feature_frame(
            history=history,
            forecast_start=forecast_start_ts,
            horizon_steps=steps,
            user_inputs=user_inputs,
            feature_columns=feature_columns,
            geojson=geojson,
            weather_forecast=weather_forecast,
        )

        if model_assets["mode"] == "bundle":
            pred_df = predict_with_bundle(model_assets["bundle"], future_features)
        else:
            pred_df = predict_with_fallback(model_assets["bundle"], future_features)

        agg_df = aggregate_horizon_predictions(pred_df, horizon_label)
        agg_df = merge_geo_labels(agg_df, geojson)
        allocation_df, division_summary, milp_meta = run_dynamic_milp(
            agg_df=agg_df,
            officer_share=officer_share,
            vehicle_duty_share=vehicle_duty_share,
            enforce_spread=enforce_spread,
            min_active_share=min_active_share,
            max_cars_per_neighborhood=max_cars_per_neighborhood,
            max_bikes_per_neighborhood=max_bikes_per_neighborhood,
            max_motorcycles_per_neighborhood=max_motorcycles_per_neighborhood,
            max_total_officers_per_neighborhood=max_total_officers_per_neighborhood,
        )
        all_horizon_outputs[horizon_label] = {
            "step_level": pred_df,
            "aggregated": agg_df,
            "allocation": allocation_df,
            "division_summary": division_summary,
            "milp_meta": milp_meta,
        }

    tabs = st.tabs(horizons)
    color_map = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}

    for tab, horizon_label in zip(tabs, horizons):
        with tab:
            agg_df = all_horizon_outputs[horizon_label]["aggregated"].copy()
            allocation_df = all_horizon_outputs[horizon_label]["allocation"].copy()
            division_summary = all_horizon_outputs[horizon_label]["division_summary"].copy()
            milp_meta = all_horizon_outputs[horizon_label]["milp_meta"]

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
                    "risk_score": ":.2f",
                    "p2": ":.2f",
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
                        "forecast_end": "Forecast end",
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
            m1, m2, m3 = st.columns(3)
            m1.metric("MILP status", milp_meta["solver_status"])
            m2.metric("MILP objective", f"{milp_meta['objective_value']:.1f}" if milp_meta["objective_value"] is not None else "n/a")
            m3.metric("Officer share used", f"{100*milp_meta['officer_share']:.0f}%")

            st.subheader("Optimization output — neighbourhood allocation")
            st.dataframe(
                allocation_df.sort_values(["Division", "Risk Class", "Allocated Total Officers"], ascending=[True, False, False]),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                label=f"Download neighborhood allocation ({horizon_label})",
                data=allocation_df.to_csv(index=False).encode("utf-8"),
                file_name=f"milp_neighborhood_allocation_{horizon_label.lower().replace(' ', '_')}.csv",
                mime="text/csv",
            )

            st.subheader("Optimization output — division summary")
            st.dataframe(division_summary, use_container_width=True, hide_index=True)
            st.download_button(
                label=f"Download division summary ({horizon_label})",
                data=division_summary.to_csv(index=False).encode("utf-8"),
                file_name=f"milp_division_summary_{horizon_label.lower().replace(' ', '_')}.csv",
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
            row = agg_df.loc[agg_df["HOOD_158_CODE"] == normalize_hood_code(selected_hood)].copy()
            if len(row):
                hood_all.append(
                    row.iloc[0][["horizon", "hood_name", "risk_bucket", "pred_class", "risk_score", "p0", "p1", "p2", "forecast_end"]]
                )
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
