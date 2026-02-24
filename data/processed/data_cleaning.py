from pathlib import Path
import numpy as np
import pandas as pd

COLL_PATH = Path("raw_data") / "Traffic_Collisions 2023-2025.xlsx"
WEATHER_PATH = Path("raw_data") / "weather raw 23-25.xlsx"
OUT_DIR = Path("processed_data")


# =========================================================
# COLLISIONS (EVENT LEVEL): LOAD + CLEAN
# - Delete NSA
# - Delete N/R or NR rows for selected columns
# - Fix date/hour and build date_time_hour
# - Build HOOD_158_CODE
# - Fix coordinates (keep LAT_WGS84/LONG_WGS84, drop X/Y)
# - Encode YES/NO => 0/1 integers (uint8), no decimals
# =========================================================

coll = pd.read_excel(COLL_PATH, engine="openpyxl")
coll.columns = [str(c).strip() for c in coll.columns]

# --- 1) Delete NSA HOOD rows ---
hood_upper = coll["HOOD_158"].astype(str).str.strip().str.upper()
coll = coll[hood_upper.ne("NSA")].copy()

# --- 2) Delete N/R or NR rows in important columns ---
# (You requested to delete them. Add/remove columns here if needed.)
nr_cols = [
    "DIVISION",
    "INJURY_COLLISIONS", "FTR_COLLISIONS", "PD_COLLISIONS",
    "PEDESTRIAN", "BICYCLE", "AUTOMOBILE", "MOTORCYCLE", "PASSENGER"
]
nr_cols = [c for c in nr_cols if c in coll.columns]

for c in nr_cols:
    x = coll[c].astype(str).str.strip().str.upper()
    coll = coll[~x.isin(["N/R", "NR"])].copy()

# --- 3) Time: keep OCC_DATE and OCC_HOUR, build date_time_hour ---
coll["OCC_DATE"] = pd.to_datetime(coll["OCC_DATE"], errors="coerce").dt.floor("D")

coll["OCC_HOUR"] = pd.to_numeric(coll["OCC_HOUR"], errors="coerce")
coll.loc[~coll["OCC_HOUR"].between(0, 23), "OCC_HOUR"] = np.nan
coll["OCC_HOUR"] = coll["OCC_HOUR"].astype("Int64")

coll["date_time_hour"] = coll["OCC_DATE"] + pd.to_timedelta(
    coll["OCC_HOUR"].fillna(0).astype(int), unit="h"
)

# --- 4) HOOD code: make 3-digit string ---
hood_digits = coll["HOOD_158"].astype(str).str.extract(r"(\d+)", expand=False)
hood_num = pd.to_numeric(hood_digits, errors="coerce").astype("Int64")
coll["HOOD_158_CODE"] = hood_num.astype(str).str.zfill(3).replace("<NA>", np.nan)
coll = coll[coll["HOOD_158_CODE"].notna()].copy()

# --- 5) Coordinates: keep LAT_WGS84/LONG_WGS84; if X/Y exist, drop them ---
for c in ["LAT_WGS84", "LONG_WGS84", "X", "Y", "x", "y"]:
    if c in coll.columns:
        coll[c] = pd.to_numeric(coll[c], errors="coerce")

# If LAT/LONG are missing but X/Y exist, use Y as lat and X as lon
if ("LAT_WGS84" not in coll.columns or "LONG_WGS84" not in coll.columns) and ("Y" in coll.columns and "X" in coll.columns):
    coll["LAT_WGS84"] = coll["Y"]
    coll["LONG_WGS84"] = coll["X"]
if ("LAT_WGS84" not in coll.columns or "LONG_WGS84" not in coll.columns) and ("y" in coll.columns and "x" in coll.columns):
    coll["LAT_WGS84"] = coll["y"]
    coll["LONG_WGS84"] = coll["x"]

# Drop X/Y duplicates
coll = coll.drop(columns=[c for c in ["X","Y","x","y"] if c in coll.columns], errors="ignore")

# Remove invalid (0,0) coords
if "LAT_WGS84" in coll.columns and "LONG_WGS84" in coll.columns:
    coll = coll[~((coll["LAT_WGS84"].fillna(0) == 0) & (coll["LONG_WGS84"].fillna(0) == 0))].copy()

# --- 6) Encode YES/NO -> strict 0/1 integer (uint8), NO decimals ---
yes_no_cols = [
    "INJURY_COLLISIONS","FTR_COLLISIONS","PD_COLLISIONS",
    "PEDESTRIAN","BICYCLE","AUTOMOBILE","MOTORCYCLE","PASSENGER"
]
yes_no_cols = [c for c in yes_no_cols if c in coll.columns]

for c in yes_no_cols:
    x = coll[c].astype(str).str.strip().str.upper()
    coll[c + "_01"] = (x == "YES").astype("uint8")   # always integer 0/1

# --- 7) Deduplicate by EVENT_UNIQUE_ID (safety) ---
if "EVENT_UNIQUE_ID" in coll.columns:
    coll = coll.sort_values("date_time_hour").drop_duplicates("EVENT_UNIQUE_ID", keep="first").copy()

# --- 8) Keep only event-level columns needed for dashboard/drilldown ---
keep_event = [
    "EVENT_UNIQUE_ID",
    "OCC_DATE","OCC_HOUR","date_time_hour",
    "HOOD_158_CODE","NEIGHBOURHOOD_158",
    "DIVISION",
    "LAT_WGS84","LONG_WGS84",
    "FATALITIES",
    "INJURY_COLLISIONS_01","FTR_COLLISIONS_01","PD_COLLISIONS_01",
    "PEDESTRIAN_01","BICYCLE_01","AUTOMOBILE_01","MOTORCYCLE_01","PASSENGER_01",
]
keep_event = [c for c in keep_event if c in coll.columns]
coll_event_clean = coll[keep_event].copy()

# Save event-level cleaned dataset (dashboard drilldown)
coll_event_clean.to_csv(OUT_DIR / "collisions_clean_event.csv", index=False)

# =========================================================
# COLLISIONS: AGGREGATE TO HOOD × 3-HOUR BLOCK
# - Adds time features: block_hour, OCC_DOW, OCC_MONTH, is_weekend
# - Keeps type/user counts (injury/ftr/pd/ped/bike etc.)
# - (NO year)
# =========================================================

coll_event_clean["time_3h"] = pd.to_datetime(coll_event_clean["date_time_hour"], errors="coerce").dt.floor("3h")

coll_3h = (coll_event_clean.groupby(["HOOD_158_CODE","time_3h"], as_index=False)
           .agg(
               collisions=("EVENT_UNIQUE_ID","count") if "EVENT_UNIQUE_ID" in coll_event_clean.columns else ("time_3h","size"),
               injury_collisions=("INJURY_COLLISIONS_01","sum") if "INJURY_COLLISIONS_01" in coll_event_clean.columns else ("time_3h","size"),
               ftr_collisions=("FTR_COLLISIONS_01","sum") if "FTR_COLLISIONS_01" in coll_event_clean.columns else ("time_3h","size"),
               pd_collisions=("PD_COLLISIONS_01","sum") if "PD_COLLISIONS_01" in coll_event_clean.columns else ("time_3h","size"),
               fatalities=("FATALITIES","sum") if "FATALITIES" in coll_event_clean.columns else ("time_3h","size"),

               pedestrian_collisions=("PEDESTRIAN_01","sum") if "PEDESTRIAN_01" in coll_event_clean.columns else ("time_3h","size"),
               bicycle_collisions=("BICYCLE_01","sum") if "BICYCLE_01" in coll_event_clean.columns else ("time_3h","size"),
               automobile_collisions=("AUTOMOBILE_01","sum") if "AUTOMOBILE_01" in coll_event_clean.columns else ("time_3h","size"),
               motorcycle_collisions=("MOTORCYCLE_01","sum") if "MOTORCYCLE_01" in coll_event_clean.columns else ("time_3h","size"),
               passenger_collisions=("PASSENGER_01","sum") if "PASSENGER_01" in coll_event_clean.columns else ("time_3h","size"),
           ))

# Force integer counts (no decimals)
for c in coll_3h.columns:
    if c not in ["HOOD_158_CODE","time_3h"]:
        coll_3h[c] = pd.to_numeric(coll_3h[c], errors="coerce").fillna(0).astype(int)

# Time features (important for model)
coll_3h["block_hour"] = coll_3h["time_3h"].dt.hour                 # 0,3,6,...,21
coll_3h["OCC_DOW"] = coll_3h["time_3h"].dt.day_name()              # names for dashboard
coll_3h["OCC_MONTH"] = coll_3h["time_3h"].dt.month_name()          # names for dashboard
coll_3h["dow_num"] = coll_3h["time_3h"].dt.dayofweek               # 0-6 numeric for modelling
coll_3h["month_num"] = coll_3h["time_3h"].dt.month                 # 1-12 numeric for modelling
coll_3h["is_weekend"] = (coll_3h["dow_num"] >= 5).astype(int)

# (Optional) save aggregated collisions
coll_3h.to_csv(OUT_DIR / "collisions_3h_hood.csv", index=False)
