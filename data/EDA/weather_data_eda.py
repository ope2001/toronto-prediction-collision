# ============================================================
# BASIC EDA for weather raw 23-25.xlsx
# (Load + Inspect + Raw Plots only)
# No cleaning / No aggregation / No saving
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# 0) CONFIG
# ----------------------------
FILE_PATH = r"C:\Users\Tanjiro\Downloads\Project\Project\raw_data\weather raw 23-25.xlsx"  # update this to your actual file path

# ----------------------------
# 1) LOAD DATA
# ----------------------------
def load_data(path):
    df = pd.read_excel(path, sheet_name=0)

    print("\n" + "="*70)
    print("RAW DATA LOADED")
    print("="*70)
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    return df


# ----------------------------
# 2) BASIC INSPECTION
# ----------------------------
def inspect_data(df):
    print("\n" + "="*70)
    print("FIRST 5 ROWS")
    print("="*70)
    print(df.head())

    print("\n" + "="*70)
    print("LAST 5 ROWS")
    print("="*70)
    print(df.tail())

    print("\n" + "="*70)
    print("DATA TYPES")
    print("="*70)
    print(df.dtypes)

    print("\n" + "="*70)
    print("NON-NULL COUNTS")
    print("="*70)
    print(df.info())

    print("\n" + "="*70)
    print("MISSING VALUES")
    print("="*70)
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2)
    }).sort_values("missing_pct", ascending=False)
    print(missing)

    print("\n" + "="*70)
    print("DUPLICATES")
    print("="*70)
    print(f"Full duplicate rows: {df.duplicated().sum()}")

    # Optional: timestamp duplicate check (only inspection, no cleaning)
    if "DMY" in df.columns and "Time" in df.columns:
        dmy = pd.to_datetime(df["DMY"], errors="coerce")
        hr = pd.to_numeric(df["Time"], errors="coerce")
        dt_preview = dmy.dt.floor("D") + pd.to_timedelta(hr.fillna(0), unit="h")
        print(f"Duplicate timestamp-like rows (DMY + Time): {dt_preview.duplicated().sum()}")


# ----------------------------
# 3) NUMERIC SUMMARY
# ----------------------------
def numeric_summary(df):
    print("\n" + "="*70)
    print("NUMERIC SUMMARY")
    print("="*70)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not num_cols:
        print("No numeric columns detected.")
        return []

    print(df[num_cols].describe().T)

    # extra quick stats
    extra = pd.DataFrame({
        "n_unique": df[num_cols].nunique(),
        "missing": df[num_cols].isna().sum(),
        "missing_pct": (df[num_cols].isna().mean() * 100).round(2)
    })
    print("\nExtra numeric column stats:")
    print(extra)

    return num_cols


# ----------------------------
# 4) CATEGORICAL SUMMARY
# ----------------------------
def categorical_summary(df, top_n=15):
    print("\n" + "="*70)
    print("CATEGORICAL SUMMARY")
    print("="*70)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Include date-like columns stored as object
    if len(cat_cols) == 0:
        print("No categorical/object columns detected.")
        return []

    for col in cat_cols:
        print(f"\n--- {col} ---")
        print(f"Unique values: {df[col].nunique(dropna=False)}")
        print(df[col].value_counts(dropna=False).head(top_n))

    return cat_cols


# ----------------------------
# 5) DATE/TIME PREVIEW (NO CLEANING)
# ----------------------------
def datetime_preview(df):
    print("\n" + "="*70)
    print("DATE/TIME PREVIEW (RAW)")
    print("="*70)

    if "DMY" not in df.columns:
        print("Column 'DMY' not found.")
        return

    dmy = pd.to_datetime(df["DMY"], errors="coerce")
    print(f"Parsed DMY valid rows: {dmy.notna().sum()} / {len(dmy)}")
    print(f"DMY min: {dmy.min()}")
    print(f"DMY max: {dmy.max()}")

    if "Time" in df.columns:
        time_num = pd.to_numeric(df["Time"], errors="coerce")
        print(f"\nTime column valid numeric rows: {time_num.notna().sum()} / {len(time_num)}")
        print("Time unique sample (sorted):")
        print(sorted(time_num.dropna().unique())[:30])

        # Build raw datetime preview only (no filtering)
        dt_preview = dmy.dt.floor("D") + pd.to_timedelta(time_num.fillna(0), unit="h")
        print(f"\nDatetime preview min: {dt_preview.min()}")
        print(f"Datetime preview max: {dt_preview.max()}")


# ----------------------------
# 6) RAW DATA QUALITY CHECKS (ONLY FLAGS)
# ----------------------------
def raw_quality_flags(df):
    print("\n" + "="*70)
    print("RAW QUALITY FLAGS (NO CLEANING)")
    print("="*70)

    checks = {}

    def count_invalid(col, condition):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            checks[col] = int(condition(s).sum())

    count_invalid("relative_humidity", lambda s: (s < 0) | (s > 100))
    count_invalid("cloud_cover_8", lambda s: (s < 0) | (s > 8))
    count_invalid("visibility", lambda s: s < 0)
    count_invalid("wind_speed", lambda s: s < 0)
    count_invalid("rain", lambda s: s < 0)
    count_invalid("snow", lambda s: s < 0)
    count_invalid("snow_on_ground", lambda s: s < 0)

    if "Time" in df.columns:
        t = pd.to_numeric(df["Time"], errors="coerce")
        checks["Time outside 0-23"] = int(((t < 0) | (t > 23)).sum())

    if checks:
        print(pd.Series(checks, name="Invalid count"))
    else:
        print("No standard weather columns found for raw checks.")


# ----------------------------
# 7) MISSING VALUES PLOT
# ----------------------------
def plot_missing_values(df):
    miss_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    miss_pct = miss_pct[miss_pct > 0]

    if miss_pct.empty:
        print("\nNo missing values to plot.")
        return

    plt.figure(figsize=(12, 4))
    plt.bar(miss_pct.index.astype(str), miss_pct.values)
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Missing (%)")
    plt.title("Missing Values by Column (Raw Data)")
    plt.tight_layout()
    plt.show()


# ----------------------------
# 8) HISTOGRAMS (RAW NUMERIC)
# ----------------------------
def plot_numeric_histograms(df, num_cols, max_cols=12):
    if not num_cols:
        return

    # Limit for cleaner basic EDA
    cols = num_cols[:max_cols]
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cols):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        axes[i].hist(s, bins=30)
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Raw Numeric Distributions", y=1.02)
    plt.tight_layout()
    plt.show()


# ----------------------------
# 9) BOXPLOTS (RAW NUMERIC)
# ----------------------------
def plot_basic_boxplots(df):
    candidate_cols = [
        "temperature", "dew_point", "wind_speed", "pressure_station", "pressure_sea",
        "relative_humidity", "visibility", "rain", "snow", "snow_on_ground"
    ]
    cols = [c for c in candidate_cols if c in df.columns]

    if not cols:
        print("\nNo common weather numeric columns found for boxplots.")
        return

    n = len(cols)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cols):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        axes[i].boxplot(s, vert=True)
        axes[i].set_title(f"Boxplot: {col}")
        axes[i].set_ylabel(col)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


# ----------------------------
# 10) SIMPLE TIME PLOTS (RAW ONLY)
# ----------------------------
def plot_raw_time_trends(df):
    # Build preview datetime without cleaning
    if "DMY" not in df.columns or "Time" not in df.columns:
        print("\nDMY/Time not found. Skipping time trend plots.")
        return

    dmy = pd.to_datetime(df["DMY"], errors="coerce")
    hr = pd.to_numeric(df["Time"], errors="coerce")
    dt = dmy.dt.floor("D") + pd.to_timedelta(hr.fillna(0), unit="h")

    plot_df = df.copy()
    plot_df["datetime_preview"] = dt

    # only rows where datetime parsed successfully
    plot_df = plot_df.dropna(subset=["datetime_preview"]).sort_values("datetime_preview")

    trend_cols = ["temperature", "relative_humidity", "wind_speed", "visibility", "pressure_sea"]
    trend_cols = [c for c in trend_cols if c in plot_df.columns]

    for col in trend_cols:
        s = pd.to_numeric(plot_df[col], errors="coerce")

        plt.figure(figsize=(14, 4))
        plt.plot(plot_df["datetime_preview"], s, linewidth=0.7)
        plt.title(f"Raw Time Trend: {col}")
        plt.xlabel("Datetime (preview from DMY + Time)")
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()


# ----------------------------
# 11) CORRELATION HEATMAP (RAW NUMERIC)
# ----------------------------
def plot_correlation_heatmap(df, num_cols):
    if len(num_cols) < 2:
        print("\nNot enough numeric columns for correlation heatmap.")
        return

    corr = df[num_cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.title("Correlation Heatmap (Raw Numeric Columns)")
    plt.tight_layout()
    plt.show()


# ----------------------------
# 12) RUN BASIC EDA
# ----------------------------
def run_basic_eda():
    df = load_data(FILE_PATH)

    inspect_data(df)
    num_cols = numeric_summary(df)
    categorical_summary(df)
    datetime_preview(df)
    raw_quality_flags(df)

    # Plots
    plot_missing_values(df)
    plot_numeric_histograms(df, num_cols)
    plot_basic_boxplots(df)
    plot_raw_time_trends(df)
    plot_correlation_heatmap(df, num_cols)

    print("\nBasic EDA complete ✅ (No cleaning/aggregation performed)")


# ----------------------------
# 13) EXECUTE
# ----------------------------
if __name__ == "__main__":
    run_basic_eda()