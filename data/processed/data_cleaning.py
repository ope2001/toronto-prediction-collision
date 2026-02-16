from pathlib import Path
import pandas as pd

RAW_DIR = Path(r"C:\Users\Tanjiro\Desktop\MDA UNF\T5\Capstone project\github\toronto-prediction-collision\data\raw\data\raw")

files = {
    "collision": "Collision raw.xlsx",
    "weather_daily": "Weather daily.xlsx",
    "weather_hourly": "Weather hourly.xlsx",
}

def read_xlsx(path: Path) -> pd.DataFrame:
    # reads first sheet by default
    return pd.read_excel(path, engine="openpyxl")

def quick_check(df: pd.DataFrame, name: str) -> None:
    print(f"\n=== {name} ===")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nDtypes (top 25):")
    print(df.dtypes.head(25))
    print("\nMissing values (top 10):")
    print(df.isna().sum().sort_values(ascending=False).head(10))
    print("\nHead:")
    print(df.head(5))

# Read all 3
dfs = {}
for key, fname in files.items():
    path = RAW_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    dfs[key] = read_xlsx(path)
    quick_check(dfs[key], fname)

# Access anytime:
collision_df = dfs["collision"]
weather_daily_df = dfs["weather_daily"]
weather_hourly_df = dfs["weather_hourly"]
