
# Run thes scrips to download the raw data files into the directory data/raw/
# Run "pip install requests" in VS code terminal 
# In the terminal, navigate to the repository folder where the download_data.py is and run: python download_data.py
# Ths will download three Excel files into data/raw/ folder

from pathlib import Path
import re
import sys
import requests

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

URLS = {
    "Collision raw.xlsx": "https://docs.google.com/spreadsheets/d/1enTSAf9-ujI7G_d_lceGvGpRMsp3i1p-/edit?usp=drive_link&ouid=115083030499694317354&rtpof=true&sd=true",
    "Weather daily.xlsx": "https://docs.google.com/spreadsheets/d/1r4l3YHJ7GC73SVE7E-jv8c1YmC6D6clX/edit?usp=drive_link&ouid=115083030499694317354&rtpof=true&sd=true",
    "Weather hourly.xlsx": "https://docs.google.com/spreadsheets/d/1T1U9Yf_Fu4RJeeMOBjGkn9hYcfL8hU8l/edit?usp=drive_link&ouid=115083030499694317354&rtpof=true&sd=true",
}

def extract_sheet_id(url: str) -> str:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
    if not m:
        raise ValueError(f"Cannot extract Sheet ID from URL: {url}")
    return m.group(1)

def download_as_xlsx(sheet_url: str, out_path: Path):
    sheet_id = extract_sheet_id(sheet_url)
    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"

    resp = requests.get(export_url, stream=True)
    resp.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

def main():
    print("Downloading files into data/raw/ ...\n")

    for filename, url in URLS.items():
        try:
            out = RAW_DIR / filename
            print(f"Downloading: {filename}")
            download_as_xlsx(url, out)
            size_mb = out.stat().st_size / (1024 * 1024)
            print(f"✅ Saved to: {out} ({size_mb:.2f} MB)\n")
        except Exception as e:
            print(f"❌ Failed to download {filename}")
            print("Reason:", e)
            sys.exit(1)

    print("All downloads complete.")
    print("Check your files here:", RAW_DIR.resolve())

if __name__ == "__main__":
    main()
