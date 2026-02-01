from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests
import yaml
from tqdm import tqdm

from src.utils.io import ensure_dir, save_csv

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_city_list(cfg: dict) -> list[str]:
    """
    Prefer cities extracted from the structured dataset (data/processed/cities.txt),
    otherwise fall back to config.yaml cities list.
    """
    cities_txt = Path("data/processed/cities.txt")
    if cities_txt.exists():
        cities = [ln.strip() for ln in cities_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if cities:
            print(f"Using {len(cities)} cities from {cities_txt}")
            return cities

    cities = cfg.get("cities", [])
    print(f"Using {len(cities)} fallback cities from config.yaml")
    return cities


def fetch_city_articles(
    city: str,
    query_spine: str,
    start: str,
    end: str,
    maxrecords: int,
    sort: str,
) -> list[dict]:
    """
    Fetch a list of articles from GDELT DOC API for a given city and query spine.
    NOTE: GDELT can rate-limit (429). We handle failures upstream; keep this simple.
    """
    # Keep quotes around city name to reduce false positives; broad query spine handles coverage.
    q = f'"{city}" AND {query_spine}'

    params = {
        "query": q,
        "mode": "ArtList",
        "format": "json",
        "startdatetime": start.replace("-", "") + "000000",
        "enddatetime": end.replace("-", "") + "235959",
        "maxrecords": int(maxrecords),
        "sort": sort,
    }

    r = requests.get(GDELT_URL, params=params, timeout=60)

    # If rate-limited, raise for upstream handling/logging
    r.raise_for_status()

    data = r.json()
    articles = data.get("articles", []) or []

    out = []
    for a in articles:
        out.append(
            {
                "city": city,
                "seendate": a.get("seendate"),
                "url": a.get("url"),
                "title": a.get("title"),
                "description": a.get("description"),
                "language": a.get("language"),
                "domain": a.get("domain"),
                "sourceCountry": a.get("sourceCountry"),
            }
        )
    return out


def main(cfg_path: str = "legacy/conflict-risk/config.yaml") -> None:
    cfg = load_config(cfg_path)

    start = cfg["date_range"]["start"]
    end = cfg["date_range"]["end"]
    query_spine = cfg["gdelt"]["query_spine"]
    maxrecords = int(cfg["gdelt"]["maxrecords_per_city"])
    sort = cfg["gdelt"]["sort"]

    cities = load_city_list(cfg)

    raw_dir = ensure_dir("data/raw")
    all_rows: list[dict] = []

    # Slower delay reduces 429 errors and increases total successful cities.
    delay_seconds = 1.5

    for city in tqdm(cities, desc="Fetching GDELT"):
        try:
            rows = fetch_city_articles(city, query_spine, start, end, maxrecords, sort)
            all_rows.extend(rows)
        except Exception as e:
            # Common failures: 429 Too Many Requests, or HTML response causing JSON decode issues.
            print(f"[WARN] Failed city={city}: {e}")

        time.sleep(delay_seconds)

    df = pd.DataFrame(all_rows)
    out_path = Path(raw_dir) / "gdelt_articles.csv"
    save_csv(df, out_path)
    print(f"Saved {len(df):,} articles -> {out_path}")


if __name__ == "__main__":
    main()
