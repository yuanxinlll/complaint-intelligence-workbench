"""Inspect the small official API probe. No training or automatic bulk download."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unicodedata
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/raw/probe_2024.json"
PARAMS = {
    "date_received_min": "2024-01-01",
    "date_received_max": "2025-01-01",
    "size": 100,
    "sort": "created_date_asc",
    "no_highlight": "true",
}
URL = (
    "https://www.consumerfinance.gov/data-research/consumer-complaints/"
    "search/api/v1/?" + urlencode(PARAMS)
)
NARRATIVE_KEYS = ("complaint_what_happened", "consumer_complaint_narrative", "Consumer complaint narrative", "narrative")


def normalize_text(value):
    """Normalize Unicode and whitespace for inspection; retain words and numbers."""
    return " ".join(unicodedata.normalize("NFKC", str(value)).split()).casefold()


def inspect_probe(path):
    """Return counts and schema from saved API JSON, without logging complaint text."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("timed_out"):
        raise ValueError("API query timed out; this response is not accepted.")
    if payload.get("_shards", {}).get("failed", 0):
        raise ValueError("API reported failed shards; this response is not accepted.")
    rows = [hit["_source"] for hit in payload["hits"]["hits"]]
    if not rows:
        raise ValueError("No records returned.")
    fields = sorted(set().union(*(row.keys() for row in rows)))
    texts = [next((row[key] for key in NARRATIVE_KEYS if row.get(key)), "") for row in rows]
    normalized = [normalize_text(text) for text in texts if normalize_text(text)]
    ids = [row.get("complaint_id") for row in rows if row.get("complaint_id") is not None]
    dates = [datetime.fromisoformat(row["date_received"].replace("Z", "+00:00")).date().isoformat() for row in rows]
    report = {
        "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file": str(path.relative_to(ROOT)),
        "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_bytes": path.stat().st_size,
        "request_url": URL,
        "filters": PARAMS,
        "sample_selection": "First 100 API records sorted by created_date_asc; not a representative sample.",
        "returned_records": len(rows),
        "matching_total": payload["hits"]["total"],
        "api_meta": payload.get("_meta", {}),
        "field_types": {key: sorted({type(row[key]).__name__ for row in rows if key in row}) for key in fields},
        "missing_or_blank": {key: sum(row.get(key) is None or str(row.get(key, "")).strip() == "" for row in rows) for key in fields},
        "narrative_fields_present": [key for key in NARRATIVE_KEYS if key in fields],
        "usable_narratives": len(normalized),
        "excluded_from_training": len(rows) - len(normalized),
        "exclusion_reason": "No available narrative text; these records cannot train a text classifier.",
        "duplicate_id_excess_rows": len(ids) - len(set(ids)),
        "normalized_duplicate_narrative_excess_rows": len(normalized) - len(set(normalized)),
        "narrative_duplicate_check_limitation": "No text available; zero duplicates is not evidence of text uniqueness.",
        "sample_date_min": min(dates),
        "sample_date_max": max(dates),
        "dates_in_requested_2024_range": all("2024-01-01" <= date < "2025-01-01" for date in dates),
        "sample_product_counts": dict(Counter(row.get("product") for row in rows)),
        "sample_issue_counts": dict(Counter(row.get("issue") for row in rows)),
        "full_query_product_counts": {bucket["key"]: bucket["doc_count"] for bucket in payload["aggregations"]["product"]["product"]["buckets"]},
        "training_ready": False,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download at most 100 records only if the raw file does not exist.")
    args = parser.parse_args()
    if args.download:
        if RAW_PATH.exists():
            raise FileExistsError("Raw file already exists; it will not be overwritten.")
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL, timeout=45) as response:
            content = response.read(5_000_001)
        if len(content) > 5_000_000:
            raise ValueError("Probe exceeded the 5 MB safety limit.")
        json.loads(content)
        with RAW_PATH.open("xb") as output:
            output.write(content)
    report = inspect_probe(RAW_PATH)
    report_path = ROOT / "reports/probe_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["returned_records", "raw_bytes", "narrative_fields_present", "usable_narratives", "excluded_from_training", "duplicate_id_excess_rows", "sample_date_min", "sample_date_max", "training_ready"]}, indent=2, ensure_ascii=False))
    print("Report:", report_path)


if __name__ == "__main__":
    main()
