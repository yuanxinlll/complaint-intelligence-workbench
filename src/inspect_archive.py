"""Inspect an official historical CSV and preserve a bounded, uncleaned subset."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRODUCT = "Debt collection"
REQUIRED = ["Complaint ID", "Date received", "Product", "Issue", "Consumer complaint narrative"]
SOURCE_URL = "https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/"


def normalize_text(text):
    """Create a duplicate comparison key, without changing the saved raw text."""
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_archive(source):
    """Read CSV in chunks; return metadata and the selected original rows."""
    rows = 0
    missing = Counter()
    products = Counter()
    narrative_products = Counter()
    ids = Counter()
    invalid_dates = 0
    date_min, date_max = None, None
    selected_chunks = []
    columns = None
    for chunk in pd.read_csv(source, dtype=str, keep_default_na=False, chunksize=25000, encoding="utf-8-sig"):
        if columns is None:
            columns = chunk.columns.tolist()
            absent = set(REQUIRED) - set(columns)
            if absent:
                raise ValueError(f"Missing required columns: {sorted(absent)}")
        rows += len(chunk)
        blank = chunk.apply(lambda column: column.str.strip().eq(""))
        missing.update(blank.sum().to_dict())
        products.update(chunk["Product"])
        has_text = ~blank["Consumer complaint narrative"]
        narrative_products.update(chunk.loc[has_text, "Product"])
        ids.update(chunk.loc[~blank["Complaint ID"], "Complaint ID"])
        dates = pd.to_datetime(chunk["Date received"], format="%Y-%m-%d", errors="coerce")
        invalid_dates += int(dates.isna().sum())
        if dates.notna().any():
            low, high = dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")
            date_min = min(date_min or low, low)
            date_max = max(date_max or high, high)
        # No class-based filtering, deduplication or model selection in stage 1.
        mask = chunk["Product"].eq(PRODUCT) & has_text & ~blank["Issue"]
        selected_chunks.append(chunk.loc[mask].copy())
    selected = pd.concat(selected_chunks, ignore_index=True)
    if selected.empty or len(selected) > 20000:
        raise ValueError(f"Selected size {len(selected)} is outside the agreed 1–20,000 bound.")
    normalized = selected["Consumer complaint narrative"].map(normalize_text)
    groups = pd.DataFrame({"text": normalized, "issue": selected["Issue"]}).groupby("text")["issue"].nunique()
    repeated = normalized.duplicated(keep=False)
    counts = selected["Issue"].value_counts()
    monthly = pd.crosstab(selected["Issue"], selected["Date received"].str[:7])
    examples = []
    # Deterministic short public examples; no whole narratives in console logs.
    ordered = selected.sort_values(["Date received", "Complaint ID"], kind="stable")
    for _, row in ordered.head(3).iterrows():
        text = " ".join(row["Consumer complaint narrative"].split())
        examples.append({"complaint_id": row["Complaint ID"], "received_date": row["Date received"], "issue": row["Issue"], "excerpt": text[:220] + ("…" if len(text) > 220 else "")})
    report = {
        "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(source), "official_archive_page": SOURCE_URL, "downloaded_by": "user; exact download time not recorded", "sha256": sha256_file(source), "bytes": source.stat().st_size},
        "archive": {"rows": rows, "columns": columns, "read_dtype": "All columns explicitly read as strings to preserve IDs and empty cells; dates validated separately.", "missing_or_blank": dict(missing), "invalid_dates": invalid_dates, "date_min": date_min, "date_max": date_max, "duplicate_id_excess_rows": sum(count - 1 for count in ids.values()), "product_counts": dict(products), "product_counts_with_narrative": dict(narrative_products)},
        "selection": {"product": PRODUCT, "date_min": selected["Date received"].min(), "date_max": selected["Date received"].max(), "rows": len(selected), "class_counts": counts.to_dict(), "class_percentages": (counts / len(selected) * 100).round(2).to_dict(), "monthly_class_counts": monthly.to_dict(), "missing_or_blank": selected.apply(lambda column: column.str.strip().eq("").sum()).to_dict(), "duplicate_id_excess_rows": int(selected["Complaint ID"].duplicated().sum()), "normalized_duplicate_excess_rows": int(normalized.duplicated().sum()), "normalized_duplicate_group_count": int(normalized[repeated].nunique()), "rows_in_duplicate_groups": int(repeated.sum()), "conflicting_label_group_count": int((groups > 1).sum()), "normalization": "NFKC Unicode, whitespace collapse, casefold; raw CSV unchanged", "duplicate_policy": "Stage 1 reports duplicates only. Stage 2 must prevent duplicate text groups crossing chronological splits and report handling of conflicting labels.", "class_exclusions": [], "selection_exclusions": {"other_products": rows - products[PRODUCT], "selected_product_without_narrative": products[PRODUCT] - narrative_products[PRODUCT], "selected_product_with_narrative_but_missing_issue": narrative_products[PRODUCT] - len(selected)}, "text_length_characters": selected["Consumer complaint narrative"].str.len().describe(percentiles=[.5, .9, .99]).to_dict(), "public_examples_truncated": examples},
        "limitations": ["Historical public complaints, not a representative consumer sample.", "Issue is a consumer-selected label, not expert verified ground truth.", "Single product and only two months; later evaluation cannot establish long-term robustness.", "Monthly class counts are descriptive; no model or threshold selected and no train/validation/test split yet.", "Similar text does not necessarily describe the same event.", "Original download is not modified; subset retains all original columns, but only narrative will be a model feature."],
    }
    return report, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    subset = ROOT / "data/raw/debt_collection_2024_nov_dec.csv"
    report_path = ROOT / "reports/archive_report.json"
    if subset.exists():
        raise FileExistsError(f"Refusing to overwrite raw subset: {subset}")
    report, selected = inspect_archive(args.source)
    subset.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(subset, index=False, encoding="utf-8", mode="x")
    # Verify saved subset round-trips exactly, including original narrative content.
    restored = pd.read_csv(subset, dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(selected, restored)
    report["saved_subset"] = {"path": str(subset.relative_to(ROOT)), "bytes": subset.stat().st_size, "sha256": sha256_file(subset), "roundtrip_verified": True}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"archive_rows": report["archive"]["rows"], "selection": {key: report["selection"][key] for key in ["product", "rows", "class_counts", "monthly_class_counts", "normalized_duplicate_excess_rows", "normalized_duplicate_group_count", "rows_in_duplicate_groups", "conflicting_label_group_count", "selection_exclusions"]}, "saved_subset": report["saved_subset"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
