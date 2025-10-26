
#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, pandas as pd, io, hashlib
from typing import Dict, Any, List
from greenchain.schema import Envelope, SourceInfo, envelope_json_schema_for_asi, IntervalReading
from greenchain.asi_client import ASIClient

def sha256_bytes(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()

def excel_or_csv_to_text(path: str) -> str:
    if path.lower().endswith(".csv"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    # Excel: convert each sheet to CSV-like text separated by headers
    xl = pd.ExcelFile(path)
    parts: List[str] = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        parts.append(f"### SHEET: {sheet} ###")
        parts.append(df.to_csv(index=False))
    return "\n".join(parts)

def build_prompt_for_tabular(text: str) -> str:
    return (
        "You are converting an energy meter export (CSV or Excel converted to CSV text) into Greenchain's "
        "canonical JSON. The file likely contains interval meter readings (kWh consumption) and sometimes a billed total. "
        "Return ONLY JSON that matches the provided schema. Use UTC for timestamps; infer ts_end_utc if only a single timestamp exists per row "
        "by adding the dataset's interval length. If column names vary (e.g., energy, consumption, import_kwh), interpret sensibly.\n\n"
        "CSV TEXT START\n"
        f"{text}\n"
        "CSV TEXT END\n"
    )

def main():
    ap = argparse.ArgumentParser(description="Greenchain Excel/CSV ingestion → unified JSON via ASI:One")
    ap.add_argument("--file", required=True, help="Path to .csv or .xlsx")
    ap.add_argument("--output", default=None, help="Where to write JSON (defaults to stdout)")
    ap.add_argument("--filename", default=None, help="Optional original filename")
    args = ap.parse_args()

    with open(args.file, "rb") as f:
        raw_bytes = f.read()
    body_text = excel_or_csv_to_text(args.file)

    src = SourceInfo(source_type="excel", filename=args.filename or os.path.basename(args.file), sha256=sha256_bytes(raw_bytes), extra={})
    env = Envelope(source=src)

    client = ASIClient()
    schema = envelope_json_schema_for_asi()
    prompt = build_prompt_for_tabular(body_text)
    extracted: Dict[str, Any] = client.chat_structured(prompt, schema)
    if "interval_readings" in extracted:
        env.interval_readings = [IntervalReading(**ir) for ir in extracted["interval_readings"]]
    # Some spreadsheets might include a billed period too; accept it if present
    if "billing_statements" in extracted:
        from greenchain.schema import BillingStatement
        env.billing_statements = [BillingStatement(**bs) for bs in extracted["billing_statements"]]
    env.close()

    out = env.model_dump()
    if args.output:
        with open(args.output, "w") as w:
            json.dump(out, w, indent=2, default=str)
    else:
        print(json.dumps(out, indent=2, default=str))

if __name__ == "__main__":
    main()
