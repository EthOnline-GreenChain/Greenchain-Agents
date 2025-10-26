
#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, sys, hashlib
from typing import Dict, Any
from greenchain.schema import Envelope, SourceInfo, envelope_json_schema_for_asi, BillingStatement, IntervalReading
from greenchain.pdf_text import extract_text_from_pdf
from greenchain.asi_client import ASIClient

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def build_prompt_for_pdf(text: str) -> str:
    # Keep the prompt short; ASI One will see the full text
    return (
        "You are converting utility electricity bills (PDF text) into a structured JSON payload "
        "for Greenchain. Extract the billing statement fields and, if any usage tables are present, "
        "also extract interval readings. Use ISO dates, UTC datetimes for intervals, kWh for energy.\n\n"
        "PDF TEXT START\n"
        f"{text}\n"  # Cap to avoid accidental token explosion
        "PDF TEXT END\n"
    )

def main():
    ap = argparse.ArgumentParser(description="Greenchain PDF ingestion → unified JSON via ASI:One")
    ap.add_argument("--file", required=True, help="Path to a PDF bill")
    ap.add_argument("--output", default=None, help="Where to write JSON (defaults to stdout)")
    ap.add_argument("--filename", default=None, help="Optional original filename (if different)")
    args = ap.parse_args()

    with open(args.file, "rb") as f:
        pdf_bytes = f.read()

    text = extract_text_from_pdf(args.file)
    src = SourceInfo(source_type="pdf", filename=args.filename or os.path.basename(args.file), sha256=sha256_bytes(pdf_bytes), extra={})
    env = Envelope(source=src)

    client = ASIClient()
    schema = envelope_json_schema_for_asi()
    prompt = build_prompt_for_pdf(text)
    extracted: Dict[str, Any] = client.chat_structured(prompt, schema)
    # Validate with Pydantic models
    if "billing_statements" in extracted:
        env.billing_statements = [BillingStatement(**bs) for bs in extracted["billing_statements"]]
    if "interval_readings" in extracted:
        env.interval_readings = [IntervalReading(**ir) for ir in extracted["interval_readings"]]
    env.close()

    out = env.model_dump()
    if args.output:
        with open(args.output, "w") as w:
            json.dump(out, w, indent=2, default=str)
    else:
        print(json.dumps(out, indent=2, default=str))

if __name__ == "__main__":
    main()
