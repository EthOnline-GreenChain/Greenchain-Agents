
# Greenchain Ingestion (PDF, Excel, Gmail, GCP CO₂)

**Drop-in, Python-only scripts** for Greenchain's data ingestion MVP, aligned with the canonical schemas in the *Data Ingestion Spec v1.1* and the overall *Greenchain* idea spec. Outputs are a single **binding JSON envelope** that downstream agents can consume.

> Canonical entities: `billing_statements` (+ `billing_lines`), `interval_readings`, plus a `gcp_footprint` object for CO₂ estimates. (See `greenchain/schema.py`.)

- Spec references: Data Ingestion Spec v1.1 and Greenchain (provided)  
- ASI:One docs (API + structured outputs): https://docs.asi1.ai/documentation/getting-started/overview and https://docs.asi1.ai/documentation/build-with-asi-one/structured-data

## What’s included

```
greenchain_ingest/
  greenchain/
    __init__.py
    schema.py           # Pydantic models + JSON schema for ASI:One
    asi_client.py       # Minimal ASI One HTTP client (OpenAI compatible)
    pdf_text.py         # 'pdfloader' style text extraction (LangChain or PyPDF fallback)
  scripts/
    pdf_ingest.py       # (1) Upload PDF → ASI One → JSON
    excel_ingest.py     # (2) Upload Excel/CSV → ASI One → JSON
    gmail_listener.py   # (3) Poll Gmail for sender → PDF attachments → JSON
    gcp_co2.py          # (4) Estimate CO₂ for current GCE VM
  runner.py             # Single FastAPI app exposing endpoints for each flow
  requirements.txt
  README.md
```

## Install

```bash
cd greenchain_ingest
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configure ASI:One

Set an API key and (optionally) model:

```bash
export ASI_ONE_API_KEY="sk-..."
export ASI_ONE_BASE_URL="https://api.asi1.ai/v1"   # default
export ASI_ONE_MODEL="asi1-extended"               # default used for structured JSON
```

> ASI:One is OpenAI-compatible (`/v1/chat/completions`). Model names include `asi1-mini`, `asi1-fast`, `asi1-extended`. See docs.


## 1) PDF ingestion

```bash
python scripts/pdf_ingest.py --file ./samples/bill.pdf --output out.json
```

## 2) Excel / CSV ingestion

```bash
python scripts/excel_ingest.py --file ./samples/intervals.xlsx --output out.json
```

## 3) Gmail (pull) — one-shot poll for a sender

1. Create a Google OAuth client (Desktop App) and download `credentials.json`.
2. First run opens a browser to consent and stores `token.json` locally.

```bash
python scripts/gmail_listener.py --sender billing@utility.com --credentials credentials.json --token token.json
```

Outputs are written to `./out_gmail/*.json`.

## 4) GCP CO₂ for *this* instance

Run on a Compute Engine VM with default service account (or pass env fallbacks):

```bash
python scripts/gcp_co2.py --hours 1 --avg-util 0.30
```

> Uses metadata server to detect region + machine type and the Compute API (if available) to resolve vCPU & memory.  
> Grid carbon intensity is a simple mapping from Google’s region table; adjust as needed.

## 5) Runner (all endpoints)

```bash
uvicorn runner:app --host 0.0.0.0 --port 8000
```

**Endpoints**

- `POST /ingest/pdf` — `multipart/form-data` with `file`
- `POST /ingest/excel` — `multipart/form-data` with `file`
- `GET  /gcp/co2?hours=1&avg_util=0.3`
- (For Gmail, use the script for now to keep auth simple.)

## Output binding schema

All flows return an **Envelope**:

```jsonc
{
  "schema_version": "greenchain.ingest.v1",
  "run_id": "...",
  "started_at": "...",
  "finished_at": "...",
  "source": { "source_type": "pdf|excel|gmail|gcp", "filename": "...", "sha256": "..." },
  "billing_statements": [
    {
      "billing_period_start": "YYYY-MM-DD",
      "billing_period_end": "YYYY-MM-DD",
      "statement_date": "YYYY-MM-DD",
      "subtotal": 0,
      "tax": 0,
      "total": 123.45,
      "currency": "USD",
      "rate_plan": "TOU-8",
      "lines": [
        {"component_type": "energy", "description": "Energy charges", "amount": 100.0}
      ]
    }
  ],
  "interval_readings": [
    {
      "meter_id": "MTR123",
      "ts_start_utc": "2025-09-01T00:00:00Z",
      "ts_end_utc": "2025-09-01T00:30:00Z",
      "kwh": 0.42,
      "quality": "A"
    }
  ],
  "gcp_footprint": {
    "region": "us-central1",
    "hours": 1.0,
    "estimated_kwh": 0.123,
    "estimated_co2e_kg": 0.051,
    "...": "..."
  }
}
```

This aligns with the canonical tables described in *Data Ingestion Spec v1.1* (billing statements + lines, interval readings, source metadata, and run metadata).

## Notes & rationale

- We deliberately keep **LLM-only parsing** for PDFs/Excel per request (no heavy parsers). For robustness later, add rule-based extractors and use ASI:One as fallback.
- The **JSON schema** given to ASI:One is strict and validated server-side with **Pydantic**.
- Gmail flow is **pull/poll** to avoid push infra; idempotency and labeling can be added if you want.
- GCP CO₂ uses public **grid intensity** figures and CCF **energy coefficients**; it’s an estimate (operational emissions only).

---

Happy hacking! 🎯
