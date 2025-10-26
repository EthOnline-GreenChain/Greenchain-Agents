
#!/usr/bin/env python3
from __future__ import annotations

import io, os, json, uvicorn, hashlib, pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from greenchain.schema import Envelope, SourceInfo, BillingStatement, IntervalReading, envelope_json_schema_for_asi
from greenchain.pdf_text import extract_text_from_pdf
from greenchain.asi_client import ASIClient



# --- BEGIN: simple envelope coercer ---
import re
from typing import Any, Dict, List, Optional

def _get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur

def _pick(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return default

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _digits_only(s: Optional[str]) -> Optional[str]:
    if isinstance(s, str):
        ds = re.sub(r"\D", "", s)
        return ds or s
    return s

def _date_only(v: Optional[str]) -> Optional[str]:
    if isinstance(v, str) and len(v) >= 10:
        return v[:10]  # cut any T...Z
    return v

def _period_from_meter_readings(obj: Dict[str, Any]):
    readings = _get(obj, ["consumption", "meter_readings"]) or _get(obj, ["meter_readings"]) or []
    dates = []
    if isinstance(readings, list):
        for r in readings:
            rd = r.get("reading_date") or r.get("date")
            if isinstance(rd, str) and len(rd) >= 10:
                dates.append(rd[:10])
    if dates:
        return min(dates), max(dates)
    rd_prev = _get(obj, ["meter", "date_previous"])
    rd_curr = _get(obj, ["meter", "date_current"])
    if rd_prev and rd_curr:
        return _date_only(rd_prev), _date_only(rd_curr)
    return None, None

def coerce_any_to_envelope(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Already canonical
    if "billing_statements" in obj or "interval_readings" in obj:
        return obj

    # A) Wrapper shape
    if isinstance(obj.get("billingStatement"), dict):
        bs = obj["billingStatement"]
        bp = bs.get("billingPeriod", {}) or {}
        start = _pick(bs, ["billingPeriodStart"])
        end   = _pick(bs, ["billingPeriodEnd"])
        start = _date_only(start or bp.get("startDate"))
        end   = _date_only(end   or bp.get("endDate"))

        if not (start and end):
            s2, e2 = _period_from_meter_readings(bs)
            start, end = start or s2, end or e2

        pay = bs.get("paymentSummary", {}) or {}
        total_amount = _pick(pay, ["netAmountPayable","totalAmount","billAmount","amountDue","invoiceTotal"], 0.0)
        currency = _pick(bs, ["currency"], "INR")
        meter_id = _get(bs, ["serviceInformation","meterDetails","meterNumber"])

        total_kwh = _safe_float((bs.get("consumptionSummary") or {}).get("totalConsumptionKwh", 0.0))

        lines = []
        charges = bs.get("chargesBreakdown") or {}
        if isinstance(charges, dict):
            if "energyCharges" in charges:
                lines.append({"component_type":"energy","description":"Energy charges",
                              "amount": _safe_float(charges["energyCharges"].get("amount",0))})
            if "fixedCharges" in charges:
                lines.append({"component_type":"fixed","description":"Fixed charges",
                              "amount": _safe_float(charges["fixedCharges"].get("amount",0))})
            if "electricityTax" in charges:
                amt = charges["electricityTax"].get("amount", charges["electricityTax"].get("value", 0))
                lines.append({"component_type":"tax","description":"Electricity tax",
                              "amount": _safe_float(amt)})
        if not lines:
            lines = [{"component_type":"other","description":"Unspecified","amount": _safe_float(total_amount)}]

        out = {
            "billing_statements": [{
                "account_id": _digits_only(bs.get("customerAccountNumber")),
                "service_point_id": meter_id,
                "billing_period_start": start,
                "billing_period_end": end,
                "statement_date": _date_only(_pick(bs, ["billDate","statementDate"])),
                "subtotal": None,
                "tax": None,
                "total": _safe_float(total_amount),
                "currency": currency or "INR",
                "rate_plan": _get(bs, ["serviceInformation","tariffCategory"]),
                "lines": lines
            }],
            "interval_readings": []
        }
        if start and end and total_kwh > 0:
            out["interval_readings"].append({
                "meter_id": meter_id,
                "ts_start_utc": f"{start}T00:00:00Z",
                "ts_end_utc":   f"{end}T23:59:59Z",
                "kwh": total_kwh,
                "quality": "A"
            })
        return out

    # B) Flat shapes (cover all variants seen)
    start = (
        _date_only(_get(obj, ["bill_metadata","billing_period","start_date"])) or
        _date_only(_get(obj, ["billing_period","start"])) or
        _date_only(_pick(obj, ["billing_period_start","billingPeriodStart","startDate"]))
    )
    end = (
        _date_only(_get(obj, ["bill_metadata","billing_period","end_date"])) or
        _date_only(_get(obj, ["billing_period","end"])) or
        _date_only(_pick(obj, ["billing_period_end","billingPeriodEnd","endDate"]))
    )

    if not (start and end):
        s2, e2 = _period_from_meter_readings(obj)
        start, end = start or s2, end or e2

    statement_date = _date_only(
        _get(obj, ["bill_metadata","bill_date"]) or
        _pick(obj, ["bill_date","issued_date","billDate","statementDate"])
    )
    currency = _pick(obj, ["currency"], "INR")
    account_id = _digits_only(_pick(obj, ["customer_id","customerAccountNumber","account_number","account_id"]))
    meter_id = (
        _get(obj, ["meter","number"]) or
        _get(obj, ["bill_metadata","meter_number"]) or
        _get(obj, ["infrastructure_details","meter_number"]) or
        _pick(obj, ["meterNumber"])
    )

    total_amount = _pick(obj, ["total_amount","net_amount_payable","netAmountPayable","billAmount","amountDue","invoiceTotal"], 0.0)

    total_kwh = (
        _safe_float(_pick(obj, ["total_kwh","energyUsage","total_consumption_kwh","totalConsumptionKwh","consumptionKwh"], 0.0)) or
        _safe_float(_get(obj, ["meter","billed_consumption_kwh"]), 0.0) or
        _safe_float(_get(obj, ["meter","consumption_interval_kwh"]), 0.0) or
        _safe_float(_get(obj, ["consumption","total_consumed_kwh"]), 0.0)
    )

    lines = []
    charges = obj.get("charges") or {}
    if isinstance(charges, dict):
        if "total_energy_charges" in charges or "energy" in charges:
            lines.append({"component_type":"energy","description":"Energy charges",
                          "amount": _safe_float(charges.get("total_energy_charges", charges.get("energy", 0.0)))})
        if "fixed_charges" in charges or "fixed" in charges:
            lines.append({"component_type":"fixed","description":"Fixed charges",
                          "amount": _safe_float(charges.get("fixed_charges", charges.get("fixed", 0.0)))})
        if "electricity_tax" in charges:
            lines.append({"component_type":"tax","description":"Electricity tax",
                          "amount": _safe_float(charges.get("electricity_tax", 0.0))})
        other_sum = 0.0
        for k in ("ppac","total_ppac_energy_charge","pension_surcharge","other_charges","security_deposit"):
            if k in charges:
                other_sum += _safe_float(charges.get(k), 0.0)
        if other_sum:
            lines.append({"component_type":"other","description":"Other surcharges/fees","amount": round(other_sum, 2)})
    if not lines:
        lines = [{"component_type":"other","description":"Unspecified","amount": _safe_float(total_amount)}]

    out = {
        "billing_statements": [{
            "account_id": account_id,
            "service_point_id": meter_id,
            "billing_period_start": start,
            "billing_period_end": end,
            "statement_date": statement_date,
            "subtotal": None,
            "tax": None,
            "total": _safe_float(total_amount),
            "currency": currency or "INR",
            "rate_plan": _pick(obj, ["tariff_category","tariffCategory"]),
            "lines": lines
        }],
        "interval_readings": []
    }
    if start and end and total_kwh > 0:
        out["interval_readings"].append({
            "meter_id": meter_id,
            "ts_start_utc": f"{start}T00:00:00Z",
            "ts_end_utc":   f"{end}T23:59:59Z",
            "kwh": _safe_float(total_kwh),
            "quality": "A"
        })
    return out
# --- END: simple envelope coercer ---
def normalize_canonical_envelope_shapes(obj: dict) -> dict:
    """
    If top-level keys are present but inner objects use alternative keys,
    map them to our canonical model fields.
    """
    # Fix billing_statements dates if they include time or different keys
    if isinstance(obj.get("billing_statements"), list):
        for bs in obj["billing_statements"]:
            if isinstance(bs, dict):
                # ensure date-only strings
                for k in ("billing_period_start","billing_period_end","statement_date"):
                    if k in bs and isinstance(bs[k], str) and len(bs[k]) >= 10:
                        bs[k] = bs[k][:10]
                # minimal defaults
                bs.setdefault("currency", "INR")
                bs.setdefault("lines", [])

    # Fix interval_readings items from {"start","end","value"} → {"ts_start_utc","ts_end_utc","kwh"}
    if isinstance(obj.get("interval_readings"), list):
        fixed = []
        for ir in obj["interval_readings"]:
            if not isinstance(ir, dict):
                continue
            if "ts_start_utc" in ir and "ts_end_utc" in ir and "kwh" in ir:
                fixed.append(ir)
                continue
            start = ir.get("start") or ir.get("ts_start") or ir.get("from")
            end   = ir.get("end")   or ir.get("ts_end")   or ir.get("to")
            kwh   = ir.get("value") or ir.get("kwh")      or ir.get("consumption") or ir.get("energy")
            if start and end and kwh is not None:
                # produce UTC datetimes from dates if needed
                s = start[:10] + ("T00:00:00Z" if len(start) == 10 else "")
                e = end[:10]   + ("T23:59:59Z" if len(end) == 10 else "")
                fixed.append({
                    "meter_id": ir.get("meter_id"),
                    "ts_start_utc": s,
                    "ts_end_utc": e,
                    "kwh": float(kwh),
                    "quality": ir.get("quality","A")
                })
        obj["interval_readings"] = fixed
    return obj




app = FastAPI(title="Greenchain Ingestion Runner", version="0.1.0")

def sha256_bytes(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()

@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    content = await file.read()
    text = extract_text_from_pdf(content)
    client = ASIClient()
    schema = envelope_json_schema_for_asi()
    prompt = (
    "Convert the following electricity bill text to the Greenchain envelope.\n"
    "Return ONLY valid JSON with EXACTLY these top-level keys: billing_statements, interval_readings.\n"
    "billing_statements[0] fields:\n"
    "- billing_period_start: YYYY-MM-DD (no time)\n"
    "- billing_period_end: YYYY-MM-DD (no time)\n"
    "- statement_date: YYYY-MM-DD\n"
    "- total: number (net amount payable)\n"
    "- currency: 'INR' if Indian bill else 3-letter code\n"
    "interval_readings: include a single row only if total kWh is known for the period.\n"
    "If included, the row MUST be an object with 'ts_start_utc', 'ts_end_utc', 'kwh'.\n"
    "Use 'YYYY-MM-DDT00:00:00Z' and 'YYYY-MM-DDT23:59:59Z' for start/end.\n"
    "Do not include any other top-level keys. No code fences. No explanations.\n\n"
    "PDF TEXT START\n"
    f"{text}\n"
    "PDF TEXT END\n"
)


    extracted = client.chat_structured(prompt, schema)
    extracted = coerce_any_to_envelope(extracted)
    extracted = normalize_canonical_envelope_shapes(extracted) 
    src = SourceInfo(source_type="pdf", filename=file.filename, sha256=sha256_bytes(content), extra={})
    env = Envelope(source=src)
    if "billing_statements" in extracted:
        env.billing_statements = [BillingStatement(**bs) for bs in extracted["billing_statements"]]
    if "interval_readings" in extracted:
        env.interval_readings = [IntervalReading(**ir) for ir in extracted["interval_readings"]]
    env.close()
    return env.model_dump(mode="json")

@app.post("/ingest/excel")
async def ingest_excel(file: UploadFile = File(...)):
    content = await file.read()
    # Convert Excel/CSV to text per spec
    body_text = ""
    name = file.filename.lower()
    if name.endswith(".csv"):
        body_text = content.decode("utf-8", errors="ignore")
    else:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp.write(content); tmp.flush()
            xl = pd.ExcelFile(tmp.name)
            parts = []
            for sheet in xl.sheet_names:
                df = xl.parse(sheet)
                parts.append(f"### SHEET: {sheet} ###")
                parts.append(df.to_csv(index=False))
            body_text = "\n".join(parts)

    client = ASIClient()
    schema = envelope_json_schema_for_asi()
    prompt = (
        "You are converting an energy meter export (CSV or Excel converted to CSV text) into Greenchain's "
        "canonical JSON. The file likely contains interval meter readings (kWh consumption) and sometimes a billed total. "
        "Return ONLY JSON that matches the provided schema. Use UTC for timestamps; infer ts_end_utc if only a single timestamp exists per row "
        "by adding the dataset's interval length. If column names vary (e.g., energy, consumption, import_kwh), interpret sensibly.\n\n"
        "CSV TEXT START\n"
        f"{body_text[:150000]}\n"
        "CSV TEXT END\n"
    )
    extracted = client.chat_structured(prompt, schema)
    extracted = coerce_any_to_envelope(extracted)
    extracted = normalize_canonical_envelope_shapes(extracted) 
    src = SourceInfo(source_type="excel", filename=file.filename, sha256=sha256_bytes(content), extra={})
    env = Envelope(source=src)
    if "interval_readings" in extracted:
        env.interval_readings = [IntervalReading(**ir) for ir in extracted["interval_readings"]]
    if "billing_statements" in extracted:
        env.billing_statements = [BillingStatement(**bs) for bs in extracted["billing_statements"]]
    env.close()
    return env.model_dump(mode="json")

@app.get("/gcp/co2")
async def gcp_co2(hours: float = Query(1.0), avg_util: float = Query(0.3)):
    # Delegate to minimal in-process calculator (no external deps)
    from .scripts.gcp_co2 import estimate_kwh, REGION_GRID_INTENSITY, PUE
    # Context discovery
    import requests
    def meta(path: str):
        url = f"http://metadata.google.internal/computeMetadata/v1/{path}"
        headers = {"Metadata-Flavor": "Google"}
        r = requests.get(url, headers=headers, timeout=1.0)
        r.raise_for_status()
        return r.text
    try:
        project_id = meta("project/project-id")
        zone_path = meta("instance/zone")
        zone = zone_path.split("/")[-1]
        region = "-".join(zone.split("-")[:2])
        machine_type_path = meta("instance/machine-type")
        machine_type = machine_type_path.split("/")[-1]
    except Exception:
        return JSONResponse({"error": "Not running on GCE (metadata unavailable)"}, status_code=400)

    # Specs via API or env fallback
    try:
        from .scripts.gcp_co2 import get_machine_specs
        specs = get_machine_specs(project_id, zone, machine_type)
    except Exception:
        specs = {"vcpus": int(os.getenv("GC_VCPU", "2")), "memory_gib": float(os.getenv("GC_MEM_GIB", "4"))}

    intensity = REGION_GRID_INTENSITY.get(region, 400.0)
    kwh = estimate_kwh(hours=hours, vcpus=specs["vcpus"], mem_gib=specs["memory_gib"], cpu_util=avg_util)
    co2e_kg = kwh * (intensity / 1000.0)

    payload = {
        "schema_version": "greenchain.ingest.v1",
        "source": {"source_type": "gcp", "filename": None, "sha256": None, "received_at": None, "extra": {"project_id": project_id, "zone": zone, "region": region, "machine_type": machine_type}},
        "gcp_footprint": {
            "region": region,
            "hours": hours,
            "vcpu_count": specs["vcpus"],
            "memory_gib": round(specs["memory_gib"], 2),
            "cpu_utilization": avg_util,
            "pue": PUE,
            "grid_intensity_gco2_per_kwh": intensity,
            "estimated_kwh": round(kwh, 6),
            "estimated_co2e_kg": round(co2e_kg, 6),
            "method": "CCF-inspired compute+memory energy × PUE × grid factor",
            "references": [
                "https://docs.cloud.google.com/sustainability/region-carbon",
                "https://www.cloudcarbonfootprint.org/docs/methodology/"
            ]
        }
    }
    return JSONResponse(payload)

if __name__ == "__main__":
    uvicorn.run("runner:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
