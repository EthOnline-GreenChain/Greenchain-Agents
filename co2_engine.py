#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, sys, json, math, uuid, time, argparse, dataclasses
from typing import Any, Dict, Optional, List, Tuple
import requests

# =========================
# ASI Agentic client
# =========================
ASI_API_KEY = os.getenv("ASI_ONE_API_KEY", "")
ASI_ENDPOINT = "https://api.asi1.ai/v1/chat/completions"
ASI_MODEL = "asi1-fast-agentic"
ASI_TIMEOUT = 1000
_SESSION_MAP: Dict[str, str] = {}

def _asi_session_id(conv_id: str) -> str:
    sid = _SESSION_MAP.get(conv_id)
    if sid is None:
        sid = str(uuid.uuid4())
        _SESSION_MAP[conv_id] = sid
    return sid

def _extract_first_json(text: str) -> Dict[str, Any]:
    """Extract the first balanced JSON object from a possibly chatty response."""
    if not text:
        raise ValueError("Empty response from ASI")
    text = text.strip()
    # Fast path
    try:
        return json.loads(text)
    except Exception:
        pass
    # Slow path: scan for first balanced {...}
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found in ASI response")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    return json.loads(candidate)
    raise ValueError("Could not find balanced JSON object in ASI response")

def asi_chat_json(conv_id: str, system_prompt: str, user_content_obj: Dict[str, Any], *, verbose=False) -> Dict[str, Any]:
    """Call ASI and return a parsed JSON object, robust to chatty output."""
    headers = {
        "Authorization": f"Bearer {ASI_API_KEY}",
        "x-session-id": _asi_session_id(conv_id),
        "Content-Type": "application/json",
    }
    payload = {
        "model": ASI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_content_obj, ensure_ascii=False)}
        ],
        # Many providers accept this OpenAI-style hint. If ignored, we still parse safely.
        "response_format": {"type": "json_object"},
        "stream": False
    }
    try:
        resp = requests.post(ASI_ENDPOINT, headers=headers, json=payload, timeout=ASI_TIMEOUT)
        if verbose:
            print(f"[ASI status] {resp.status_code}", file=sys.stderr)
        resp.raise_for_status()
        resp_json = resp.json()
        content = resp_json["choices"][0]["message"]["content"]
        if verbose:
            print("[ASI raw]", content[:500], "..." if len(content) > 500 else "", file=sys.stderr)
        if not content or not content.strip():
            raise ValueError("ASI returned empty content")
        return _extract_first_json(content)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"ASI HTTP error: {e}") from e
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"ASI response shape unexpected: {resp.text[:400]}") from e
    except ValueError as e:
        raise RuntimeError(f"ASI parsing error: {e}. Content: {content[:200] if 'content' in locals() else 'N/A'}") from e
    except Exception as e:
        raise RuntimeError(f"ASI parsing error: {e}") from e

# =========================
# Models and config
# =========================
@dataclasses.dataclass
class ReportingConfig:
    year: int
    country: str
    state_or_region: Optional[str] = None
    industry: Optional[str] = None
    method: str = "auto"                   # "location","market","auto"
    standard: str = "GHG-Protocol-Scope2"
    include_t_and_d_scope3: bool = False
    t_and_d_pct_override: Optional[float] = None
    supplier_specific_factor_kg_per_kwh: Optional[float] = None
    residual_mix_factor_kg_per_kwh: Optional[float] = None
    recs_mwh: float = 0.0
    ppa_kwh: float = 0.0

@dataclasses.dataclass
class FactorResult:
    kg_per_kwh: float
    source_name: str
    source_url: str
    version: str
    region_key: str
    notes: Optional[str] = None
    citations: Optional[List[str]] = None
    t_and_d_loss_fraction: Optional[float] = None

# Optional seeds for resilience if ASI is unreachable.
SEED_FACTORS: Dict[Tuple[str,int], float] = {
    # Conservative placeholders. Prefer ASI resolution always.
    ("IN", 2024): 0.757,    # CEA v20 combined margin (kgCO2e/kWh)
    ("UK", 2024): 0.22535,  # UK Gov 2024 grid electricity (kgCO2e/kWh)
}
SEED_META: Dict[Tuple[str,int], Dict[str,str]] = {
    ("IN", 2024): {
        "source_name": "India CEA CO2 Baseline Database v20",
        "source_url": "https://cea.nic.in/cdm-co2-baseline-database/",
        "version": "v20 (2024)",
        "region_key": "India-grid-combined-margin",
    },
    ("UK", 2024): {
        "source_name": "UK Government GHG Conversion Factors 2024",
        "source_url": "https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting",
        "version": "2024",
        "region_key": "UK-national",
    },
}

ASI_FACTOR_SYSTEM = """You are an expert carbon accounting analyst. Return ONLY a compact JSON object.
Fields: kg_per_kwh (number), source_name, source_url, version, region_key, t_and_d_loss_fraction (number or null), citations (array of 1-4 URLs).
Pick authoritative electricity grid factor for the given country, state/region, year, and standard.
If state/region is US, use appropriate eGRID region and include T&D loss if available.
NEVER MISS kg_per_kwh
"""

ASI_INFER_LOCALE_SYSTEM = """You are an expert data profiler. Return ONLY JSON.
Given a parsed electricity-bill ingestion JSON, infer:
- country: 2-letter ISO country code like "IN","US","UK" (or best guess)
- state_or_region: optional best-guess subregion (state, province, or grid subregion), or null
Consider filename text, currency, date formats, utility names, addresses, and any hints.
Fields: country, state_or_region.
"""

# =========================
# Factor resolution
# =========================
def resolve_factor_via_asi(cfg: ReportingConfig, *, verbose=False) -> FactorResult:
    conv_id = f"co2e-factor-{uuid.uuid4()}"
    print(cfg)
    method = "location-based" if cfg.method in ("location", "auto") else "market-based"
    user_payload = {
        "country": cfg.country,
        "state_or_region": cfg.state_or_region,
        "year": cfg.year,
        "method": method,
        "standard": cfg.standard
    }
    try:
        data = asi_chat_json(conv_id, ASI_FACTOR_SYSTEM, user_payload, verbose=verbose)
        
        # Validate required fields
        kg_per_kwh_value = data.get("kg_per_kwh")
        if kg_per_kwh_value is None:
            raise ValueError("ASI returned null or missing kg_per_kwh")
        if not data.get("source_name"):
            raise ValueError("ASI returned null or missing source_name")
        if not data.get("source_url"):
            raise ValueError("ASI returned null or missing source_url")
        
        return FactorResult(
            kg_per_kwh=float(kg_per_kwh_value),
            source_name=str(data["source_name"]),
            source_url=str(data["source_url"]),
            version=str(data.get("version","")),
            region_key=str(data.get("region_key","")),
            notes=None,
            citations=list(data.get("citations", [])) or None,
            t_and_d_loss_fraction=(float(data["t_and_d_loss_fraction"]) if data.get("t_and_d_loss_fraction") is not None else None),
        )
    except Exception as e:
        # Seed fallback
        key = (cfg.country.upper(), cfg.year)
        if key in SEED_FACTORS and key in SEED_META:
            meta = SEED_META[key]
            return FactorResult(
                kg_per_kwh=float(SEED_FACTORS[key]),
                source_name=meta["source_name"],
                source_url=meta["source_url"],
                version=meta["version"],
                region_key=meta["region_key"],
                notes=f"Seed fallback due to ASI error: {e}",
                citations=[meta["source_url"]],
                t_and_d_loss_fraction=None
            )
        raise

# =========================
# Scope math
# =========================
def compute_scope2_and_tandd(kwh_total: float, cfg: ReportingConfig, factor: FactorResult) -> Dict[str, Any]:
    scope2_location_tonnes = (kwh_total * factor.kg_per_kwh) / 1000.0

    mb_tonnes = None
    if cfg.method == "market":
        if cfg.supplier_specific_factor_kg_per_kwh is not None:
            kwh_market = max(0.0, kwh_total - (cfg.recs_mwh * 1000.0) - cfg.ppa_kwh)
            mb_tonnes = (kwh_market * cfg.supplier_specific_factor_kg_per_kwh) / 1000.0
        else:
            f = cfg.residual_mix_factor_kg_per_kwh or factor.kg_per_kwh
            mb_tonnes = (kwh_total * f) / 1000.0

    tandd_tonnes = None
    tandd_fraction = cfg.t_and_d_pct_override if cfg.t_and_d_pct_override is not None else factor.t_and_d_loss_fraction
    if cfg.include_t_and_d_scope3 and tandd_fraction:
        extra_kwh = kwh_total * float(tandd_fraction)
        tandd_tonnes = (extra_kwh * factor.kg_per_kwh) / 1000.0

    return {
        "scope2_location_tonnes": round(scope2_location_tonnes, 6),
        "scope2_market_tonnes": (round(mb_tonnes, 6) if mb_tonnes is not None else None),
        "scope3_3_tandd_tonnes": (round(tandd_tonnes, 6) if tandd_tonnes is not None else None),
        "factor_used": dataclasses.asdict(factor),
    }

# =========================
# IO helpers
# =========================
def load_json(path: Optional[str]) -> Dict[str, Any]:
    if path and path != "-":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.load(sys.stdin)

def save_json(obj: Dict[str, Any], path: Optional[str]):
    out = json.dumps(obj, ensure_ascii=False, indent=2)
    if path and path != "-":
        with open(path, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        print(out)

# =========================
# Locale inference
# =========================
def infer_country_state_via_asi(full_input_json: Dict[str, Any], *, verbose=False) -> Tuple[str, Optional[str], bool, Optional[str]]:
    """Return (country, state_or_region, used_llm, llm_reason_note)."""
    try:
        conv_id = f"co2e-locale-{uuid.uuid4()}"
        data = asi_chat_json(conv_id, ASI_INFER_LOCALE_SYSTEM, {"input": full_input_json}, verbose=verbose)
        country = str(data.get("country") or "").upper()
        region = data.get("state_or_region")
        if country:
            return country, (str(region) if region else None), True, "Inferred by ASI from input JSON"
    except Exception as e:
        return "IN", None, False, f"LLM inference failed: {e}"

    # Fallback
    return "IN", None, False, "LLM inference returned empty code"

def infer_country_state_heuristic(input_json: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    country = "IN"
    state = None
    try:
        cur = (input_json.get("billing_statements") or [{}])[0].get("currency")
        if cur == "USD":
            country = "US"
        elif cur == "GBP":
            country = "UK"
        elif cur == "INR":
            country = "IN"
    except Exception:
        pass
    return country, state

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="GreenChain CO2e Engine")
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSON path or - for stdin")
    ap.add_argument("--out", dest="out_path", default="-", help="Output JSON path or - for stdout")
    ap.add_argument("--country", type=str, help="Override country code, e.g., US, UK, IN")
    ap.add_argument("--region", type=str, help="Override state/region")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--method", type=str, choices=["auto","location","market"], default="auto")
    ap.add_argument("--standard", type=str, default="GHG-Protocol-Scope2")
    ap.add_argument("--industry", type=str, default=None)
    ap.add_argument("--include-tandd", action="store_true", help="Report Scope 3.3 T&D losses separately")
    ap.add_argument("--tandd-pct", type=float, default=None, help="Override T&D loss fraction, e.g., 0.043 for 4.3%")
    ap.add_argument("--supplier-factor", type=float, default=None, help="Supplier-specific kgCO2e/kWh for market-based")
    ap.add_argument("--residual-mix", type=float, default=None, help="Residual mix kgCO2e/kWh for market-based")
    ap.add_argument("--recs-mwh", type=float, default=0.0, help="EACs/RECs volume in MWh")
    ap.add_argument("--ppa-kwh", type=float, default=0.0, help="PPA-covered kWh")
    ap.add_argument("--verbose", action="store_true", help="Print debug info to stderr")
    args = ap.parse_args()

    input_json = load_json(args.in_path)
    intervals = input_json.get("interval_readings") or []
    kwh_total = sum(float(r.get("kwh") or 0.0) for r in intervals)

    # 1) LLM-based country/region inference first, as you asked
    country_llm, region_llm, used_llm, llm_note = infer_country_state_via_asi(input_json, verbose=args.verbose)

    # 2) Override chain
    country = country_llm
    state_or_region = region_llm
    if args.country:
        country = args.country
    if args.region:
        state_or_region = args.region

    # 3) Heuristic fallback if LLM inference returned empty values
    if not country:
        country, _fallback_state = infer_country_state_heuristic(input_json)

    cfg = ReportingConfig(
        year=args.year,
        country=country,
        state_or_region=state_or_region,
        industry=args.industry,
        method=args.method,
        standard=args.standard,
        include_t_and_d_scope3=bool(args.include_tandd),
        t_and_d_pct_override=args.tandd_pct,
        supplier_specific_factor_kg_per_kwh=args.supplier_factor,
        residual_mix_factor_kg_per_kwh=args.residual_mix,
        recs_mwh=args.recs_mwh,
        ppa_kwh=args.ppa_kwh,
    )

    # Resolve factor via ASI with robust parsing
    factor = resolve_factor_via_asi(cfg, verbose=args.verbose)

    # Compute
    result = compute_scope2_and_tandd(kwh_total, cfg, factor)

    # Build footprint
    gcp = {
        "reporting": {
            "standard": cfg.standard,
            "year": cfg.year,
            "country": cfg.country,
            "state_or_region": cfg.state_or_region,
            "industry": cfg.industry,
            "method": cfg.method,
            "include_t_and_d_scope3": cfg.include_t_and_d_scope3,
        },
        "inputs": {
            "kwh_total": kwh_total,
            "interval_count": len(intervals),
        },
        "outputs": result,
        "diagnostics": {
            "locale_inference": {
                "used_llm": used_llm,
                "note": llm_note,
                "country_inferred": country_llm,
                "state_or_region_inferred": region_llm,
            },
            "ts_generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    }

    input_json["co2_footprint"] = gcp
    save_json(input_json, args.out_path)

if __name__ == "__main__":
    main()