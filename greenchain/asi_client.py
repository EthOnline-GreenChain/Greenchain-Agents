from __future__ import annotations

import os, sys, json, re, requests
from typing import Any, Dict, Optional, List

ASI_BASE_URL = os.getenv("ASI_ONE_BASE_URL", "https://api.asi1.ai/v1")
ASI_API_KEY  = os.getenv("ASI_ONE_API_KEY", "")
ASI_MODEL    = os.getenv("ASI_ONE_MODEL", "asi1-extended")
ASI_DEBUG    = os.getenv("ASI_ONE_DEBUG", "0") == "1"

class ASIClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None, timeout: int = 90):
        self.api_key = api_key or ASI_API_KEY
        if not self.api_key:
            raise RuntimeError("ASI_ONE_API_KEY is required")
        self.base_url = base_url or ASI_BASE_URL
        self.model = model or ASI_MODEL
        self.timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # ----------------------------
    # Helpers to robustly parse JSON
    # ----------------------------
    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Given any text, try to extract the largest valid JSON object."""
        text = text.strip()

        # 1) If it's plain JSON, parse directly
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except Exception:
                pass

        # 2) Code fences ```json ... ```
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
        if fence:
            try:
                return json.loads(fence.group(1))
            except Exception:
                pass

        # 3) Best-effort: first { ... last } slice
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidate = text[first:last+1]
            try:
                return json.loads(candidate)
            except Exception:
                # try to strip trailing commentary after closing braces
                m = re.search(r"(\{.*\})", candidate, flags=re.S)
                if m:
                    try:
                        return json.loads(m.group(1))
                    except Exception:
                        pass
        return None

    def _parse_structured_choice(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """OpenAI-compatible parsing with several fallbacks."""
        choices = data.get("choices") or []
        print(choices)
        if not choices:
            return None
        msg = (choices[0] or {}).get("message") or {}

        # 1) message.content as str
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            obj = self._extract_json_from_text(content)
            if obj is not None:
                return obj

        # 2) message.content as list of parts with {text: ...}
        if isinstance(content, list):
            buf: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    if isinstance(part.get("text"), str):
                        buf.append(part["text"])
                    elif isinstance(part.get("content"), str):
                        buf.append(part["content"])
            if buf:
                obj = self._extract_json_from_text("".join(buf))
                if obj is not None:
                    return obj

        # 3) tool_calls[].function.arguments as str JSON
        for tc in msg.get("tool_calls") or []:
            fn = (tc or {}).get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str) and args.strip():
                obj = self._extract_json_from_text(args)
                if obj is not None:
                    return obj

        # 4) already-parsed
        for k in ("parsed",):
            obj = msg.get(k) or (choices[0] or {}).get(k) or data.get(k)
            if isinstance(obj, dict):
                return obj

        return None

    # ----------------------------
    # Shape normalizer → our schema
    # ----------------------------
    def _normalize_to_envelope_schema(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert provider-specific shapes (wrapped or flat) into:
        {"billing_statements":[...], "interval_readings":[...]}
        """
        # Already canonical
        if "billing_statements" in obj or "interval_readings" in obj:
            return obj

        def _first(d: Dict[str, Any], keys, default=None):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return default

        def _safe_float(x, default=0.0):
            try:
                return float(x)
            except Exception:
                return default

        # Case 1: wrapped { "billingStatement": {...} }
        if isinstance(obj.get("billingStatement"), dict):
            bs = obj["billingStatement"]
            bp = bs.get("billingPeriod", {}) or {}
            start = _first(bs, ["billingPeriodStart"], None) or bp.get("startDate")
            end   = _first(bs, ["billingPeriodEnd"], None) or bp.get("endDate")
            pay   = bs.get("paymentSummary", {}) or {}
            total_amount = _first(pay, ["netAmountPayable", "totalAmount", "billAmount", "amountDue", "invoiceTotal"], 0.0)
            currency = _first(bs, ["currency"], "INR")

            lines = []
            charges = bs.get("chargesBreakdown", {}) or {}
            if "energyCharges" in charges:
                lines.append({"component_type":"energy","description":"Energy charges","amount":_safe_float(charges["energyCharges"].get("amount",0))})
            if "fixedCharges" in charges:
                lines.append({"component_type":"fixed","description":"Fixed charges","amount":_safe_float(charges["fixedCharges"].get("amount",0))})
            if "electricityTax" in charges:
                amt = charges["electricityTax"].get("amount", charges["electricityTax"].get("value", 0))
                lines.append({"component_type":"tax","description":"Electricity tax","amount":_safe_float(amt)})
            extras_total = 0.0
            for k in ("pensionSurcharge","surcharge","securityDeposit"):
                v = charges.get(k)
                if isinstance(v, dict):
                    extras_total += _safe_float(v.get("amount", 0))
                elif isinstance(v, (int,float,str)):
                    extras_total += _safe_float(v, 0)
            if extras_total:
                lines.append({"component_type":"other","description":"Other surcharges/fees","amount":round(extras_total,2)})

            meter_id = _first(bs.get("serviceInformation",{}).get("meterDetails",{}) or {}, ["meterNumber"])
            total_kwh = _safe_float(bs.get("consumptionSummary", {}).get("totalConsumptionKwh", 0.0))

            out = {
                "billing_statements": [{
                    "account_id": bs.get("customerAccountNumber"),
                    "service_point_id": meter_id,
                    "billing_period_start": start,
                    "billing_period_end": end,
                    "statement_date": bs.get("billDate"),
                    "subtotal": None,
                    "tax": None,
                    "total": _safe_float(total_amount),
                    "currency": currency,
                    "rate_plan": _first(bs.get("serviceInformation",{}) or {}, ["tariffCategory"]),
                    "lines": lines or [{"component_type":"other","description":"Unspecified","amount":_safe_float(total_amount)}],
                }],
                "interval_readings": []
            }
            if start and end and total_kwh > 0:
                out["interval_readings"].append({
                    "meter_id": meter_id,
                    "ts_start_utc": f"{start}T00:00:00Z",
                    "ts_end_utc":   f"{end}T23:59:59Z",
                    "kwh": round(total_kwh, 6),
                    "quality": "A"
                })
            return out

        # Case 2: flat bill object (your sample)
        looks_flat = any(k in obj for k in [
            "billDate","billingPeriodStart","billingPeriodEnd","billingCompany",
            "customerAccountNumber","energyUsage","energyBreakdown"
        ])
        if looks_flat:
            start = _first(obj, ["billingPeriodStart", "startDate"])
            end   = _first(obj, ["billingPeriodEnd", "endDate"])
            currency = _first(obj, ["currency"], "INR")
            total_amount = _first(obj, ["netAmountPayable","billAmount","amountDue","invoiceTotal"], 0.0)

            # Build lines
            lines = []
            # energyBreakdown has usage*rate; compute if present
            energy_amount = 0.0
            for r in obj.get("energyBreakdown", []) or []:
                usage = _safe_float(r.get("usage", 0))
                rate  = _safe_float(r.get("rate", 0))
                if usage and rate:
                    energy_amount += usage * rate
            if energy_amount > 0:
                lines.append({"component_type":"energy","description":"Slab energy charges","amount":round(energy_amount,2)})

            if "fixedChargesTotal" in obj:
                lines.append({"component_type":"fixed","description":"Fixed charges","amount":_safe_float(obj.get("fixedChargesTotal",0))})

            if not lines and _safe_float(total_amount) > 0:
                lines.append({"component_type":"other","description":"Unspecified","amount":_safe_float(total_amount)})

            # Interval readings: one coarse period row if we have total kWh
            total_kwh = _safe_float(_first(obj, ["energyUsage","totalConsumptionKwh","consumptionKwh"], 0.0))
            meter_id = obj.get("meterNumber")  # may be absent in flat shape

            out = {
                "billing_statements": [{
                    "account_id": obj.get("customerAccountNumber"),
                    "service_point_id": meter_id,
                    "billing_period_start": start,
                    "billing_period_end": end,
                    "statement_date": obj.get("billDate") or obj.get("statementDate"),
                    "subtotal": None,
                    "tax": None,
                    "total": _safe_float(total_amount),
                    "currency": currency,
                    "rate_plan": obj.get("tariffCategory"),
                    "lines": lines
                }],
                "interval_readings": []
            }
            if start and end and total_kwh > 0:
                out["interval_readings"].append({
                    "meter_id": meter_id,
                    "ts_start_utc": f"{start}T00:00:00Z",
                    "ts_end_utc":   f"{end}T23:59:59Z",
                    "kwh": round(total_kwh, 6),
                    "quality": "A"
                })
            return out

        # Unknown shape → return as-is; caller may decide
        return obj


    # ----------------------------
    # Public API
    # ----------------------------
    def chat_structured(
        self,
        prompt: str,
        json_schema: Dict[str, Any],
        system: str = "Return ONLY valid JSON matching the provided schema."
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            # Ask for json_schema, but some hosts still return custom shapes
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "greenchain_ingestion", "strict": True, "schema": json_schema}
            },
            "temperature": 0,
            "top_p": 1,
            "n": 1,
            "stream": False,
            "max_tokens": 50000,
        }
        url = f"{self.base_url}/chat/completions"
        resp = requests.post(url, headers=self._headers, json=payload, timeout=self.timeout)
        if not resp.ok:
            if ASI_DEBUG:
                print(f"[ASI DEBUG] HTTP {resp.status_code} {resp.text}", file=sys.stderr)
            raise RuntimeError(f"ASI request failed: {resp.status_code} {resp.text}")

        data = resp.json()

        # Try to parse JSON from the choice in multiple ways
        parsed = self._parse_structured_choice(data)
        if parsed is None:
            if ASI_DEBUG:
                print("[ASI DEBUG] Raw response (truncated):", file=sys.stderr)
                try:
                    print(json.dumps(data, indent=2)[:4000], file=sys.stderr)
                except Exception:
                    print(str(data)[:4000], file=sys.stderr)
            raise RuntimeError("ASI returned no structured content")

        # Normalize alt shapes to our Envelope keys
        normalized = self._normalize_to_envelope_schema(parsed)
        return normalized
