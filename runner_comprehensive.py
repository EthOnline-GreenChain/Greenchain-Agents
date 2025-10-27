#!/usr/bin/env python3
"""
GreenChain Comprehensive Integrated Runner
===========================================
Complete platform combining ingestion, CO2 engine, and procurement uAgents.
"""



# Initialize working directory and PYTHONPATH so local packages are discoverable
import os, sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))

# Standard library
import io
import json
import uuid
import base64
import hashlib
import time as time_module
import requests

# Third-party
import pandas as pd
import uvicorn
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, Body
from fastapi.responses import JSONResponse, FileResponse

# Ingestion
from greenchain.schema import Envelope, SourceInfo, BillingStatement, IntervalReading, envelope_json_schema_for_asi
from greenchain.pdf_text import extract_text_from_pdf
from greenchain.asi_client import ASIClient

# CO2 engine
from co2_engine import (
    ReportingConfig,
    resolve_factor_via_asi,
    compute_scope2_and_tandd,
    infer_country_state_via_asi,
    infer_country_state_heuristic,
)

# Messaging
from uagents import Bureau, Agent
from uagents_core.envelope import Envelope as MsgEnvelope

# Procurement orchestration (from procurement package)
from procurement import (
    build_curator,
    build_provider_in,
    build_provider_np,
    build_payment,
    build_retire,
    build_bundler,
    FootprintIntent,
    FOOTPRINT_PROTOCOL_DIGEST,
    FOOTPRINT_SCHEMA_DIGEST,
    INTENT_PROTOCOL,
)


# Import envelope coercion utilities
try:
    from runner import coerce_any_to_envelope, normalize_canonical_envelope_shapes, sha256_bytes
except ImportError:
    # Fallback implementations if runner module not available
    def coerce_any_to_envelope(data):
        return data
    def normalize_canonical_envelope_shapes(data):
        return data
    def sha256_bytes(content):
        return hashlib.sha256(content).hexdigest()

# ============================================================================
# Configuration
# ============================================================================
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))
BUREAU_PORT = int(os.getenv("BUREAU_PORT", "8001"))
TEMP_DATA_DIR = Path("/tmp/greenchain_data")
TEMP_DATA_DIR.mkdir(exist_ok=True)



from decimal import Decimal, InvalidOperation

def as_decimal(x, default=Decimal("0")) -> Decimal:
    if x is None:
        return default
    try:
        # handles int, float, Decimal, and strings like "1,234.56" or "nan"
        s = str(x).strip().replace(",", "")
        if s.lower() in {"", "none", "null", "nan"}:
            return default
        return Decimal(s)
    except (InvalidOperation, ValueError, TypeError):
        return default


# ============================================================================
# Procurement Bureau Management
# ============================================================================

class ProcurementOrchestrator:
    """
    Manages the uAgents Bureau for carbon credit procurement.
    Handles agent initialization, request tracking, and status monitoring.
    """
    
    def __init__(self):
        self.bureau: Optional[Bureau] = None
        self.is_running = False
        self.curator = None
        self.agents = {}
        # Track procurement requests
        self.procurement_requests: Dict[str, Dict[str, Any]] = {}
        
    def initialize_bureau(self):
        """Initialize all procurement agents and create Bureau"""
        print("\n" + "=" * 80)
        print("Initializing Procurement Bureau")
        print("=" * 80)
        
        try:
            # Build all agents
            prov_in = build_provider_in(name="ProviderIN", seed="seed provider in 1/2/3")
            prov_np = build_provider_np(name="ProviderNP", seed="seed provider np 1/2/3")
            payment = build_payment(name="PaymentOrchestrator", seed="seed payment 1/2/3")
            registrar = build_retire(name="RetirementRegistrar", seed="seed retire 1/2/3")
            bundler = build_bundler(name="ProofBundler", seed="seed bundler 1/2/3")
            
            # Build curator without demo intent (will be triggered via API)
            curator = build_curator(
                name="BasketCurator",
                seed="seed curator 1/2/3",
                provider_addresses=[prov_in.address, prov_np.address],
                payment_orchestrator_addr=payment.address,
                retirement_registrar_addr=registrar.address,
                proof_bundler_addr=bundler.address,
                demo_intent=None,  # No demo intent - triggered via API
            )
            
            bridge = Agent(name="ApiBridge", seed="seed api bridge 1/2/3")

            self.agents = {
                "curator": curator,
                "prov_in": prov_in,
                "prov_np": prov_np,
                "payment": payment,
                "registrar": registrar,
                "bundler": bundler,
                "bridge": bridge,   # NEW
            }
            # Validation
            for name, agent in self.agents.items():
                if agent is None:
                    raise RuntimeError(f"{name} built as None")
                if not isinstance(agent, Agent):
                    raise RuntimeError(f"{name} is not a uAgents.Agent. Got {type(agent)}")
                print(f"[OK] {name}: {agent.address}")
            
            # Create Bureau
            self.bureau = Bureau(
                port=BUREAU_PORT,
                agents=list(self.agents.values()),
                endpoint=f"http://0.0.0.0:{BUREAU_PORT}/submit",
            )
            
            # Store curator for later use
            self.curator = curator
            print(f"\n[Bureau] Initialized on port {BUREAU_PORT}")
            print(f"[Bureau] Endpoint: http://0.0.0.0:{BUREAU_PORT}/submit")
            print("=" * 80 + "\n")
        except Exception as e:
            print(f"[Bureau] Failed to initialize: {e}")
            raise
        
    async def start_bureau_async(self):
        """Start Bureau asynchronously in the FastAPI event loop"""
        if self.is_running:
            print("[Bureau] Already running")
            return
        
        print(f"[Bureau] Initialized on port {BUREAU_PORT}")
        self.is_running = True
        
        # NOTE: Bureau.run() creates its own event loop and blocks
        # For production, run Bureau in a separate process
        print("[Bureau] ⚠️  Note: Bureau requires separate process for full operation")
        print("[Bureau] Agents initialized but message processing is disabled")
    
    def trigger_procurement(self, file_id: str, intent: FootprintIntent) -> str:
        """Trigger procurement for a given footprint intent"""
        # Generate request ID
        request_id = f"proc_{file_id}_{int(time_module.time() * 1000)}"
        
        # Store request
        self.procurement_requests[request_id] = {
            "file_id": file_id,
            "intent": intent.__dict__,
            "status": "submitted",
            "timestamp": time_module.time(),
            "results": None,
            "stages": {
                "submission": "completed",
                "provider_query": "pending",
                "payment": "pending",
                "retirement": "pending",
                "proof_generation": "pending"
            }
        }
        
        print(f"\n{'=' * 80}")
        print(f"[Procurement] Request {request_id}")
        print(f"[Procurement] Amount: {intent.tco2e} tCO2e")
        print(f"[Procurement] Period: {intent.period}")
        print(f"[Procurement] Country: {intent.facility_country}")
        print(f"[Procurement] Policy: {intent.policy_name}")
        print(f"{'=' * 80}\n")
        
        # Send to Bureau if running
        bureau_success = self._send_to_bureau(request_id, intent)
        if bureau_success:
            self.procurement_requests[request_id]["stages"]["provider_query"] = "in_progress"
        
        return request_id
    

    def _send_to_bureau(self, request_id: str, intent: FootprintIntent) -> bool:
        """Send procurement request to Bureau agent network"""
        try:
            bureau_url = f"http://localhost:{BUREAU_PORT}/submit"

            # Curator must be running inside this Bureau
            if not self.agents or "curator" not in self.agents:
                print("[Procurement] ⚠️  Bureau not initialized, cannot send request")
                return False

            curator_addr = self.agents["curator"].address

            # Use a REAL local agent for both sender AND signing (payment is fine)
            sender_agent = self.agents.get("payment", self.agents["curator"])
            sender_addr  = sender_agent.address

            # Resolve FootprintIntent schema digest from the protocol registry
            schema_digest = next(
                (d for d, model in INTENT_PROTOCOL.models.items() if model is FootprintIntent),
                None
            )
            if not schema_digest:
                print("[Procurement] ⚠️  FootprintIntent schema digest not found")
                return False

            # Payload delivered to the curator handler (you can include request_id for tracing)
            payload_obj = {"request_id": request_id, **intent.__dict__}

            # Build the Exchange-Protocol envelope (note: no 'source' here)
            env = MsgEnvelope(
                version=1,
                sender=sender_addr,
                target=curator_addr,
                session=str(uuid.uuid4()),
                schema_digest=schema_digest,
                protocol_digest=FOOTPRINT_PROTOCOL_DIGEST,  # e.g. "proto:<64hex>"
            )
            # Base64 payload encoding handled internally
            env.encode_payload(json.dumps(payload_obj))

            # Sign with the SAME agent as 'sender'
            env.sign(sender_agent)

            print(f"[Procurement] → Sending request to Bureau at {bureau_url}")
            resp = requests.post(bureau_url, json=env.model_dump(mode="json"), timeout=5.0)

            if resp.status_code == 200:
                print("[Procurement] ✓ Request sent successfully")
                return True

            print(f"[Procurement] ⚠️  Bureau returned {resp.status_code}: {resp.text}")
            return False

        except requests.exceptions.ConnectionError:
            print(f"[Procurement] ⚠️  Cannot connect to Bureau on port {BUREAU_PORT}")
            return False
        except requests.exceptions.Timeout:
            print("[Procurement] ⚠️  Bureau connection timeout")
            return False
        except Exception as e:
            print(f"[Procurement] ⚠️  Failed to send to Bureau: {e}")
            return False


    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a procurement request"""
        return self.procurement_requests.get(request_id)
    
    def list_requests(self, file_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """List procurement requests, optionally filtered by file_id"""
        requests = []
        for req_id, req_data in self.procurement_requests.items():
            if file_id is None or req_data["file_id"] == file_id:
                requests.append({
                    "request_id": req_id,
                    **req_data
                })
        # Sort by timestamp descending and limit
        return sorted(requests, key=lambda x: x["timestamp"], reverse=True)[:limit]

# Global procurement orchestrator
procurement = ProcurementOrchestrator()

# ============================================================================
# Lifespan Context Manager
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    print("\n🚀 Starting GreenChain Comprehensive Platform...")
    print(f"FastAPI will run on port {FASTAPI_PORT}")
    print(f"Bureau will run on port {BUREAU_PORT}")
    
    # Initialize and start procurement bureau
    try:
        procurement.initialize_bureau()
        await procurement.start_bureau_async()
        print("✅ Procurement orchestrator initialized\n")
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize procurement: {e}")
        print("Continuing without procurement support\n")
    
    yield
    
    # Shutdown
    print("\n🛑 Shutting down GreenChain Platform...")

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="GreenChain Comprehensive Platform",
    version="2.0.0",
    description="Complete ingestion, CO2 calculation, and procurement orchestration",
    lifespan=lifespan
)

# ============================================================================
# Ingestion Endpoints
# ============================================================================

@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Ingest electricity bill from PDF.
    
    Uses ASI (LLM) to extract billing information and interval readings.
    Returns canonical envelope JSON with file_id for downstream processing.
    
    **Next Steps:**
    - Calculate CO2: POST /co2/calculate?file_id={file_id}
    - Full Pipeline: POST /pipeline/full?file_id={file_id}
    """
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
    
    src = SourceInfo(
        source_type="pdf",
        filename=file.filename,
        sha256=sha256_bytes(content),
        extra={}
    )
    
    env = Envelope(source=src)
    if "billing_statements" in extracted:
        env.billing_statements = [BillingStatement(**bs) for bs in extracted["billing_statements"]]
    if "interval_readings" in extracted:
        env.interval_readings = [IntervalReading(**ir) for ir in extracted["interval_readings"]]
    env.close()
    
    # Save to temp directory
    envelope_data = env.model_dump(mode="json")
    file_id = env.source.sha256[:16]
    output_path = TEMP_DATA_DIR / f"envelope_{file_id}.json"
    with open(output_path, "w") as f:
        json.dump(envelope_data, f, indent=2)
    
    return {
        **envelope_data,
        "file_id": file_id,
        "next_steps": {
            "calculate_co2": f"/co2/calculate?file_id={file_id}",
            "calculate_and_procure": f"/pipeline/full?file_id={file_id}"
        }
    }

@app.post("/ingest/excel")
async def ingest_excel(file: UploadFile = File(...)):
    """
    Ingest meter readings from Excel or CSV file.
    
    Uses ASI (LLM) to parse spreadsheet data and extract interval readings.
    Returns canonical envelope JSON with file_id for downstream processing.
    
    **Supported Formats:**
    - CSV files
    - Excel files (.xlsx, .xls)
    - Multiple sheets supported for Excel
    
    **Next Steps:**
    - Calculate CO2: POST /co2/calculate?file_id={file_id}
    - Full Pipeline: POST /pipeline/full?file_id={file_id}
    """
    content = await file.read()
    body_text = ""
    name = file.filename.lower()
    
    if name.endswith(".csv"):
        body_text = content.decode("utf-8", errors="ignore")
    else:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
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
    
    src = SourceInfo(
        source_type="excel",
        filename=file.filename,
        sha256=sha256_bytes(content),
        extra={}
    )
    
    env = Envelope(source=src)
    if "interval_readings" in extracted:
        env.interval_readings = [IntervalReading(**ir) for ir in extracted["interval_readings"]]
    if "billing_statements" in extracted:
        env.billing_statements = [BillingStatement(**bs) for bs in extracted["billing_statements"]]
    env.close()
    
    envelope_data = env.model_dump(mode="json")
    file_id = env.source.sha256[:16]
    output_path = TEMP_DATA_DIR / f"envelope_{file_id}.json"
    with open(output_path, "w") as f:
        json.dump(envelope_data, f, indent=2)
    
    return {
        **envelope_data,
        "file_id": file_id,
        "next_steps": {
            "calculate_co2": f"/co2/calculate?file_id={file_id}",
            "calculate_and_procure": f"/pipeline/full?file_id={file_id}"
        }
    }

# ============================================================================
# CO2 Calculation Endpoints
# ============================================================================

@app.post("/co2/calculate")
async def calculate_co2(
    envelope: Optional[Dict[str, Any]] = Body(None),
    file_id: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
    region: Optional[str] = Query(None),
    year: int = Query(2025),
    method: str = Query("location"),
    include_tandd: bool = Query(True),
    verbose: bool = Query(False)
):
    """
    Calculate CO2 emissions from electricity consumption data.
    
    **Input Options:**
    1. Pass envelope JSON directly in request body
    2. Reference a file_id from previous ingestion
    
    **Calculation Methods:**
    - `location`: Location-based (Scope 2) using grid average
    - `market`: Market-based (Scope 2) using supplier-specific factors
    
    **Returns:**
    - Comprehensive CO2 footprint with Scope 2 and optional Scope 3 (T&D losses)
    - Emission factors used with sources
    - file_id for procurement trigger
    
    **Example Envelope:**
    ```json
    {
        "interval_readings": [
            {"kwh": 100, "start_time": "2024-01-01T00:00:00Z"},
            {"kwh": 150, "start_time": "2024-01-01T01:00:00Z"}
        ],
        "billing_statement": {
            "billing_period_start": "2024-01-01",
            "billing_period_end": "2024-01-31"
        }
    }
    ```
    
    **Next Steps:**
    - Trigger procurement: POST /procurement/trigger?file_id={file_id}
    - Full pipeline: POST /pipeline/full
    """
    # Load input data - either from body or file_id
    if envelope:
        input_json = envelope
        # Generate a file_id for this request
        file_id = hashlib.sha256(json.dumps(envelope).encode()).hexdigest()[:16]
    elif file_id:
        input_path = TEMP_DATA_DIR / f"envelope_{file_id}.json"
        if not input_path.exists():
            # Try with co2_ prefix (if already calculated)
            input_path = TEMP_DATA_DIR / f"co2_{file_id}.json"
            if not input_path.exists():
                return JSONResponse(
                    {"error": f"File ID {file_id} not found. Ingest data first or provide envelope in body."},
                    status_code=404
                )
        with open(input_path) as f:
            input_json = json.load(f)
    else:
        return JSONResponse(
            {"error": "Either envelope JSON body or file_id query parameter is required"},
            status_code=400
        )
    
    # Calculate total kWh from interval_readings
    intervals = input_json.get("interval_readings") or []
    kwh_total = sum(float(r.get("kwh") or 0.0) for r in intervals)
    
    if kwh_total == 0:
        return JSONResponse(
            {"error": "No kWh consumption found in data"},
            status_code=400
        )
    
    # Infer country/region via LLM
    country_llm, region_llm, used_llm, llm_note = infer_country_state_via_asi(
        input_json,
        verbose=verbose
    )
    
    # Override chain: explicit params > LLM inference > heuristic fallback
    final_country = country or country_llm
    final_region = region or region_llm
    
    # Heuristic fallback
    if not final_country:
        final_country, _ = infer_country_state_heuristic(input_json)
    
    # Build reporting config
    cfg = ReportingConfig(
        year=year,
        country=final_country,
        state_or_region=final_region,
        method=method,
        standard="GHG Protocol",
        include_t_and_d_scope3=include_tandd,
    )
    
    # Resolve emission factor
    factor = resolve_factor_via_asi(cfg, verbose=verbose)
    
    # Compute emissions
    result = compute_scope2_and_tandd(kwh_total, cfg, factor)
    
    # Build response
    co2_footprint = {
        "reporting": {
            "standard": cfg.standard,
            "year": cfg.year,
            "country": cfg.country,
            "state_or_region": cfg.state_or_region,
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
        }
    }
    
    # Save updated envelope with CO2 footprint
    input_json["co2_footprint"] = co2_footprint
    output_path = TEMP_DATA_DIR / f"co2_{file_id}.json"
    with open(output_path, "w") as f:
        json.dump(input_json, f, indent=2)
    
    return {
        **co2_footprint,
        "file_id": file_id,
        "saved_to": str(output_path),
        "next_steps": {
            "trigger_procurement": f"/procurement/trigger?file_id={file_id}",
            "full_pipeline": "/pipeline/full",
            "download": f"/co2/download/{file_id}"
        }
    }

@app.get("/co2/download/{file_id}")
async def download_co2(file_id: str):
    """
    Download CO2 calculation results as JSON file.
    
    Returns the complete envelope with CO2 footprint calculation.
    """
    output_path = TEMP_DATA_DIR / f"co2_{file_id}.json"
    
    if not output_path.exists():
        return JSONResponse(
            {"error": f"CO2 calculation for file_id {file_id} not found"},
            status_code=404
        )
    
    return FileResponse(
        path=output_path,
        media_type="application/json",
        filename=f"co2_footprint_{file_id}.json"
    )

# ============================================================================
# Procurement Endpoints
# ============================================================================

@app.post("/procurement/trigger")
async def trigger_procurement(
    file_id: str = Query(..., description="File ID from CO2 calculation"),
    policy_name: str = Query("Balanced quality v1", description="Procurement policy"),
    currency_token: str = Query("PYUSD", description="Payment currency token"),
    auto_execute: bool = Query(True, description="Auto-execute procurement")
):
    """
    Trigger procurement for calculated CO2 emissions.
    
    Loads the CO2 calculation from file_id and creates a procurement request
    to the Bureau agent network.
    
    **Prerequisites:**
    - CO2 must be calculated first (POST /co2/calculate)
    
    **Procurement Flow:**
    1. Load CO2 footprint from file_id
    2. Create FootprintIntent with emission data
    3. Submit to Bureau agent network
    4. Track request with unique request_id
    
    **Returns:**
    - request_id for status tracking
    - Procurement intent details
    - Bureau endpoint information
    
    **Next Steps:**
    - Check status: GET /procurement/status/{request_id}
    - List all requests: GET /procurement/requests
    """
    # Load CO2 calculation
    input_path = TEMP_DATA_DIR / f"co2_{file_id}.json"
    if not input_path.exists():
        return JSONResponse(
            {"error": f"File {file_id} not found. Calculate CO2 first."},
            status_code=404
        )
    
    with open(input_path, "r") as f:
        data = json.load(f)
    
    # Extract CO2 footprint
    co2_footprint = data.get("co2_footprint")
    if not co2_footprint:
        return JSONResponse(
            {"error": "No CO2 footprint found in file. Recalculate CO2."},
            status_code=400
        )
    
    # Extract emission data
    outputs = co2_footprint.get("outputs", {})
    reporting = co2_footprint.get("reporting", {})
    
    scope2_tonnes = outputs.get("scope2_location_tonnes", 0.0) if reporting.get("method") == "location" else outputs.get("scope_2_market_tco2e", 0.0)
    scope3_tonnes = outputs.get("scope3_3_tandd_tonnes", 0.0) if reporting.get("include_t_and_d_scope3") else 0.0
    
    # new
    s2 = as_decimal(scope2_tonnes)
    s3 = as_decimal(scope3_tonnes)
    total_tonnes_dec = s2 + s3
    total_tonnes = float(total_tonnes_dec)  # if downstream expects float

    if total_tonnes <= 0:
        # Give a clear client error instead of a 500 if nothing to procure
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail="No emissions found to procure offsets for. scope2_tonnes and scope3_tonnes were missing or zero."
        )

    
    if total_tonnes <= 0:
        return JSONResponse(
            {"error": "Total emissions must be greater than zero"},
            status_code=400
        )
    
    # Get period from billing statement or default
    bs = data.get("billing_statement") or {}
    period = bs.get("billing_period_start", "")[:7] or "2025-01"  # YYYY-MM
    
    # Create FootprintIntent
    intent = FootprintIntent(
        tco2e=round(total_tonnes, 5),
        period=period,
        facility_country=reporting.get("country", "IN"),
        factor_source=outputs.get("factor_used", {}).get("source_name", "Unknown"),
        scope=["scope2_location"] + (["scope3_3"] if scope3_tonnes else []),
        currency_token=currency_token,
        policy_name=policy_name,
        auto_execute=auto_execute,
    )
    
    # Trigger procurement and get request_id
    request_id = procurement.trigger_procurement(file_id, intent)
    
    # Get updated status (may show Bureau connection status)
    status = procurement.get_request_status(request_id)
    
    return {
        "status": "procurement_triggered",
        "request_id": request_id,
        "intent": intent.__dict__,
        "file_id": file_id,
        "bureau_endpoint": f"http://0.0.0.0:{BUREAU_PORT}/submit",
        "bureau_status": "sent" if status.get("stages", {}).get("provider_query") == "in_progress" else "bureau_not_running",
        "message": "Procurement request created and sent to Bureau (if running)",
        "next_steps": {
            "check_status": f"/procurement/status/{request_id}",
            "list_all": f"/procurement/requests?file_id={file_id}",
        }
    }

@app.get("/procurement/status/{request_id}")
async def get_procurement_status(request_id: str):
    """
    Get status of a specific procurement request.
    
    Returns detailed status including:
    - Overall status
    - Individual stage progress
    - Timestamp
    - Intent details
    - Results (if completed)
    """
    status = procurement.get_request_status(request_id)
    
    if not status:
        return JSONResponse(
            {"error": f"Procurement request {request_id} not found"},
            status_code=404
        )
    
    return {
        "request_id": request_id,
        **status
    }

@app.get("/procurement/requests")
async def list_procurement_requests(
    file_id: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=100)
):
    """
    List procurement requests, optionally filtered by file_id.
    
    **Query Parameters:**
    - file_id: Filter by specific file_id
    - limit: Maximum number of requests to return (1-100)
    
    **Returns:**
    - List of procurement requests sorted by timestamp (newest first)
    """
    requests = procurement.list_requests(file_id=file_id, limit=limit)
    
    return {
        "count": len(requests),
        "file_id_filter": file_id,
        "limit": limit,
        "requests": requests
    }

@app.get("/procurement/status")
async def procurement_system_status():
    """
    Get status of procurement agent network.
    
    Returns:
    - Bureau running status
    - Agent addresses
    - Total requests tracked
    - Configuration details
    """
    return {
        "bureau_running": procurement.is_running,
        "bureau_port": BUREAU_PORT,
        "bureau_endpoint": f"http://0.0.0.0:{BUREAU_PORT}/submit",
        "agents": {
            name: agent.address 
            for name, agent in procurement.agents.items()
        } if procurement.agents else {},
        "total_requests": len(procurement.procurement_requests),
        "note": "Full Bureau operation requires separate process to avoid event loop conflicts"
    }

# ============================================================================
# Full Pipeline Endpoint
# ============================================================================

@app.post("/pipeline/full")
async def full_pipeline(
    envelope: Optional[Dict[str, Any]] = Body(None),
    file_id: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
    region: Optional[str] = Query(None),
    year: int = Query(2025),
    policy_name: str = Query("Balanced quality v1"),
    currency_token: str = Query("PYUSD"),
    auto_execute: bool = Query(True)
):
    """
    Execute complete pipeline: Calculate CO2 + Trigger Procurement.
    
    This convenience endpoint combines both steps in a single request.
    
    **Input Options:**
    1. Pass envelope JSON directly in request body
    2. Reference a file_id from previous ingestion
    
    **Pipeline Flow:**
    1. Calculate CO2 emissions
    2. Create procurement request
    3. Submit to Bureau agent network
    
    **Example Request:**
    ```bash
    curl -X POST "http://localhost:8000/pipeline/full?year=2024" \\
         -H "Content-Type: application/json" \\
         -d '{
           "interval_readings": [
             {"kwh": 100, "start_time": "2024-01-01T00:00:00Z"}
           ]
         }'
    ```
    
    **Returns:**
    - Complete CO2 calculation results
    - Procurement request details
    - Tracking IDs for both operations
    """
    # Step 1: Calculate CO2
    co2_result = await calculate_co2(
        envelope=envelope,
        file_id=file_id,
        country=country,
        region=region,
        year=year
    )
    
    # Extract file_id from CO2 result if we used envelope body
    if envelope and not file_id:
        file_id = co2_result.get("file_id")
    
    if not file_id:
        return JSONResponse(
            {"error": "CO2 calculation failed to generate file_id"},
            status_code=500
        )
    
    # Step 2: Trigger procurement
    procurement_result = await trigger_procurement(
        file_id=file_id,
        policy_name=policy_name,
        currency_token=currency_token,
        auto_execute=auto_execute
    )
    
    return {
        "status": "pipeline_complete",
        "co2_calculation": co2_result,
        "procurement": procurement_result,
        "file_id": file_id,
        "message": "CO2 calculated and procurement triggered successfully"
    }

# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/")
async def root():
    """
    API information and available endpoints.
    
    Returns comprehensive API documentation with:
    - Service description
    - Available endpoint categories
    - Port configuration
    - Bureau status
    """
    return {
        "service": "GreenChain Comprehensive Platform",
        "version": "2.0.0",
        "description": "Complete platform for electricity bill ingestion, CO2 calculation, and carbon credit procurement",
        "components": {
            "ingestion": "FastAPI endpoints for PDF/Excel/GCP",
            "co2_engine": "Scope 2/3 emission calculations with GHG Protocol compliance",
            "procurement": "uAgents Bureau for carbon credit procurement with multi-provider support"
        },
        "endpoints": {
            "ingestion": {
                "pdf": "POST /ingest/pdf - Upload PDF electricity bill",
                "excel": "POST /ingest/excel - Upload Excel/CSV meter data",
            },
            "co2_calculation": {
                "calculate": "POST /co2/calculate - Calculate emissions",
                "download": "GET /co2/download/{file_id} - Download results"
            },
            "procurement": {
                "trigger": "POST /procurement/trigger - Trigger procurement",
                "status_system": "GET /procurement/status - Bureau status",
                "status_request": "GET /procurement/status/{request_id} - Request status",
                "list_requests": "GET /procurement/requests - List all requests"
            },
            "pipeline": {
                "full": "POST /pipeline/full - Complete flow: ingest→calculate→procure"
            }
        },
        "ports": {
            "fastapi": FASTAPI_PORT,
            "bureau": BUREAU_PORT
        },
        "bureau_status": {
            "initialized": procurement.is_running,
            "agents": len(procurement.agents),
            "agent_addresses": {
                name: agent.address 
                for name, agent in procurement.agents.items()
            } if procurement.agents else {}
        },
        "documentation": {
            "swagger": f"http://0.0.0.0:{FASTAPI_PORT}/docs",
            "redoc": f"http://0.0.0.0:{FASTAPI_PORT}/redoc"
        }
    }

@app.get("/health")
async def health():
    """
    Health check endpoint.
    
    Returns system health status including:
    - FastAPI server status
    - Bureau initialization status
    - Agent count
    - Request count
    """
    return {
        "status": "healthy",
        "fastapi": "running",
        "bureau": "initialized" if procurement.is_running else "not_initialized",
        "agents": len(procurement.agents),
        "requests_tracked": len(procurement.procurement_requests),
        "timestamp": time_module.time()
    }

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  GreenChain Comprehensive Platform v2.0.0")
    print("=" * 80)
    print(f"\n  FastAPI Server: http://0.0.0.0:{FASTAPI_PORT}")
    print(f"  Bureau Server: http://0.0.0.0:{BUREAU_PORT}")
    print(f"  API Documentation: http://0.0.0.0:{FASTAPI_PORT}/docs")
    print("\n" + "=" * 80)
    print("\n  Components:")
    print("  ✓ PDF/Excel/GCP Ingestion")
    print("  ✓ CO2 Calculation Engine (Scope 2 & 3)")
    print("  ✓ Procurement Agent Network")
    print("  ✓ Full Pipeline Orchestration")
    print("\n" + "=" * 80)
    print("\n  Note: Bureau initialized but requires separate process for full operation")
    print("  See documentation for production deployment with multiprocessing")
    print("\n" + "=" * 80 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=FASTAPI_PORT,
    )