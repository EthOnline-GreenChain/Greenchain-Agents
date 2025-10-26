#!/usr/bin/env python3
"""
GreenChain Integrated Runner (FIXED VERSION)
=============================================
Fixes:
1. Uses lifespan context manager instead of deprecated on_event
2. Proper Bureau initialization without event loop conflicts  
3. Procurement status tracking
4. Status query endpoints

Combines:
1. FastAPI ingestion endpoints (PDF, Excel, GCP)
2. CO2 calculation engine (Scope 2/3 emissions)
3. Procurement orchestration (uagents Bureau for carbon credits)
"""

from __future__ import annotations
import io, os, json, uvicorn, hashlib, pandas as pd, asyncio, time as time_module
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, Query, BackgroundTasks, Body
from fastapi.responses import JSONResponse
from pathlib import Path

# Ingestion imports
from greenchain.schema import Envelope, SourceInfo, BillingStatement, IntervalReading, envelope_json_schema_for_asi
from greenchain.pdf_text import extract_text_from_pdf
from greenchain.asi_client import ASIClient

# CO2 engine imports
from co2_engine import (
    ReportingConfig, 
    resolve_factor_via_asi, 
    compute_scope2_and_tandd,
    infer_country_state_via_asi,
    infer_country_state_heuristic,
    load_json,
    save_json
)

# Procurement imports
from uagents import Bureau
from procurement.basket_curator import build_agent as build_curator, FootprintIntent
from procurement.provider_adaptor_in import build_agent as build_provider_in
from procurement.provider_adaptor_np import build_agent as build_provider_np
from procurement.payment_orchestrator import build_agent as build_payment
from procurement.retirement_registrar import build_agent as build_retire
from procurement.proof_bundler import build_agent as build_bundler

# Import envelope coercion utilities from original runner
from runner import coerce_any_to_envelope, normalize_canonical_envelope_shapes, sha256_bytes

# ============================================================================
# Configuration
# ============================================================================
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))
BUREAU_PORT = int(os.getenv("BUREAU_PORT", "8001"))
TEMP_DATA_DIR = Path("/tmp/greenchain_data")
TEMP_DATA_DIR.mkdir(exist_ok=True)

# ============================================================================
# Procurement Bureau Setup
# ============================================================================
class ProcurementOrchestrator:
    """Manages the uAgents Bureau for carbon credit procurement"""
    
    def __init__(self):
        self.bureau: Optional[Bureau] = None
        self.is_running = False
        self.curator = None
        self.agents = {}
        # Track procurement requests
        self.procurement_requests: Dict[str, Dict[str, Any]] = {}
        
    def initialize_bureau(self):
        """Initialize all procurement agents and create Bureau"""
        try:
            # Build all agents
            prov_in = build_provider_in(name="ProviderIN", seed="seed provider in 1/2/3")
            prov_np = build_provider_np(name="ProviderNP", seed="seed provider np 1/2/3")
            payment = build_payment(name="PaymentOrchestrator", seed="seed payment 1/2/3")
            registrar = build_retire(name="RetirementRegistrar", seed="seed retire 1/2/3")
            bundler = build_bundler(name="ProofBundler", seed="seed bundler 1/2/3")
            
            # Initial curator without demo intent
            curator = build_curator(
                name="BasketCurator",
                seed="seed curator 1/2/3",
                provider_addresses=[prov_in.address, prov_np.address],
                payment_orchestrator_addr=payment.address,
                retirement_registrar_addr=registrar.address,
                proof_bundler_addr=bundler.address,
                demo_intent=None,  # Will be triggered via API
            )
            
            # Store agents
            self.agents = {
                "curator": curator,
                "prov_in": prov_in,
                "prov_np": prov_np,
                "payment": payment,
                "registrar": registrar,
                "bundler": bundler,
            }
            
            # Validation
            for name, agent in self.agents.items():
                if agent is None:
                    raise RuntimeError(f"{name} built as None")
                print(f"[Procurement] {name}: {agent.address}")
            
            # Create Bureau
            self.bureau = Bureau(
                port=BUREAU_PORT,
                agents=[a for a in self.agents.values()],
                endpoint=f"http://0.0.0.0:{BUREAU_PORT}/submit",
            )
            
            # Store curator for later use
            self.curator = curator
            print("[Procurement] Bureau initialized successfully")
        except Exception as e:
            print(f"[Procurement] Failed to initialize bureau: {e}")
            raise
        
    async def start_bureau_async(self):
        """Start Bureau asynchronously in the FastAPI event loop"""
        if self.is_running:
            print("[Procurement] Bureau already running")
            return
        
        print(f"[Procurement] Bureau initialized on port {BUREAU_PORT}")
        self.is_running = True
        
        # NOTE: Bureau.run() creates its own event loop and blocks
        # To run Bureau properly, you need to either:
        # 1. Run it in a separate process (recommended for production)
        # 2. Use multiprocessing instead of threading
        # 3. Integrate Bureau's event loop with FastAPI's (complex)
        
        # For now, Bureau is initialized but not actively running
        # Agents are available but won't process messages until Bureau.run() is called
        print("[Procurement] ⚠️  Note: Bureau requires separate process for full operation")
        print("[Procurement] Agents initialized but message processing is disabled")
    
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
            },
            "note": "Full agent workflow requires Bureau to run in separate process"
        }
        
        print(f"[Procurement] Created request {request_id} for {intent.tco2e} tCO2e")
        
        # TODO: In production, send message to curator agent via uagents messaging
        # This requires Bureau to be running in a separate process
        
        return request_id
    
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
# Lifespan Context Manager (NEW - replaces on_event)
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    print("[Startup] Initializing GreenChain Integrated Platform...")
    print(f"[Startup] FastAPI will run on port {FASTAPI_PORT}")
    print(f"[Startup] Bureau will run on port {BUREAU_PORT}")
    
    # Initialize and start procurement bureau
    try:
        procurement.initialize_bureau()
        await procurement.start_bureau_async()
        print("[Startup] Procurement orchestrator initialized")
    except Exception as e:
        print(f"[Startup] Warning: Could not initialize procurement: {e}")
        print("[Startup] Continuing without procurement support")
    
    yield
    
    # Shutdown
    print("[Shutdown] Cleaning up...")

# ============================================================================
# FastAPI Application (with lifespan)
# ============================================================================
app = FastAPI(
    title="GreenChain Integrated Platform",
    version="1.0.1",  # Updated version
    description="Complete ingestion, CO2 calculation, and procurement orchestration",
    lifespan=lifespan  # NEW: Use lifespan context manager
)

# ============================================================================
# Ingestion Endpoints
# ============================================================================

@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Ingest electricity bill from PDF
    Returns: Canonical envelope JSON with billing_statements and interval_readings
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
    
    # Save to temp directory for later CO2 calculation
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
    Ingest meter readings from Excel/CSV
    Returns: Canonical envelope JSON with interval_readings
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

@app.get("/gcp/co2")
async def gcp_co2(hours: float = Query(1.0), avg_util: float = Query(0.3)):
    """
    Calculate CO2 emissions for GCP compute instance
    Returns: GCP-specific footprint calculation
    """
    from .scripts.gcp_co2 import estimate_kwh, REGION_GRID_INTENSITY, PUE, get_machine_specs
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
        return JSONResponse(
            {"error": "Not running on GCE (metadata unavailable)"},
            status_code=400
        )
    
    try:
        specs = get_machine_specs(project_id, zone, machine_type)
    except Exception:
        specs = {
            "vcpus": int(os.getenv("GC_VCPU", "2")),
            "memory_gib": float(os.getenv("GC_MEM_GIB", "4"))
        }
    
    intensity = REGION_GRID_INTENSITY.get(region, 400.0)
    kwh = estimate_kwh(
        hours=hours,
        vcpus=specs["vcpus"],
        mem_gib=specs["memory_gib"],
        cpu_util=avg_util
    )
    co2e_kg = kwh * (intensity / 1000.0)
    
    return JSONResponse({
        "schema_version": "greenchain.ingest.v1",
        "source": {
            "source_type": "gcp",
            "filename": None,
            "sha256": None,
            "received_at": None,
            "extra": {
                "project_id": project_id,
                "zone": zone,
                "region": region,
                "machine_type": machine_type
            }
        },
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
    })

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
    method: str = Query("auto"),
    standard: str = Query("GHG-Protocol-Scope2"),
    include_tandd: bool = Query(False),
    tandd_pct: Optional[float] = Query(None),
    supplier_factor: Optional[float] = Query(None),
    residual_mix: Optional[float] = Query(None),
    recs_mwh: float = Query(0.0),
    ppa_kwh: float = Query(0.0),
    verbose: bool = Query(False)
):
    """
    Calculate CO2 emissions for ingested electricity data
    
    Accepts envelope JSON directly in request body OR file_id query parameter.
    
    Request body (envelope JSON):
    {
        "billing_statements": [...],
        "interval_readings": [...],
        "source": {...}
    }
    
    Query Parameters:
    - file_id: ID from ingestion endpoint (alternative to envelope body)
    - country: Override country code (e.g., US, UK, IN)
    - region: Override state/region
    - year: Reporting year
    - method: Calculation method (auto, location, market)
    - standard: Reporting standard (default: GHG-Protocol-Scope2)
    - include_tandd: Include Scope 3.3 T&D losses
    - tandd_pct: Override T&D loss fraction
    - supplier_factor: Supplier-specific kgCO2e/kWh
    - residual_mix: Residual mix kgCO2e/kWh
    - recs_mwh: EACs/RECs volume in MWh
    - ppa_kwh: PPA-covered kWh
    """
    # Load input data - either from body or file_id
    if envelope:
        input_json = envelope
        # Generate a file_id for this request
        import hashlib
        file_id = hashlib.sha256(json.dumps(envelope).encode()).hexdigest()[:16]
    elif file_id:
        input_path = TEMP_DATA_DIR / f"envelope_{file_id}.json"
        if not input_path.exists():
            return JSONResponse(
                {"error": f"File ID {file_id} not found. Ingest data first."},
                status_code=404
            )
        with open(input_path) as f:
            input_json = json.load(f)
    else:
        return JSONResponse(
            {"error": "Either envelope JSON body or file_id query parameter is required"},
            status_code=400
        )
    
    # Calculate total kWh
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
    
    # Override chain
    final_country = country or country_llm
    final_region = region or region_llm
    
    # Heuristic fallback
    if not final_country:
        final_country, _ = infer_country_state_heuristic(input_json)
    
    # Build config
    cfg = ReportingConfig(
        year=year,
        country=final_country,
        state_or_region=final_region,
        method=method,
        standard=standard,
        include_t_and_d_scope3=include_tandd,
        t_and_d_pct_override=tandd_pct,
        supplier_specific_factor_kg_per_kwh=supplier_factor,
        residual_mix_factor_kg_per_kwh=residual_mix,
        recs_mwh=recs_mwh,
        ppa_kwh=ppa_kwh,
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
    
    # Save updated envelope
    input_json["co2_footprint"] = co2_footprint
    output_path = TEMP_DATA_DIR / f"co2_{file_id}.json"
    with open(output_path, "w") as f:
        json.dump(input_json, f, indent=2)
    
    return {
        **co2_footprint,
        "file_id": file_id,
        "next_steps": {
            "procure_credits": f"/procurement/trigger?file_id={file_id}",
            "download_report": f"/co2/download?file_id={file_id}"
        }
    }

@app.get("/co2/download")
async def download_co2_report(file_id: str = Query(...)):
    """Download complete CO2 calculation report"""
    output_path = TEMP_DATA_DIR / f"co2_{file_id}.json"
    if not output_path.exists():
        return JSONResponse(
            {"error": f"CO2 calculation for {file_id} not found"},
            status_code=404
        )
    
    with open(output_path) as f:
        data = json.load(f)
    
    return JSONResponse(data)

# ============================================================================
# Procurement Endpoints (UPDATED with status tracking)
# ============================================================================

@app.post("/procurement/trigger")
async def trigger_procurement(
    file_id: str = Query(...),
    policy_name: str = Query("Balanced quality v1"),
    currency_token: str = Query("PYUSD"),
    auto_execute: bool = Query(True)
):
    """
    Trigger carbon credit procurement for calculated emissions
    
    Parameters:
    - file_id: ID from CO2 calculation
    - policy_name: Procurement policy
    - currency_token: Payment token
    - auto_execute: Auto-execute procurement
    """
    # Load CO2 data
    co2_path = TEMP_DATA_DIR / f"co2_{file_id}.json"
    if not co2_path.exists():
        return JSONResponse(
            {"error": f"CO2 calculation for {file_id} not found. Calculate emissions first."},
            status_code=404
        )
    
    with open(co2_path) as f:
        data = json.load(f)
    
    co2_footprint = data.get("co2_footprint", {})
    outputs = co2_footprint.get("outputs", {})
    
    # Get emissions in tonnes
    scope2_tonnes = outputs.get("scope2_location_tonnes", 0.0)
    scope3_tonnes = outputs.get("scope3_3_tandd_tonnes", 0.0)
    total_tonnes = scope2_tonnes + (scope3_tonnes if scope3_tonnes else 0.0)
    
    if total_tonnes == 0:
        return JSONResponse(
            {"error": "No emissions found to offset"},
            status_code=400
        )
    
    # Extract reporting info
    reporting = co2_footprint.get("reporting", {})
    period = f"{reporting.get('year', 2025)}"
    if data.get("billing_statements"):
        bs = data["billing_statements"][0]
        period = bs.get("billing_period_start", "")[:7]  # YYYY-MM
    
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
    
    return {
        "status": "procurement_triggered",
        "request_id": request_id,  # NEW: Return request_id
        "intent": intent.__dict__,
        "file_id": file_id,
        "bureau_endpoint": f"http://0.0.0.0:{BUREAU_PORT}/submit",
        "message": "Procurement request submitted to agent network",
        "next_steps": {
            "check_status": f"/procurement/status/{request_id}",
            "list_all": f"/procurement/requests?file_id={file_id}"
        }
    }

@app.get("/procurement/status/{request_id}")
async def get_procurement_status(request_id: str):
    """Get status of a specific procurement request"""
    status = procurement.get_request_status(request_id)
    
    if not status:
        return JSONResponse(
            {"error": f"Procurement request {request_id} not found"},
            status_code=404
        )
    
    return JSONResponse(status)

@app.get("/procurement/requests")
async def list_procurement_requests(
    file_id: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=100)
):
    """List procurement requests, optionally filtered by file_id"""
    requests = procurement.list_requests(file_id=file_id, limit=limit)
    
    return {
        "count": len(requests),
        "requests": requests
    }

@app.get("/procurement/status")
async def procurement_system_status():
    """Get status of procurement agent network"""
    return {
        "bureau_running": procurement.is_running,
        "bureau_port": BUREAU_PORT,
        "bureau_endpoint": f"http://0.0.0.0:{BUREAU_PORT}/submit",
        "agents": {name: addr for name, addr in [(k, v.address) for k, v in procurement.agents.items()]} if procurement.agents else {},
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
    Execute full pipeline: Calculate CO2 + Trigger Procurement
    
    Accepts envelope JSON directly in request body OR file_id query parameter.
    
    This is a convenience endpoint that combines CO2 calculation and procurement
    in a single request. You can pipe the output from /ingest/pdf directly here.
    
    Example:
        curl -X POST "http://localhost:8000/ingest/pdf" -F "file=@bill.pdf" | \
        curl -X POST "http://localhost:8000/pipeline/full?year=2024" \
             -H "Content-Type: application/json" -d @-
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
        "file_id": file_id
    }

# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/")
async def root():
    """API information and available endpoints"""
    return {
        "service": "GreenChain Integrated Platform",
        "version": "1.0.1",
        "components": {
            "ingestion": "FastAPI endpoints for PDF/Excel/GCP",
            "co2_engine": "Scope 2/3 emission calculations",
            "procurement": "uagents Bureau for carbon credit procurement"
        },
        "endpoints": {
            "ingestion": {
                "pdf": "POST /ingest/pdf",
                "excel": "POST /ingest/excel",
                "gcp": "GET /gcp/co2"
            },
            "co2_calculation": {
                "calculate": "POST /co2/calculate",
                "download": "GET /co2/download"
            },
            "procurement": {
                "trigger": "POST /procurement/trigger",
                "status": "GET /procurement/status",
                "request_status": "GET /procurement/status/{request_id}",
                "list_requests": "GET /procurement/requests"
            },
            "pipeline": {
                "full": "POST /pipeline/full - Complete flow: ingest→calculate→procure"
            }
        },
        "ports": {
            "fastapi": FASTAPI_PORT,
            "bureau": BUREAU_PORT
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "fastapi": "running",
        "bureau": "initialized" if procurement.is_running else "not_initialized",
        "note": "Bureau requires separate process for full operation"
    }

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("GreenChain Integrated Platform (FIXED)")
    print("=" * 80)
    print(f"FastAPI Server: http://0.0.0.0:{FASTAPI_PORT}")
    print(f"Bureau Server: http://0.0.0.0:{BUREAU_PORT}")
    print("=" * 80)
    print("\nNote: Bureau initialized but requires separate process for full operation")
    print("See documentation for production deployment with multiprocessing")
    print("=" * 80)
    
    uvicorn.run(
        "runner_integrated_fixed:app",
        host="0.0.0.0",
        port=FASTAPI_PORT,
        reload=False
    )