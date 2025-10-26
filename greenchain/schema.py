# greenchain/schema.py
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator
from datetime import datetime, date
from uuid import uuid4

SCHEMA_VERSION = "greenchain.ingest.v1"

class SourceInfo(BaseModel):
    source_type: Literal["pdf", "excel", "gmail", "gcp"]
    filename: Optional[str] = None
    sha256: Optional[str] = None
    received_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    extra: Dict[str, Any] = Field(default_factory=dict)

class BillingLine(BaseModel):
    component_type: Literal["energy", "demand", "fixed", "credit", "tax", "fee", "other"]
    description: Optional[str] = None
    quantity: Optional[float] = None
    uom: Optional[str] = None
    rate: Optional[float] = None
    amount: float

class BillingStatement(BaseModel):
    statement_id: str = Field(default_factory=lambda: str(uuid4()))
    account_id: Optional[str] = None
    service_point_id: Optional[str] = None
    billing_period_start: date
    billing_period_end: date
    statement_date: Optional[date] = None
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: float
    currency: str = "USD"
    rate_plan: Optional[str] = None
    lines: List[BillingLine] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_range(self):
        if self.billing_period_end < self.billing_period_start:
            raise ValueError("billing_period_end must be >= billing_period_start")
        return self

class IntervalReading(BaseModel):
    reading_id: str = Field(default_factory=lambda: str(uuid4()))
    meter_id: Optional[str] = None
    ts_start_utc: datetime
    ts_end_utc: datetime
    kwh: float
    kw_demand: Optional[float] = None
    quality: Optional[Literal["A","E","U"]] = "U"
    source: Literal["api","pdf","csv","gmail","gcp","excel"] = "pdf"
    ingestion_run_id: Optional[str] = None

    @model_validator(mode="after")
    def _check_time(self):
        if self.ts_end_utc <= self.ts_start_utc:
            raise ValueError("ts_end_utc must be > ts_start_utc")
        return self

class GCPFootprint(BaseModel):
    region: str
    hours: float
    vcpu_count: int
    memory_gib: float
    cpu_utilization: float = Field(ge=0.0, le=1.0)
    pue: float = 1.1
    grid_intensity_gco2_per_kwh: float
    estimated_kwh: float
    estimated_co2e_kg: float
    method: str = "CCF-inspired compute + memory energy × PUE × grid factor"
    references: List[str] = Field(default_factory=list)

class Envelope(BaseModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    started_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    finished_at: Optional[datetime] = None

    source: SourceInfo
    billing_statements: List[BillingStatement] = Field(default_factory=list)
    interval_readings: List[IntervalReading] = Field(default_factory=list)
    gcp_footprint: Optional[GCPFootprint] = None

    def close(self):
        self.finished_at = datetime.utcnow()

def envelope_json_schema_for_asi() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "billing_statements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "account_id": {"type":"string"},
                        "service_point_id": {"type":"string"},
                        "billing_period_start": {"type":"string", "format":"date"},
                        "billing_period_end": {"type":"string", "format":"date"},
                        "statement_date": {"type":"string", "format":"date"},
                        "subtotal": {"type":"number"},
                        "tax": {"type":"number"},
                        "total": {"type":"number"},
                        "currency": {"type":"string"},
                        "rate_plan": {"type":"string"},
                        "lines": {
                            "type":"array",
                            "items": {
                                "type":"object",
                                "additionalProperties": False,
                                "properties": {
                                    "component_type": {"type":"string"},
                                    "description": {"type":"string"},
                                    "quantity": {"type":"number"},
                                    "uom": {"type":"string"},
                                    "rate": {"type":"number"},
                                    "amount": {"type":"number"}
                                },
                                "required": ["component_type", "amount"]
                            }
                        }
                    },
                    "required": ["billing_period_start","billing_period_end","total","currency"]
                }
            },
            "interval_readings": {
                "type": "array",
                "items": {
                    "type":"object",
                    "additionalProperties": False,
                    "properties": {
                        "meter_id": {"type":"string"},
                        "ts_start_utc": {"type":"string","format":"date-time"},
                        "ts_end_utc": {"type":"string","format":"date-time"},
                        "kwh": {"type":"number"},
                        "kw_demand": {"type":"number"},
                        "quality": {"type":"string"}
                    },
                    "required": ["ts_start_utc","ts_end_utc","kwh"]
                }
            }
        },
        "required": ["billing_statements"]
    }
