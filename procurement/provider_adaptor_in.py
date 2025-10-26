# provider_adapter_in.py
from __future__ import annotations
import time, uuid
from typing import List, Dict
from uagents import Agent, Context, Model, Protocol, Field

# Re-declare minimal models to keep file standalone
class Lot(Model):
    provider_id: str
    credit_id: str
    registry: str
    project_id: str
    project_class: str
    country: str
    vintage: int
    available_tonnes: float
    unit_price_pyusd: float
    min_lot_tonnes: float
    proof_required: bool = False
    co_benefits_rating: float = 0.0
    risk_delivery: float = 0.1

class QuoteRequest(Model):
    tco2e: float
    policy_name: str
    country_hint: str
    period: str

class QuoteResponse(Model):
    lots: List[Lot]
    ttl_seconds: int

class ReserveLine(Model):
    credit_id: str
    provider_id: str
    tonnes: float

class Hold(Model):
    hold_id: str
    provider_id: str
    credit_id: str
    tonnes: float
    unit_price: float
    expires_at: str

class ReserveRequest(Model):
    lots: List[ReserveLine]
    hold_ttl: int = 120

class ReserveResponse(Model):
    holds: List[Hold]

QUOTE_PROTO = "curation.quote.v1"
RESERVE_PROTO = "curation.reserve.v1"

def build_agent(name: str, seed: str) -> Agent:
    agent = Agent(name=name, seed=seed)

    # -- Mock inventory (India) —
    inventory: Dict[str, Lot] = {
        "IN-NATURE-2020": Lot(
            provider_id=agent.address, credit_id="IN-NATURE-2020", registry="Verra",
            project_id="P-IN-001", project_class="nature_based", country="IN",
            vintage=2020, available_tonnes=10.0, unit_price_pyusd=9.8, min_lot_tonnes=0.08,
            co_benefits_rating=0.75, risk_delivery=0.05
        ),
        "IN-TECH-2021": Lot(
            provider_id=agent.address, credit_id="IN-TECH-2021", registry="Gold Standard",
            project_id="P-IN-002", project_class="tech_based", country="IN",
            vintage=2021, available_tonnes=10.0, unit_price_pyusd=10.2, min_lot_tonnes=0.08,
            co_benefits_rating=0.6, risk_delivery=0.07
        ),
    }

    proto_q = Protocol(name=QUOTE_PROTO)
    proto_r = Protocol(name=RESERVE_PROTO)

    @proto_q.on_message(model=QuoteRequest, replies=QuoteResponse)
    async def handle_quote(ctx: Context, sender: str, msg: QuoteRequest):
        lots = list(inventory.values())
        ctx.logger.info(f"[{name}] Quote for {msg.tco2e} t (country_hint={msg.country_hint}) -> {len(lots)} lots")
        await ctx.send(sender, QuoteResponse(lots=lots, ttl_seconds=120))

    @proto_r.on_message(model=ReserveRequest, replies=ReserveResponse)
    async def handle_reserve(ctx: Context, sender: str, msg: ReserveRequest):
        holds: List[Hold] = []
        now = int(time.time())
        for line in msg.lots:
            lot = inventory.get(line.credit_id)
            if not lot or line.tonnes <= 0 or line.tonnes > lot.available_tonnes:
                ctx.logger.warning(f"[{name}] Reserve rejected for {line.credit_id}")
                continue
            lot.available_tonnes -= line.tonnes
            holds.append(Hold(
                hold_id="HOLD-" + uuid.uuid4().hex[:8],
                provider_id=agent.address,
                credit_id=line.credit_id,
                tonnes=line.tonnes,
                unit_price=lot.unit_price_pyusd,
                expires_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + msg.hold_ttl)),
            ))
        await ctx.send(sender, ReserveResponse(holds=holds))

    agent.include(proto_q)
    agent.include(proto_r)
    return agent
