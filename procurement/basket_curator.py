# basket_curator.py
from __future__ import annotations
import asyncio, json, math, time, uuid, hashlib
from typing import List, Optional, Dict, Tuple
from dataclasses import asdict

from uagents import Agent, Context, Model, Protocol, Field

# ---------- Message & Data Models (kept here so the file is standalone) ----------
class FootprintIntent(Model):
    tco2e: float
    period: str                         # "YYYY-MM"
    facility_country: str               # e.g., "IN"
    factor_source: str                  # e.g., "CEA 2022-23"
    scope: List[str]
    currency_token: str                 # e.g., "PYUSD"
    policy_name: str = "Balanced quality v1"
    auto_execute: bool = True

class Lot(Model):
    provider_id: str
    credit_id: str
    registry: str
    project_id: str
    project_class: str                  # "nature_based" | "tech_based"
    country: str
    vintage: int
    available_tonnes: float
    unit_price_pyusd: float
    min_lot_tonnes: float
    proof_required: bool = False
    co_benefits_rating: float = 0.0      # 0..1
    risk_delivery: float = 0.1           # 0..1 (lower is better)

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

class Fill(Model):
    credit_id: str
    tonnes: float
    unit_price: float
    hold_id: str

class SettleRequest(Model):
    order_id: str
    holds: List[Hold]
    buyer_wallet: str
    spend_limit_pyusd: float

class SettleResponse(Model):
    tx_hashes: List[str]
    fills: List[Fill]

class RetireRequest(Model):
    fills: List[Fill]
    retire_for_wallet: str
    claim_text: str
    proof_bundle_hash: str

class Retirement(Model):
    credit_id: str
    tonnes: float
    tx_hash: str
    proof_hash: str

class RetireResponse(Model):
    retirements: List[Retirement]

class BasketLine(Model):
    provider_id: str
    credit_id: str
    project_id: str
    registry: str
    cls: str
    country: str
    vintage: int
    tonnes: float
    unit_price_pyusd: float
    score: float

class BasketDraft(Model):
    target_tco2e: float
    lines: List[BasketLine]
    expected_total_pyusd: float
    policy_name: str

class ProofBundleRequest(Model):
    order_id: str
    basket_manifest: BasketDraft
    provider_docs: List[Dict] = []
    fills: List[Fill] = []
    retirement_txs: List[str] = []
    attestations: List[Dict] = []
    hash_algo: str = "keccak256"

class ProofBundleResponse(Model):
    bundle_hash: str

class TradeResult(Model):
    order_id: str
    run_id: str
    policy_name: str
    target_tco2e: float
    basket: BasketDraft
    reservation_holds: List[Hold]
    settlement: SettleResponse
    retirements: RetireResponse
    proof_bundle_hash: str

class LastTradeResponse(Model):
    status: str
    trade: Optional[TradeResult] = None


# ---------- Protocol names (for clarity) ----------
QUOTE_PROTO = "curation.quote.v1"
RESERVE_PROTO = "curation.reserve.v1"
SETTLE_PROTO = "curation.settle.v1"
RETIRE_PROTO = "curation.retire.v1"
# Proof bundler is internal call (no strict proto name needed here)

# ---------- Default Balanced Policy (hackathon-ready) ----------
BALANCED_POLICY = {
    "name": "Balanced quality v1",
    "min_vintage": 2016,
    "target_mix": [
        {"class": "nature_based", "share": 0.5, "registry": ["Verra", "Gold Standard"]},
        {"class": "tech_based", "share": 0.5, "registry": ["Gold Standard", "Puro"]},
    ],
    "constraints": {
        "max_share_per_project": 0.4,
        "min_registry_quality": "tier1",
        "country_affinity": {"IN": 0.7, "neighbors": 0.3},
        "price_ceiling_per_tonne": 20.0,
        "allow_fractional_tonnes": True,
    },
    "scoring_weights": {
        "price": 0.35,
        "registry": 0.2,
        "vintage": 0.15,
        "country_match": 0.15,
        "co_benefits": 0.1,
        "delivery_risk": 0.05,
    },
    "auto_execute": True,
    "retire_immediately": True,
}

# ---------- Curator builder ----------
def build_agent(
    name: str,
    seed: str,
    provider_addresses,
    payment_orchestrator_addr: str,
    retirement_registrar_addr: str,
    proof_bundler_addr: str,
    demo_intent,                     # FootprintIntent instance
    policy: dict = None,
) -> Agent:
    agent = Agent(name=name, seed=seed)

    @agent.on_event("startup")
    async def on_start(ctx: Context):
        ctx.storage.set("providers", provider_addresses or [])
        ctx.storage.set("payment_addr", payment_orchestrator_addr)
        ctx.storage.set("retire_addr", retirement_registrar_addr)
        ctx.storage.set("bundler_addr", proof_bundler_addr)
        ctx.storage.set("policy", policy or {})
        ctx.storage.set("last_trade", None)

        buyer_wallet = agent.address               # <— use the agent’s address
        ctx.storage.set("buyer_wallet", buyer_wallet)

        ctx.logger.info(f"[{name}] built ok at {agent.address}")  # <— changed

        try:
            # import here to avoid circulars
            from basket_curator import kickoff_procurement
            await kickoff_procurement(ctx, demo_intent, buyer_wallet=buyer_wallet)  # <— pass it in
        except Exception as e:
            ctx.logger.error(f"[{name}] kickoff failed: {e}")


    @agent.on_rest_get("/last-trade", LastTradeResponse)
    async def get_last_trade(ctx: Context) -> LastTradeResponse:
        data = ctx.storage.get("last_trade")
        return LastTradeResponse(status="ok" if data else "no-trade-yet", trade=data)

    return agent   # <-- DO NOT REMOVE



# ---------- Utility: simple scoring/optimization (greedy for tiny orders) ----------
def registry_quality(registry: str) -> float:
    tier = {
        "Gold Standard": 1.0,
        "Puro": 0.95,
        "Verra": 0.9,
    }
    return tier.get(registry, 0.6)

def country_affinity(lot_country: str, facility_country: str) -> float:
    if lot_country == facility_country:
        return 1.0
    # extremely light "neighbor" notion for demo (adjust as needed)
    neighbors = {"IN": {"NP", "BD", "PK", "LK"}}
    return 0.6 if lot_country in neighbors.get(facility_country, set()) else 0.2

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def round_tonnes(x: float, places: int = 4) -> float:
    q = 10**places
    return math.floor(x * q + 0.5) / q

def score_lot(lot: Lot, facility_country: str, price_ceiling: float, weights: Dict, current_year: int) -> float:
    w = weights
    price_term = (1 - min(lot.unit_price_pyusd / price_ceiling, 1.0))
    reg_term = registry_quality(lot.registry)
    vint_term = clamp((lot.vintage - 2010) / max(1, (current_year - 2010)), 0, 1)
    country_term = country_affinity(lot.country, facility_country)
    co_term = lot.co_benefits_rating
    deliv_term = (1 - lot.risk_delivery)

    return (
        w["price"] * price_term
        + w["registry"] * reg_term
        + w["vintage"] * vint_term
        + w["country_match"] * country_term
        + w["co_benefits"] * co_term
        + w["delivery_risk"] * deliv_term
    )

# ---------- Orchestration ----------
async def kickoff_procurement(ctx: Context, intent: FootprintIntent, buyer_wallet: Optional[str] = None):

    run_id = str(uuid.uuid4())
    order_id = "ORD-" + run_id[:8]
    policy = ctx.storage.get("policy") or BALANCED_POLICY
    providers: List[str] = ctx.storage.get("providers") or []
    pay_addr: str = ctx.storage.get("payment_addr")
    retire_addr: str = ctx.storage.get("retire_addr")
    bundler_addr: str = ctx.storage.get("bundler_addr")
    buyer_wallet = buyer_wallet or ctx.storage.get("buyer_wallet") or ""

    # 1) Intake intent — round target & add 2% over-provision
    target_raw = intent.tco2e
    target = round_tonnes(target_raw)
    planned_target = round_tonnes(target * 1.02)
    ctx.logger.info(f"[{order_id}] Intake: target={target}t, over_provision={planned_target}t, country={intent.facility_country}")

    # 2) Discovery — fan out quotes to providers via send_and_receive
    quote_req = QuoteRequest(
        tco2e=planned_target,
        policy_name=intent.policy_name,
        country_hint=intent.facility_country,
        period=intent.period,
    )
    responses: List[QuoteResponse] = []
    async def ask(provider_addr: str):
        reply, status = await ctx.send_and_receive(provider_addr, quote_req, response_type=QuoteResponse)
        if isinstance(reply, QuoteResponse):
            return reply
        ctx.logger.warning(f"[{order_id}] Provider {provider_addr} quote failed: {status}")
        return None

    provider_tasks = [ask(addr) for addr in providers]
    results = await asyncio.gather(*provider_tasks)
    for r in results:
        if r:
            responses.append(r)

    # Flatten lots & apply screening
    lots: List[Lot] = []
    for r in responses:
        lots.extend(r.lots)

    policy_c = policy["constraints"]
    min_vintage = policy["min_vintage"]
    price_cap = policy_c["price_ceiling_per_tonne"]

    screened = [
        lot for lot in lots
        if lot.vintage >= min_vintage and lot.unit_price_pyusd <= price_cap
    ]
    if not screened:
        ctx.logger.error(f"[{order_id}] No supply after screening (min_vintage={min_vintage}, price_cap={price_cap})")
        return

    # 3) Scoring
    weights = policy["scoring_weights"]
    current_year = time.gmtime().tm_year
    scored: List[Tuple[Lot, float]] = [
        (lot, score_lot(lot, intent.facility_country, price_cap, weights, current_year)) for lot in screened
    ]
    # 4) Greedy optimizer with share buckets (nature/tech)
    target_mix = policy["target_mix"]
    share_targets = {bucket["class"]: round_tonnes(planned_target * bucket["share"]) for bucket in target_mix}
    selected: List[BasketLine] = []

    # Index lots per class
    by_class: Dict[str, List[Tuple[Lot, float]]] = {}
    for lot, sc in scored:
        by_class.setdefault(lot.project_class, []).append((lot, sc))
    # Greedy: sort by score per unit price within each class
    for cls in by_class:
        by_class[cls].sort(key=lambda t: (t[1] / max(1e-6, t[0].unit_price_pyusd)), reverse=True)

    # Track per-project cap
    max_share_per_project = policy_c["max_share_per_project"]
    per_project_cap = planned_target * max_share_per_project
    project_taken: Dict[str, float] = {}

    for bucket in target_mix:
        cls = bucket["class"]
        need = share_targets.get(cls, 0.0)
        if need <= 0:
            continue
        for lot, sc in by_class.get(cls, []):
            if need <= 0:
                break
            # limit by availability, min lot, and per-project cap
            remaining_project_cap = per_project_cap - project_taken.get(lot.project_id, 0.0)
            if remaining_project_cap <= 0:
                continue
            take = min(need, lot.available_tonnes, remaining_project_cap)
            if take <= 0:
                continue
            take = round_tonnes(max(take, lot.min_lot_tonnes))
            if take <= 0:
                continue

            selected.append(BasketLine(
                provider_id=lot.provider_id,
                credit_id=lot.credit_id,
                project_id=lot.project_id,
                registry=lot.registry,
                cls=lot.project_class,
                country=lot.country,
                vintage=lot.vintage,
                tonnes=take,
                unit_price_pyusd=lot.unit_price_pyusd,
                score=sc
            ))
            need = round_tonnes(need - take)
            project_taken[lot.project_id] = project_taken.get(lot.project_id, 0.0) + take

    # Sanity: if under-filled due to caps, top-up from any class
    total_selected = round_tonnes(sum(line.tonnes for line in selected))
    if total_selected < planned_target:
        short = round_tonnes(planned_target - total_selected)
        fallback_pool = sorted(scored, key=lambda t: (t[1] / max(1e-6, t[0].unit_price_pyusd)), reverse=True)
        for lot, sc in fallback_pool:
            if short <= 0:
                break
            remaining_project_cap = per_project_cap - project_taken.get(lot.project_id, 0.0)
            take = min(short, lot.available_tonnes, max(0.0, remaining_project_cap))
            take = round_tonnes(max(take, lot.min_lot_tonnes))
            if take <= 0:
                continue
            selected.append(BasketLine(
                provider_id=lot.provider_id,
                credit_id=lot.credit_id,
                project_id=lot.project_id,
                registry=lot.registry,
                cls=lot.project_class,
                country=lot.country,
                vintage=lot.vintage,
                tonnes=take,
                unit_price_pyusd=lot.unit_price_pyusd,
                score=sc
            ))
            short = round_tonnes(short - take)
            project_taken[lot.project_id] = project_taken.get(lot.project_id, 0.0) + take

    expected_spend = round(sum(line.tonnes * line.unit_price_pyusd for line in selected), 4)
    draft = BasketDraft(
        target_tco2e=planned_target,
        lines=selected,
        expected_total_pyusd=expected_spend,
        policy_name=policy["name"]
    )
    ctx.logger.info(f"[{order_id}] Draft lines={len(selected)}, expected_spend={expected_spend} PYUSD")

    # 5) Reservation (batch by provider)
    # group lines per provider
    per_provider: Dict[str, List[ReserveLine]] = {}
    for line in selected:
        per_provider.setdefault(line.provider_id, []).append(ReserveLine(
            credit_id=line.credit_id, provider_id=line.provider_id, tonnes=line.tonnes
        ))

    holds: List[Hold] = []
    async def reserve_with(provider_addr: str, lines: List[ReserveLine]):
        req = ReserveRequest(lots=lines, hold_ttl=120)
        reply, status = await ctx.send_and_receive(provider_addr, req, response_type=ReserveResponse)
        if isinstance(reply, ReserveResponse):
            return reply.holds
        ctx.logger.warning(f"[{order_id}] Reserve failed with {provider_addr}: {status}")
        return []

    reserve_tasks = [reserve_with(addr, per_provider.get(addr, [])) for addr in providers]
    reserve_results = await asyncio.gather(*reserve_tasks)
    for hset in reserve_results:
        holds.extend(hset)

    if not holds:
        ctx.logger.error(f"[{order_id}] Reservation failed at all providers.")
        return

    # 6) Settlement
    spend_cap = expected_spend * 1.05  # small slippage buffer
    settle_req = SettleRequest(
        order_id=order_id,
        holds=holds,
        buyer_wallet=buyer_wallet,  # demo: use agent address
        spend_limit_pyusd=spend_cap
    )
    settle_reply, status = await ctx.send_and_receive(pay_addr, settle_req, response_type=SettleResponse)
    if not isinstance(settle_reply, SettleResponse):
        ctx.logger.error(f"[{order_id}] Settlement failed: {status}")
        return

    # 7) Proof bundle (draft-only fields; fills/tx below)
    bundle_req = ProofBundleRequest(
        order_id=order_id,
        basket_manifest=draft,
        fills=settle_reply.fills,
        retirement_txs=[],
        hash_algo="keccak256",
    )
    bundle_reply, status = await ctx.send_and_receive(bundler_addr, bundle_req, response_type=ProofBundleResponse)
    if not isinstance(bundle_reply, ProofBundleResponse):
        ctx.logger.error(f"[{order_id}] Proof bundling failed: {status}")
        return

    # 8) Retirement (if enabled)
    if policy.get("retire_immediately", True):
        retire_req = RetireRequest(
            fills=settle_reply.fills,
            retire_for_wallet=buyer_wallet,
            claim_text=f"GreenChain demo retirement for {intent.period} ({intent.factor_source})",
            proof_bundle_hash=bundle_reply.bundle_hash,
        )
        retire_reply, status = await ctx.send_and_receive(retire_addr, retire_req, response_type=RetireResponse)
        if not isinstance(retire_reply, RetireResponse):
            ctx.logger.error(f"[{order_id}] Retirement failed: {status}")
            return
    else:
        retire_reply = RetireResponse(retirements=[])

    # 9) Emit result & store
    trade = TradeResult(
        order_id=order_id,
        run_id=run_id,
        policy_name=policy["name"],
        target_tco2e=planned_target,
        basket=draft,
        reservation_holds=holds,
        settlement=settle_reply,
        retirements=retire_reply,
        proof_bundle_hash=bundle_reply.bundle_hash,
    )
    ctx.storage.set("last_trade", trade.dict())
    ctx.logger.info(f"[{order_id}] Done. Retired {sum(r.tonnes for r in retire_reply.retirements):.4f} t, proof={bundle_reply.bundle_hash}")
