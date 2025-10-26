# proof_bundler.py
from __future__ import annotations
import json, hashlib
from uagents import Agent, Context, Model, Protocol

# Reuse minimal simple structures
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
    lines: list[BasketLine]
    expected_total_pyusd: float
    policy_name: str

class Fill(Model):
    credit_id: str
    tonnes: float
    unit_price: float
    hold_id: str

class ProofBundleRequest(Model):
    order_id: str
    basket_manifest: BasketDraft
    provider_docs: list[dict] = []
    fills: list[Fill] = []
    retirement_txs: list[str] = []
    attestations: list[dict] = []
    hash_algo: str = "keccak256"

class ProofBundleResponse(Model):
    bundle_hash: str

def build_agent(name: str, seed: str) -> Agent:
    agent = Agent(name=name, seed=seed)
    proto = Protocol(name="curation.proofbundle.v1")

    @proto.on_message(model=ProofBundleRequest, replies=ProofBundleResponse)
    async def bundle(ctx: Context, sender: str, msg: ProofBundleRequest):
        # Minimal canonical bundle – hashes only (no PII, no raw invoices)
        bundle = {
            "order_id": msg.order_id,
            "basket_manifest": msg.basket_manifest.dict(),
            "fills": [f.dict() for f in msg.fills],
            "retirement_txs": msg.retirement_txs,
            "attestations": msg.attestations,
            "hash_algo": msg.hash_algo,
        }
        # Use SHA3-256 (Keccak family) for demo; produces 0x-prefixed hex
        h = hashlib.sha3_256(json.dumps(bundle, sort_keys=True).encode("utf-8")).hexdigest()
        bundle_hash = "0x" + h
        ctx.logger.info(f"[{msg.order_id}] Proof bundle hash={bundle_hash[:18]}…")
        await ctx.send(sender, ProofBundleResponse(bundle_hash=bundle_hash))

    agent.include(proto)
    return agent
