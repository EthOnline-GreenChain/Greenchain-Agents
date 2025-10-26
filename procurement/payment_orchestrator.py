# payment_orchestrator.py
from __future__ import annotations
import uuid
from typing import List
from uagents import Agent, Context, Model, Protocol

class Hold(Model):
    hold_id: str
    provider_id: str
    credit_id: str
    tonnes: float
    unit_price: float
    expires_at: str

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

SETTLE_PROTO = "curation.settle.v1"

def build_agent(name: str, seed: str) -> Agent:
    agent = Agent(name=name, seed=seed)
    proto = Protocol(name=SETTLE_PROTO)

    @proto.on_message(model=SettleRequest, replies=SettleResponse)
    async def settle(ctx: Context, sender: str, msg: SettleRequest):
        # Demo settlement: assume all holds fill successfully
        total = sum(h.tonnes * h.unit_price for h in msg.holds)
        if total > msg.spend_limit_pyusd:
            await ctx.send(sender, SettleResponse(tx_hashes=[], fills=[]))
            return

        txs: List[str] = []
        fills: List[Fill] = []
        for h in msg.holds:
            txs.append("0xTX" + uuid.uuid4().hex[:12])
            fills.append(Fill(credit_id=h.credit_id, tonnes=h.tonnes, unit_price=h.unit_price, hold_id=h.hold_id))
        ctx.logger.info(f"[{msg.order_id}] Settled {len(fills)} fills, spend={total:.4f} PYUSD")
        await ctx.send(sender, SettleResponse(tx_hashes=txs, fills=fills))

    agent.include(proto)
    return agent
