# retirement_registrar.py
from __future__ import annotations
import uuid
from typing import List
from uagents import Agent, Context, Model, Protocol

class Fill(Model):
    credit_id: str
    tonnes: float
    unit_price: float
    hold_id: str

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

RETIRE_PROTO = "curation.retire.v1"

def build_agent(name: str, seed: str) -> Agent:
    agent = Agent(name=name, seed=seed)
    proto = Protocol(name=RETIRE_PROTO)

    @proto.on_message(model=RetireRequest, replies=RetireResponse)
    async def retire(ctx: Context, sender: str, msg: RetireRequest):
        rets: List[Retirement] = []
        for f in msg.fills:
            rets.append(Retirement(
                credit_id=f.credit_id,
                tonnes=f.tonnes,
                tx_hash="0xRET" + uuid.uuid4().hex[:12],
                proof_hash=msg.proof_bundle_hash
            ))
        ctx.logger.info(f"Retired {sum(r.tonnes for r in rets):.4f} t for {msg.retire_for_wallet}")
        await ctx.send(sender, RetireResponse(retirements=rets))

    agent.include(proto)
    return agent
