# runner.py
import json
from uagents import Bureau

from basket_curator import build_agent as build_curator, FootprintIntent
from provider_adaptor_in import build_agent as build_provider_in
from provider_adaptor_np import build_agent as build_provider_np
from payment_orchestrator import build_agent as build_payment
from retirement_registrar import build_agent as build_retire
from proof_bundler import build_agent as build_bundler

# ---- Build all agents ----
prov_in = build_provider_in(name="ProviderIN", seed="seed provider in 1/2/3")
prov_np = build_provider_np(name="ProviderNP", seed="seed provider np 1/2/3")
payment = build_payment(name="PaymentOrchestrator", seed="seed payment 1/2/3")
registrar = build_retire(name="RetirementRegistrar", seed="seed retire 1/2/3")
bundler = build_bundler(name="ProofBundler", seed="seed bundler 1/2/3")

# Demo input from your ingest example
intent = FootprintIntent(
    tco2e=0.31955,
    period="2022-07",
    facility_country="IN",
    factor_source="CEA 2022-23",
    scope=["scope2_location"],
    currency_token="PYUSD",
    policy_name="Balanced quality v1",
    auto_execute=True,
)

curator = build_curator(
    name="BasketCurator",
    seed="seed curator 1/2/3",
    provider_addresses=[prov_in.address, prov_np.address],
    payment_orchestrator_addr=payment.address,
    retirement_registrar_addr=registrar.address,
    proof_bundler_addr=bundler.address,
    demo_intent=intent,
)

# runner.py (right before you construct the Bureau)
from uagents import Agent

agents_named = [
    ("curator", curator),
    ("prov_in", prov_in),
    ("prov_np", prov_np),
    ("payment", payment),
    ("registrar", registrar),
    ("bundler", bundler),
]

for name, a in agents_named:
    if a is None:
        raise RuntimeError(f"{name} built as None. Check its build_agent() return.")
    if not isinstance(a, Agent):
        raise RuntimeError(f"{name} is not a uAgents.Agent. Got {type(a)}")
    print(f"[OK] {name}: {a.address}")

bureau = Bureau(
    port=8000,
    agents=[a for _, a in agents_named],
    endpoint="http://0.0.0.0:8000/submit",
)



if __name__ == "__main__":
    bureau.run()
