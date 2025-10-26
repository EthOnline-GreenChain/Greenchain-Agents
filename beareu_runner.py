#!/usr/bin/env python3
"""
Bureau Runner - Separate Process for Agent Execution
====================================================
Run this in a separate terminal/process from runner_integrated_fixed.py

This script runs the Bureau agent network that actually processes
procurement requests.

Usage:
    Terminal 1: python runner_integrated_fixed.py
    Terminal 2: python bureau_runner.py
"""

import sys
import os

# Add current directory to Python path to find modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uagents import Bureau
import os

# Configuration
BUREAU_PORT = int(os.getenv("BUREAU_PORT", "8001"))

print("=" * 80)
print("GreenChain Bureau - Agent Network")
print("=" * 80)
print(f"Starting agent network on port {BUREAU_PORT}")
print("This will process procurement requests from the main API")
print("=" * 80)
print()

# Import agents with error handling
print("Initializing agents...")
try:
    from procurement.basket_curator import build_agent as build_curator, FootprintIntent
    from procurement.provider_adaptor_in import build_agent as build_provider_in
    from procurement.provider_adaptor_np import build_agent as build_provider_np
    from procurement.payment_orchestrator import build_agent as build_payment
    from procurement.retirement_registrar import build_agent as build_retire
    from procurement.proof_bundler import build_agent as build_bundler
except ImportError as e:
    print(f"❌ Error importing agent modules: {e}")
    print("\nMake sure these files exist in the current directory:")
    print("  - basket_curator.py")
    print("  - provider_adaptor_in.py")
    print("  - provider_adaptor_np.py")
    print("  - payment_orchestrator.py")
    print("  - retirement_registrar.py")
    print("  - proof_bundler.py")
    sys.exit(1)

# Build all agents
prov_in = build_provider_in(name="ProviderIN", seed="seed provider in 1/2/3")
prov_np = build_provider_np(name="ProviderNP", seed="seed provider np 1/2/3")
payment = build_payment(name="PaymentOrchestrator", seed="seed payment 1/2/3")
registrar = build_retire(name="RetirementRegistrar", seed="seed retire 1/2/3")
bundler = build_bundler(name="ProofBundler", seed="seed bundler 1/2/3")

# For testing, you can add a demo intent here
# Uncomment these lines to test with a demo procurement:
"""
demo_intent = FootprintIntent(
    tco2e=0.3157,
    period="2022-06",
    facility_country="IN",
    factor_source="CEA",
    scope=["scope2_location"],
    currency_token="PYUSD",
    policy_name="Balanced quality v1",
    auto_execute=True,
)
"""

curator = build_curator(
    name="BasketCurator",
    seed="seed curator 1/2/3",
    provider_addresses=[prov_in.address, prov_np.address],
    payment_orchestrator_addr=payment.address,
    retirement_registrar_addr=registrar.address,
    proof_bundler_addr=bundler.address,
    demo_intent=None,  # Change to demo_intent to test
)

# Validate agents
agents = {
    "curator": curator,
    "prov_in": prov_in,
    "prov_np": prov_np,
    "payment": payment,
    "registrar": registrar,
    "bundler": bundler,
}

for name, agent in agents.items():
    if agent is None:
        raise RuntimeError(f"{name} agent is None")
    print(f"✓ {name}: {agent.address}")

print()
print("Creating Bureau...")

# Create Bureau
bureau = Bureau(
    port=BUREAU_PORT,
    agents=list(agents.values()),
    endpoint=f"http://0.0.0.0:{BUREAU_PORT}/submit",
)

print(f"✓ Bureau created on port {BUREAU_PORT}")
print()
print("=" * 80)
print("Bureau is now running and will process procurement requests")
print()
print("NOTE: Currently there is NO automatic integration between")
print("      the API requests and Bureau agents.")
print()
print("To test Bureau directly, uncomment the demo_intent lines above")
print("and restart this script. You'll see the agents process the demo.")
print()
print("Keep this terminal open while using the API")
print("Press CTRL+C to stop")
print("=" * 80)
print()

if __name__ == "__main__":
    bureau.run()