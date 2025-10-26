#!/usr/bin/env python3
"""
GreenChain Bureau Runner
========================
Runs the uAgents Bureau separately from FastAPI.

This script should be run in a separate terminal/process alongside the FastAPI server.
It allows the agent network to process procurement requests properly.

Usage:
    Terminal 1: python runner_comprehensive.py  (FastAPI server)
    Terminal 2: python bureau_runner.py         (This Bureau)
"""

import os
from uagents import Bureau, Agent

# Import agent builders
from procurement.basket_curator_1 import build_agent as build_curator
from procurement.provider_adaptor_in import build_agent as build_provider_in
from procurement.provider_adaptor_np import build_agent as build_provider_np
from procurement.payment_orchestrator import build_agent as build_payment
from procurement.retirement_registrar import build_agent as build_retire
from procurement.proof_bundler import build_agent as build_bundler

# Configuration
BUREAU_PORT = int(os.getenv("BUREAU_PORT", "8001"))

def main():
    print("\n" + "=" * 80)
    print("GreenChain Procurement Bureau")
    print("=" * 80)
    print("\nInitializing agent network...")
    
    # Build all agents
    prov_in = build_provider_in(name="ProviderIN", seed="seed provider in 1/2/3")
    prov_np = build_provider_np(name="ProviderNP", seed="seed provider np 1/2/3")
    payment = build_payment(name="PaymentOrchestrator", seed="seed payment 1/2/3")
    registrar = build_retire(name="RetirementRegistrar", seed="seed retire 1/2/3")
    bundler = build_bundler(name="ProofBundler", seed="seed bundler 1/2/3")
    
    # Build curator WITHOUT demo intent - will receive via network
    curator = build_curator(
        name="BasketCurator",
        seed="seed curator 1/2/3",
        provider_addresses=[prov_in.address, prov_np.address],
        payment_orchestrator_addr=payment.address,
        retirement_registrar_addr=registrar.address,
        proof_bundler_addr=bundler.address,
        demo_intent=None,  # No demo - waits for API triggers
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
            raise RuntimeError(f"{name} built as None")
        if not isinstance(agent, Agent):
            raise RuntimeError(f"{name} is not a uAgents.Agent")
        print(f"[✓] {name}: {agent.address}")
    
    # Create Bureau
    bureau = Bureau(
        port=BUREAU_PORT,
        agents=list(agents.values()),
        endpoint=f"http://0.0.0.0:{BUREAU_PORT}/submit",
    )
    
    print(f"\n" + "=" * 80)
    print(f"Bureau running on port {BUREAU_PORT}")
    print(f"Endpoint: http://0.0.0.0:{BUREAU_PORT}/submit")
    print(f"Curator address: {curator.address}")
    print("=" * 80)
    print("\n⚡ Bureau is ready to process procurement requests")
    print("   Waiting for FootprintIntent messages from FastAPI...\n")
    
    # Run Bureau (this blocks)
    bureau.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nShutting down Bureau...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise