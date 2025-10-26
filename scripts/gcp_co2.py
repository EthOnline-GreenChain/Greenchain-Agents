
#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, time, requests
from typing import Dict, Any

# We estimate the operational emissions of this instance using a simplified version of
# Cloud Carbon Footprint methodology:
#   kWh = (vCPU energy + Memory energy) × PUE
#   CO2e (kg) = kWh × (grid gCO2e/kWh / 1000)
# References:
# - Grid intensity by region: https://docs.cloud.google.com/sustainability/region-carbon
# - Methodology + coefficients (PUE 1.1, memory kWh/GB-hour 0.000392): https://www.cloudcarbonfootprint.org/docs/methodology/

REGION_GRID_INTENSITY = {
    # gCO2e/kWh (2024 averages, last updated 2025-10-24 in Google doc)
    "us-central1": 413,
    "us-east1": 576,
    "us-east4": 323,
    "us-west1": 79,
    "us-west2": 169,
    "us-west3": 555,
    "us-west4": 357,
    "europe-west1": 103,
    "europe-west2": 106,
    "europe-west3": 276,
    "europe-west4": 209,
    "europe-west6": 15,
    "europe-west9": 16,
    "europe-north1": 39,
    "europe-north2": 3,
    "northamerica-northeast1": 5,
    "northamerica-northeast2": 59,
    "asia-southeast1": 367,
    "asia-southeast2": 561,
    "asia-east1": 439,
    "asia-northeast1": 453,
    "australia-southeast1": 498,
    "southamerica-east1": 67,
}

PUE = 1.1  # GCP average per CCF
MEM_KWH_PER_GB_HOUR = 0.000392  # per CCF Appendix I
MIN_W_VCPU = 0.71   # median min watts per vCPU (GCP)
MAX_W_VCPU = 4.26   # median max watts per vCPU (GCP)

def get_metadata(path: str, timeout=2.0) -> str:
    url = f"http://metadata.google.internal/computeMetadata/v1/{path}"
    headers = {"Metadata-Flavor": "Google"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text

def detect_instance_context() -> Dict[str, str]:
    try:
        project_id = get_metadata("project/project-id")
        zone_path = get_metadata("instance/zone")
        machine_type_path = get_metadata("instance/machine-type")
        name = get_metadata("instance/name")
    except Exception:
        return {}
    zone = zone_path.split("/")[-1]            # e.g., us-central1-a
    region = "-".join(zone.split("-")[:2])     # e.g., us-central1
    machine_type = machine_type_path.split("/")[-1]  # e.g., n1-standard-4
    return {"project_id": project_id, "zone": zone, "region": region, "machine_type": machine_type, "name": name}

def get_machine_specs(project_id: str, zone: str, machine_type: str) -> Dict[str, Any]:
    """Try to query Compute API for exact vCPU + memory; fall back to env overrides."""
    try:
        from googleapiclient import discovery  # type: ignore
        from google.auth.transport.requests import Request  # type: ignore
        import google.auth  # type: ignore

        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        service = discovery.build("compute", "v1", credentials=creds, cache_discovery=False)
        req = service.machineTypes().get(project=project_id, zone=zone, machineType=machine_type)
        mt = req.execute()
        vcpus = int(mt.get("guestCpus", 1))
        mem_gib = float(mt.get("memoryMb", 1024)) / 1024.0
        return {"vcpus": vcpus, "memory_gib": mem_gib}
    except Exception:
        # Fallback to env variables or defaults
        vcpus = int(os.getenv("GC_VCPU", "2"))
        mem_gib = float(os.getenv("GC_MEM_GIB", "4"))
        return {"vcpus": vcpus, "memory_gib": mem_gib}

def estimate_kwh(hours: float, vcpus: int, mem_gib: float, cpu_util: float) -> float:
    # Compute energy (per vCPU-hour) using linear interpolation between min/max watts
    avg_w_per_vcpu = MIN_W_VCPU + (MAX_W_VCPU - MIN_W_VCPU) * cpu_util
    kwh_compute = vcpus * (avg_w_per_vcpu / 1000.0) * hours
    kwh_memory = mem_gib * MEM_KWH_PER_GB_HOUR * hours
    return (kwh_compute + kwh_memory) * PUE

def main():
    ap = argparse.ArgumentParser(description="Estimate the CO2e (operational) for this GCE instance")
    ap.add_argument("--hours", type=float, default=1.0, help="Estimation window in hours")
    ap.add_argument("--avg-util", type=float, default=0.3, help="Average CPU utilization (0..1) if monitoring is unavailable")
    args = ap.parse_args()

    ctx = detect_instance_context()
    if not ctx:
        print(json.dumps({"error": "Not on GCE or metadata unavailable"}))
        return

    specs = get_machine_specs(ctx["project_id"], ctx["zone"], ctx["machine_type"])
    region = ctx["region"]
    intensity = REGION_GRID_INTENSITY.get(region, 400.0)  # gCO2/kWh fallback

    kwh = estimate_kwh(hours=args.hours, vcpus=specs["vcpus"], mem_gib=specs["memory_gib"], cpu_util=args.avg_util)
    co2e_kg = kwh * (intensity / 1000.0)

    output = {
        "schema_version": "greenchain.ingest.v1",
        "source": {"source_type": "gcp", "filename": None, "sha256": None, "received_at": None, "extra": ctx},
        "gcp_footprint": {
            "region": region,
            "hours": args.hours,
            "vcpu_count": specs["vcpus"],
            "memory_gib": round(specs["memory_gib"], 2),
            "cpu_utilization": args.avg_util,
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
    }
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
