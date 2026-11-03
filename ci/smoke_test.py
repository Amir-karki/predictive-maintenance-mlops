"""Check a running API container from the outside: it starts, loads a model, and answers.

Run from the project root while the container is running:
    python -m ci.smoke_test
    """

import sys
import time
import httpx

from turbofan.serve import CycleReading

BASE_URL = "http://127.0.0.1:8000"
STARTUP_TIMEOUT = 60 # seconds

def wait_until_healthy() -> dict:
    """Poll /health until the API answers, or give up after STARTUP_TIMEOUT."""
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        try: 
            response = httpx.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
        except httpx.TransportError:
            pass
        time.sleep(2)
    sys.exit(f"API did not become healthy within {STARTUP_TIMEOUT} s")



def make_request(n_cycles: int = 30, unit: int = 7) -> dict:
    """A valid request with every sensor CycleReading asks for."""
    cycles = []
    for c in range(1, n_cycles + 1):
        reading = {"cycle": c}
        for name, field in CycleReading.model_fields.items():
            if name != "cycle":
                reading[name] = 1 if field.annotation is int else 1.0
        cycles.append(reading)
    return {"unit": unit, "cycles": cycles}

def main() -> None:
    health = wait_until_healthy()
    print("health:", health)
    assert health["source"] == "folder", health

    response = httpx.post(f"{BASE_URL}/predict", json=make_request(), timeout=10)
    print("predict:", response.status_code, response.text)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["unit"] == 7 and body["at_cycle"] == 30, body
    print("Smoke test passed")


if __name__ == "__main__":
    main()