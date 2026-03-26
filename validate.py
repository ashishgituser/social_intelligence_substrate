#!/usr/bin/env python3
"""Pre-submission validation script for Social Intelligence Substrate.

Runs ALL checks from the hackathon checklist against either a local server
or a deployed HF Space URL.  Exit code 0 = all pass, 1 = failures found.

Usage:
    python validate.py                           # test local (http://localhost:7860)
    python validate.py --url https://xxx.hf.space # test deployed Space
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx")
    sys.exit(1)

# ── Colour helpers for terminal output ──────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓ PASS{RESET}  {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗ FAIL{RESET}  {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠ WARN{RESET}  {msg}")


def section(title: str) -> None:
    print(f"\n{BOLD}{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}{RESET}")


# ── Checks ──────────────────────────────────────────────────────────
failures: List[str] = []


def check(condition: bool, pass_msg: str, fail_msg: str) -> bool:
    if condition:
        ok(pass_msg)
    else:
        fail(fail_msg)
        failures.append(fail_msg)
    return condition


def validate_local_files() -> None:
    """Check that required files exist in the repo."""
    section("1. Repository Structure")
    root = Path(__file__).parent

    check(
        (root / "openenv.yaml").exists(),
        "openenv.yaml exists",
        "openenv.yaml MISSING — required by OpenEnv spec",
    )
    check(
        (root / "Dockerfile").exists(),
        "Dockerfile exists",
        "Dockerfile MISSING — required for HF Spaces Docker SDK",
    )
    check(
        (root / "requirements.txt").exists(),
        "requirements.txt exists",
        "requirements.txt MISSING",
    )
    check(
        (root / "README.md").exists(),
        "README.md exists",
        "README.md MISSING",
    )
    check(
        (root / "app" / "server.py").exists(),
        "app/server.py exists",
        "app/server.py MISSING — main entrypoint",
    )
    check(
        (root / "app" / "baseline.py").exists(),
        "app/baseline.py exists",
        "app/baseline.py MISSING — inference script",
    )
    check(
        (root / "app" / "grader.py").exists(),
        "app/grader.py exists",
        "app/grader.py MISSING — grader module",
    )

    # HF Spaces YAML front-matter
    readme = (root / "README.md").read_text(encoding="utf-8")
    check(
        readme.startswith("---"),
        "README.md has HF Spaces YAML front-matter",
        "README.md missing HF Spaces YAML front-matter (---\\ntitle: ...\\nsdk: docker\\n---)",
    )
    check(
        "sdk: docker" in readme,
        "README.md specifies sdk: docker",
        "README.md missing 'sdk: docker' in front-matter",
    )
    check(
        "openenv" in readme.lower(),
        "README.md references 'openenv'",
        "README.md should reference 'openenv' tag",
    )

    # .env should NOT be committed
    gitignore_path = root / ".gitignore"
    if gitignore_path.exists():
        gitignore = gitignore_path.read_text(encoding="utf-8")
        check(
            ".env" in gitignore,
            ".env is in .gitignore (API key safe)",
            ".env NOT in .gitignore — API key will leak!",
        )
    else:
        fail(".gitignore MISSING — .env may leak")
        failures.append(".gitignore MISSING")


def validate_openenv_yaml() -> None:
    """Validate openenv.yaml structure."""
    section("2. OpenEnv Spec (openenv.yaml)")
    root = Path(__file__).parent
    yaml_path = root / "openenv.yaml"
    if not yaml_path.exists():
        fail("Cannot validate — openenv.yaml missing")
        return

    content = yaml_path.read_text(encoding="utf-8")
    required_fields = ["name", "version", "environment", "observation", "action", "reward", "tasks"]
    for field in required_fields:
        check(
            f"{field}:" in content or f"{field} :" in content,
            f"openenv.yaml has '{field}' field",
            f"openenv.yaml MISSING '{field}' field",
        )

    check(
        "entrypoint:" in content,
        "openenv.yaml has 'entrypoint' in environment",
        "openenv.yaml MISSING 'entrypoint'",
    )
    check(
        "7860" in content,
        "openenv.yaml specifies port 7860",
        "openenv.yaml should specify port 7860",
    )


def validate_health(base: str, client: httpx.Client) -> bool:
    """GET / must return 200."""
    section("3. Health Check (GET /)")
    try:
        r = client.get(f"{base}/")
        passed = check(
            r.status_code == 200,
            f"GET / → {r.status_code}",
            f"GET / → {r.status_code} (expected 200)",
        )
        if passed:
            data = r.json()
            check(
                data.get("status") == "running",
                "Response has status='running'",
                f"Response status={data.get('status')} (expected 'running')",
            )
        return passed
    except Exception as e:
        fail(f"GET / failed: {e}")
        failures.append(f"GET / failed: {e}")
        return False


def validate_reset(base: str, client: httpx.Client) -> bool:
    """POST /reset must return a valid Observation."""
    section("4. Reset Endpoint (POST /reset)")
    try:
        r = client.post(f"{base}/reset", json={"task_id": "resource_acquisition", "seed": 42})
        passed = check(
            r.status_code == 200,
            f"POST /reset → {r.status_code}",
            f"POST /reset → {r.status_code} (expected 200)",
        )
        if not passed:
            return False
        obs = r.json()
        check("agent_id" in obs, "Observation has 'agent_id'", "Observation missing 'agent_id'")
        check("step_number" in obs, "Observation has 'step_number'", "Observation missing 'step_number'")
        check("own_resources" in obs, "Observation has 'own_resources'", "Observation missing 'own_resources'")
        check("visible_agents" in obs, "Observation has 'visible_agents'", "Observation missing 'visible_agents'")
        check("trade_history" in obs, "Observation has 'trade_history'", "Observation missing 'trade_history'")
        check("market_stats" in obs, "Observation has 'market_stats'", "Observation missing 'market_stats'")
        check(obs.get("step_number") == 0, "step_number == 0 after reset", f"step_number={obs.get('step_number')}")
        return True
    except Exception as e:
        fail(f"POST /reset failed: {e}")
        failures.append(f"POST /reset failed: {e}")
        return False


def validate_step(base: str, client: httpx.Client) -> bool:
    """POST /step must process actions and return StepResult."""
    section("5. Step Endpoint (POST /step)")
    try:
        # Reset first
        client.post(f"{base}/reset", json={"task_id": "resource_acquisition", "seed": 42})

        # OBSERVE
        r = client.post(f"{base}/step", json={"action_type": "OBSERVE"})
        passed = check(r.status_code == 200, "OBSERVE → 200", f"OBSERVE → {r.status_code}")
        if not passed:
            return False
        data = r.json()
        check("observation" in data, "StepResult has 'observation'", "StepResult missing 'observation'")
        check("reward" in data, "StepResult has 'reward'", "StepResult missing 'reward'")
        check("done" in data, "StepResult has 'done'", "StepResult missing 'done'")
        check(data["observation"]["step_number"] == 1, "step incremented to 1", "step not incremented")

        # PROPOSE_TRADE
        r2 = client.post(f"{base}/step", json={
            "action_type": "PROPOSE_TRADE",
            "target_agent": "provider_alpha",
            "offer_resources": {"compute": 2},
            "request_resources": {"storage": 2},
        })
        check(r2.status_code == 200, "PROPOSE_TRADE → 200", f"PROPOSE_TRADE → {r2.status_code}")
        return True
    except Exception as e:
        fail(f"POST /step failed: {e}")
        failures.append(f"POST /step failed: {e}")
        return False


def validate_state(base: str, client: httpx.Client) -> bool:
    """GET /state must return EnvironmentState."""
    section("6. State Endpoint (GET /state)")
    try:
        client.post(f"{base}/reset", json={"task_id": "resource_acquisition", "seed": 42})
        r = client.get(f"{base}/state")
        passed = check(r.status_code == 200, f"GET /state → {r.status_code}", f"GET /state → {r.status_code}")
        if passed:
            data = r.json()
            check("graph_nodes" in data, "State has 'graph_nodes'", "State missing 'graph_nodes'")
            check("metrics" in data, "State has 'metrics'", "State missing 'metrics'")
        return passed
    except Exception as e:
        fail(f"GET /state failed: {e}")
        failures.append(f"GET /state failed: {e}")
        return False


def validate_tasks(base: str, client: httpx.Client) -> bool:
    """GET /tasks must return 3+ tasks with action schema."""
    section("7. Tasks Endpoint (GET /tasks)")
    try:
        r = client.get(f"{base}/tasks")
        passed = check(r.status_code == 200, f"GET /tasks → {r.status_code}", f"GET /tasks → {r.status_code}")
        if not passed:
            return False
        tasks = r.json()
        check(isinstance(tasks, list), "Returns a list", f"Returns {type(tasks).__name__}")
        check(
            len(tasks) >= 3,
            f"Has {len(tasks)} tasks (≥ 3 required)",
            f"Only {len(tasks)} tasks (need ≥ 3)",
        )
        for t in tasks:
            check("task_id" in t, f"Task '{t.get('name', '?')}' has task_id", "Task missing task_id")
            check("action_schema" in t, f"Task '{t.get('name', '?')}' has action_schema", "Task missing action_schema")
            check(
                len(t.get("action_schema", [])) >= 1,
                f"Task '{t.get('name', '?')}' action_schema has {len(t.get('action_schema',[]))} entries",
                "Task action_schema empty",
            )
        return True
    except Exception as e:
        fail(f"GET /tasks failed: {e}")
        failures.append(f"GET /tasks failed: {e}")
        return False


def validate_grader(base: str, client: httpx.Client) -> bool:
    """GET /grader must return score in [0.0, 1.0] after a completed episode."""
    section("8. Grader Endpoint (GET /grader)")
    try:
        client.post(f"{base}/reset", json={"task_id": "resource_acquisition", "seed": 42})
        # Run episode to completion
        for _ in range(25):
            r = client.post(f"{base}/step", json={"action_type": "OBSERVE"})
            if r.json().get("done"):
                break

        r = client.get(f"{base}/grader")
        passed = check(r.status_code == 200, f"GET /grader → {r.status_code}", f"GET /grader → {r.status_code}")
        if not passed:
            return False
        data = r.json()
        score = data.get("score", -1)
        check(0.0 <= score <= 1.0, f"Score {score:.4f} in [0.0, 1.0]", f"Score {score} out of range")
        check("breakdown" in data, "Has 'breakdown'", "Missing 'breakdown'")
        check("passed" in data, "Has 'passed' field", "Missing 'passed' field")
        return True
    except Exception as e:
        fail(f"GET /grader failed: {e}")
        failures.append(f"GET /grader failed: {e}")
        return False


def validate_baseline(base: str, client: httpx.Client) -> bool:
    """POST /baseline must return scores for all tasks."""
    section("9. Baseline Endpoint (POST /baseline)")
    try:
        r = client.post(f"{base}/baseline", timeout=120.0)
        passed = check(r.status_code == 200, f"POST /baseline → {r.status_code}", f"POST /baseline → {r.status_code}")
        if not passed:
            return False
        results = r.json()
        check(isinstance(results, list), "Returns a list", f"Returns {type(results).__name__}")
        check(len(results) >= 3, f"{len(results)} task results (≥ 3)", f"Only {len(results)} results")
        for br in results:
            tid = br.get("task_id", "?")
            score = br.get("score", -1)
            check(0.0 <= score <= 1.0, f"{tid}: score={score:.4f}", f"{tid}: score={score} out of range")
            check(br.get("steps_taken", 0) > 0, f"{tid}: steps_taken={br.get('steps_taken')}", f"{tid}: 0 steps taken")
        return True
    except Exception as e:
        fail(f"POST /baseline failed: {e}")
        failures.append(f"POST /baseline failed: {e}")
        return False


def validate_full_lifecycle(base: str, client: httpx.Client) -> bool:
    """Run a full episode: reset → step*N → grader  for each task."""
    section("10. Full Lifecycle (all tasks)")
    try:
        tasks_r = client.get(f"{base}/tasks")
        task_ids = [t["task_id"] for t in tasks_r.json()]
        all_ok = True
        for tid in task_ids:
            # Reset
            r = client.post(f"{base}/reset", json={"task_id": tid, "seed": 42})
            if r.status_code != 200:
                fail(f"{tid}: reset failed ({r.status_code})")
                failures.append(f"{tid}: reset failed")
                all_ok = False
                continue

            # Step through entire episode
            done = False
            steps = 0
            while not done and steps < 60:
                sr = client.post(f"{base}/step", json={"action_type": "OBSERVE"})
                if sr.status_code != 200:
                    break
                done = sr.json().get("done", False)
                steps += 1

            # Grader
            gr = client.get(f"{base}/grader")
            if gr.status_code == 200:
                score = gr.json().get("score", -1)
                passed = 0.0 <= score <= 1.0
                check(passed, f"{tid}: lifecycle OK (score={score:.4f}, {steps} steps)", f"{tid}: invalid score {score}")
                if not passed:
                    all_ok = False
            else:
                fail(f"{tid}: grader returned {gr.status_code}")
                failures.append(f"{tid}: grader failed")
                all_ok = False
        return all_ok
    except Exception as e:
        fail(f"Lifecycle test failed: {e}")
        failures.append(f"Lifecycle test failed: {e}")
        return False


def validate_determinism(base: str, client: httpx.Client) -> bool:
    """Two runs with identical seed must produce identical grader scores."""
    section("11. Determinism Check")
    try:
        scores = []
        for run in range(2):
            client.post(f"{base}/reset", json={"task_id": "resource_acquisition", "seed": 42})
            for _ in range(25):
                r = client.post(f"{base}/step", json={"action_type": "OBSERVE"})
                if r.json().get("done"):
                    break
            gr = client.get(f"{base}/grader").json()
            scores.append(gr.get("score"))

        check(
            scores[0] == scores[1],
            f"Deterministic: run1={scores[0]:.4f} == run2={scores[1]:.4f}",
            f"NON-DETERMINISTIC: run1={scores[0]} ≠ run2={scores[1]}",
        )
        return scores[0] == scores[1]
    except Exception as e:
        fail(f"Determinism check failed: {e}")
        failures.append(f"Determinism check failed: {e}")
        return False


# ── Main ────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-submission validator")
    parser.add_argument(
        "--url",
        default="http://localhost:7860",
        help="Base URL to test (default: http://localhost:7860)",
    )
    parser.add_argument(
        "--skip-local",
        action="store_true",
        help="Skip local file checks (use when testing remote Space only)",
    )
    args = parser.parse_args()
    base = args.url.rstrip("/")

    print(f"\n{BOLD}{'═' * 60}")
    print(f"  SOCIAL INTELLIGENCE SUBSTRATE — PRE-SUBMISSION VALIDATOR")
    print(f"  Target: {base}")
    print(f"{'═' * 60}{RESET}")

    # Local file checks
    if not args.skip_local:
        validate_local_files()
        validate_openenv_yaml()

    # Wait for server to be reachable
    print(f"\n  Connecting to {base} ...")
    client = httpx.Client(timeout=30.0)
    for attempt in range(5):
        try:
            client.get(f"{base}/")
            print(f"  Connected!\n")
            break
        except Exception:
            if attempt < 4:
                time.sleep(2)
            else:
                fail(f"Cannot reach {base} after 5 attempts")
                failures.append("Server unreachable")
                _summary()
                sys.exit(1)

    # Remote endpoint checks
    validate_health(base, client)
    validate_reset(base, client)
    validate_step(base, client)
    validate_state(base, client)
    validate_tasks(base, client)
    validate_grader(base, client)
    validate_baseline(base, client)
    validate_full_lifecycle(base, client)
    validate_determinism(base, client)

    client.close()
    _summary()


def _summary() -> None:
    section("SUMMARY")
    total_checks = len(failures)
    if failures:
        print(f"\n  {RED}{BOLD}{total_checks} FAILURE(S):{RESET}")
        for f in failures:
            print(f"    • {f}")
        print(f"\n  {RED}❌ NOT READY FOR SUBMISSION{RESET}\n")
        sys.exit(1)
    else:
        print(f"\n  {GREEN}{BOLD}ALL CHECKS PASSED ✓{RESET}")
        print(f"  {GREEN}✅ READY FOR SUBMISSION{RESET}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
