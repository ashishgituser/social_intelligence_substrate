"""Tests for the FastAPI server endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.server import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["spec"] == "openenv"
        assert data["status"] == "running"


class TestResetEndpoint:
    def test_reset_default(self, client):
        resp = client.post("/reset", json={"task_id": "resource_acquisition", "seed": 42})
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "player"
        assert data["step_number"] == 0

    def test_reset_all_tasks(self, client):
        for task_id in ["resource_acquisition", "coalition_building",
                        "adversarial_resilience", "market_manipulation"]:
            resp = client.post("/reset", json={"task_id": task_id, "seed": 42})
            assert resp.status_code == 200
            assert resp.json()["task_id"] == task_id

    def test_reset_invalid_task(self, client):
        resp = client.post("/reset", json={"task_id": "invalid_task", "seed": 42})
        assert resp.status_code == 400


class TestStepEndpoint:
    def test_step_without_reset_fails(self, client):
        # Fresh app state — need a new client to guarantee no _env
        from app import server
        server._env = None
        resp = client.post("/step", json={"action_type": "OBSERVE"})
        assert resp.status_code == 400

    def test_step_observe(self, client):
        client.post("/reset", json={"task_id": "resource_acquisition", "seed": 42})
        resp = client.post("/step", json={"action_type": "OBSERVE"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["observation"]["step_number"] == 1
        assert "reward" in data

    def test_step_propose_trade(self, client):
        client.post("/reset", json={"task_id": "resource_acquisition", "seed": 42})
        resp = client.post("/step", json={
            "action_type": "PROPOSE_TRADE",
            "target_agent": "provider_alpha",
            "offer_resources": {"compute": 2},
            "request_resources": {"storage": 2},
        })
        assert resp.status_code == 200


class TestStateEndpoint:
    def test_state_after_reset(self, client):
        client.post("/reset", json={"task_id": "resource_acquisition", "seed": 42})
        resp = client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "graph_nodes" in data
        assert "metrics" in data

    def test_state_without_reset_fails(self, client):
        from app import server
        server._env = None
        resp = client.get("/state")
        assert resp.status_code == 400


class TestTasksEndpoint:
    def test_tasks_returns_list(self, client):
        resp = client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 4  # easy + medium + hard + expert
        task_ids = {t["task_id"] for t in data}
        assert task_ids == {
            "resource_acquisition",
            "coalition_building",
            "adversarial_resilience",
            "market_manipulation",
        }

    def test_tasks_include_action_schema(self, client):
        resp = client.get("/tasks")
        data = resp.json()
        for task in data:
            assert "action_schema" in task
            assert len(task["action_schema"]) == 7  # 7 action types


class TestGraderEndpoint:
    def test_grader_after_episode(self, client):
        client.post("/reset", json={"task_id": "resource_acquisition", "seed": 42})
        # Run until done
        for _ in range(25):
            resp = client.post("/step", json={"action_type": "OBSERVE"})
            if resp.json()["done"]:
                break
        resp = client.get("/grader")
        assert resp.status_code == 200
        data = resp.json()
        assert 0.0 <= data["score"] <= 1.0
        assert "breakdown" in data


class TestBaselineEndpoint:
    def test_baseline_returns_all_tasks(self, client):
        resp = client.post("/baseline")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 4
        for r in data:
            assert 0.0 <= r["score"] <= 1.0
            assert r["steps_taken"] > 0


class TestOpenEnvCompliance:
    """Verify the environment follows the OpenEnv spec contract."""

    def test_full_episode_lifecycle(self, client):
        """reset() → step()*N → state() → grader — full lifecycle."""
        # 1. Reset
        resp = client.post("/reset", json={"task_id": "resource_acquisition", "seed": 42})
        assert resp.status_code == 200
        obs = resp.json()
        assert obs["step_number"] == 0

        # 2. Step until done
        done = False
        steps = 0
        while not done and steps < 25:
            resp = client.post("/step", json={
                "action_type": "PROPOSE_TRADE",
                "target_agent": "provider_alpha",
                "offer_resources": {"compute": 1},
                "request_resources": {"storage": 1},
            })
            assert resp.status_code == 200
            result = resp.json()
            done = result["done"]
            steps += 1

        # 3. State
        resp = client.get("/state")
        assert resp.status_code == 200
        state = resp.json()
        assert state["episode_complete"] is True

        # 4. Grader
        resp = client.get("/grader")
        assert resp.status_code == 200
        grade = resp.json()
        assert "score" in grade
        assert "breakdown" in grade
