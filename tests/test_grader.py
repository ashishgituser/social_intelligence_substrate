"""Tests for the deterministic grader."""

import pytest
from app.grader import grade_episode


class TestGradeEasy:
    def test_perfect_score(self):
        metrics = {
            "task_completion": 1.0,
            "efficiency": 1.0,
            "social_capital": 1.0,
            "robustness": 1.0,
            "alliances_formed": 0,
        }
        result = grade_episode("resource_acquisition", metrics, True)
        assert result.score == 1.0
        assert result.passed

    def test_zero_score(self):
        metrics = {
            "task_completion": 0.0,
            "efficiency": 0.0,
            "social_capital": 0.0,
        }
        result = grade_episode("resource_acquisition", metrics, False)
        assert result.score == 0.0
        assert not result.passed

    def test_partial_score(self):
        metrics = {
            "task_completion": 0.5,
            "efficiency": 0.5,
            "social_capital": 0.5,
        }
        result = grade_episode("resource_acquisition", metrics, False)
        assert 0.0 < result.score < 1.0
        # 0.5*0.6 + 0.5*0.25 + 0.5*0.15 = 0.3 + 0.125 + 0.075 = 0.5
        assert result.score == pytest.approx(0.5, rel=0.01)

    def test_breakdown_keys(self):
        metrics = {"task_completion": 0.8, "efficiency": 0.7, "social_capital": 0.6}
        result = grade_episode("resource_acquisition", metrics, True)
        assert "task_completion" in result.breakdown
        assert "efficiency" in result.breakdown
        assert "social_capital" in result.breakdown


class TestGradeMedium:
    def test_alliance_quality(self):
        metrics = {
            "task_completion": 1.0,
            "efficiency": 0.5,
            "social_capital": 0.5,
            "alliances_formed": 2,
        }
        result = grade_episode("coalition_building", metrics, True)
        assert result.breakdown["alliance_quality"] == pytest.approx(0.25, rel=0.01)

    def test_alliance_caps_at_two(self):
        metrics = {
            "task_completion": 1.0,
            "efficiency": 1.0,
            "social_capital": 1.0,
            "alliances_formed": 10,
        }
        result = grade_episode("coalition_building", metrics, True)
        # alliance_quality = min(1.0, 10/2) * 0.25 = 0.25
        assert result.breakdown["alliance_quality"] == pytest.approx(0.25, rel=0.01)


class TestGradeHard:
    def test_robustness_weight(self):
        # Perfect robustness should contribute 30%
        metrics = {
            "task_completion": 0.0,
            "efficiency": 0.0,
            "social_capital": 0.0,
            "robustness": 1.0,
        }
        result = grade_episode("adversarial_resilience", metrics, False)
        assert result.breakdown["robustness"] == pytest.approx(0.3, rel=0.01)


class TestGradeExpert:
    def test_expert_grader_exists(self):
        metrics = {
            "task_completion": 0.5,
            "efficiency": 0.5,
            "social_capital": 0.5,
            "robustness": 0.5,
            "alliances_formed": 1,
        }
        result = grade_episode("market_manipulation", metrics, False)
        assert result.task_id == "market_manipulation"
        assert 0.0 <= result.score <= 1.0
        assert "robustness" in result.breakdown
        assert "alliance_quality" in result.breakdown

    def test_expert_perfect(self):
        metrics = {
            "task_completion": 1.0,
            "efficiency": 1.0,
            "social_capital": 1.0,
            "robustness": 1.0,
            "alliances_formed": 2,
        }
        result = grade_episode("market_manipulation", metrics, True)
        assert result.score == 1.0


class TestGraderDeterminism:
    def test_same_input_same_output(self):
        metrics = {
            "task_completion": 0.75,
            "efficiency": 0.6,
            "social_capital": 0.8,
            "robustness": 0.9,
            "alliances_formed": 1,
        }
        results = [
            grade_episode("adversarial_resilience", metrics, False)
            for _ in range(5)
        ]
        scores = [r.score for r in results]
        assert len(set(scores)) == 1  # all identical


class TestUnknownTask:
    def test_unknown_task_returns_zero(self):
        result = grade_episode("does_not_exist", {}, False)
        assert result.score == 0.0
        assert not result.passed
