"""
Unit tests for the deterministic logic in main.py.

Philosophy: the LLM calls are non-deterministic and cost money, so they are
covered by the integration-style eval harness (eval.py). Here we test everything
that is deterministic and free -- the readability math, JSON repair, strategy
blending, classifier defaults -- plus the pipeline's draft-selection logic
(best-draft tracking, safety fail-closed, readability tiebreak) by MOCKING the
LLM boundary. That lets us assert the orchestration is correct without any API
calls.

    python3 -m pytest -q
"""
from unittest.mock import MagicMock

import main
import prompts


# --------------------------------------------------------------------------- #
# Pure helpers: readability
# --------------------------------------------------------------------------- #
def test_count_syllables():
    assert main._count_syllables("cat") == 1
    assert main._count_syllables("sky") == 1          # 'y' counts as a vowel
    assert main._count_syllables("beautiful") == 3
    assert main._count_syllables("") == 0
    assert main._count_syllables("the") == 1          # trailing-e rule keeps >=1


def test_readability_grade_is_low_for_simple_text():
    simple = "The cat sat. The dog ran. We had fun."
    grade = main.readability_grade(simple)
    assert isinstance(grade, float)
    assert grade < 4.0                                # short simple sentences


def test_readability_grade_higher_for_complex_text():
    complex_text = ("The extraordinary philosophical implications necessitated "
                    "considerable deliberation among the participants.")
    assert main.readability_grade(complex_text) > 8.0


def test_readability_grade_empty_is_zero():
    assert main.readability_grade("") == 0.0


# --------------------------------------------------------------------------- #
# Pure helpers: JSON repair
# --------------------------------------------------------------------------- #
def test_extract_json_fenced():
    assert main._extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_embedded_in_prose():
    assert main._extract_json('Sure! {"b": 2} hope that helps') == {"b": 2}


def test_extract_json_plain():
    assert main._extract_json('{"c": 3}') == {"c": 3}


def test_extract_json_garbage_returns_empty():
    assert main._extract_json("not json at all") == {}


# --------------------------------------------------------------------------- #
# Pure helpers: strategy blending
# --------------------------------------------------------------------------- #
def test_blended_strategy_merges_two_categories():
    blended = prompts.blended_strategy(["funny", "educational"])
    assert "funny" in blended and "educational" in blended


def test_blended_strategy_unknown_falls_back():
    # unknown categories are skipped; empty result falls back to soothing text
    assert prompts.blended_strategy(["nonexistent"]) == \
        prompts.CATEGORY_STRATEGIES["soothing"]


# --------------------------------------------------------------------------- #
# Classifier defensive defaults (mock the API layer)
# --------------------------------------------------------------------------- #
def test_classify_request_defaults_on_empty_model_response(monkeypatch):
    monkeypatch.setattr(main, "call_model_json", lambda *a, **k: {})
    c = main.classify_request("anything")
    assert c["safe"] is True
    assert c["categories"] == ["soothing"]
    assert c["category"] == "soothing"                # primary derived from list
    assert c["must_include"] == []


# --------------------------------------------------------------------------- #
# Pipeline draft-selection logic (mock the whole LLM boundary)
# --------------------------------------------------------------------------- #
def _safe_classification():
    return {"safe": True, "categories": ["soothing"], "characters": [],
            "setting": "a cozy place", "must_include": [], "moral": "kindness",
            "category": "soothing"}


def _mock_pipeline(monkeypatch, drafts, verdicts, grades):
    """Patch classify/plan/write/judge/readability so generate_story is driven
    entirely by the canned drafts/verdicts/grades we supply."""
    monkeypatch.setattr(main, "classify_request",
                        lambda *a, **k: _safe_classification())
    monkeypatch.setattr(main, "plan_story", lambda *a, **k: "outline")
    monkeypatch.setattr(main, "write_story", MagicMock(side_effect=drafts))
    monkeypatch.setattr(main, "judge_story", MagicMock(side_effect=verdicts))
    monkeypatch.setattr(main, "readability_grade",
                        MagicMock(side_effect=grades) if isinstance(grades, list)
                        else MagicMock(return_value=grades))


def test_best_draft_is_kept_when_rewrite_regresses(monkeypatch):
    # Draft 1 scores 9.0, the rewrite regresses to 8.0. We must ship draft 1.
    # Grade fixed at 7.0 so the loop never early-exits on readability.
    _mock_pipeline(
        monkeypatch,
        drafts=["DRAFT1", "DRAFT2"],
        verdicts=[{"overall": 9.0, "scores": {"safety": 10}},
                  {"overall": 8.0, "scores": {"safety": 10}}],
        grades=7.0,
    )
    story, _, verdict = main.generate_story("req", verbose=False)
    assert story == "DRAFT1"
    assert verdict["overall"] == 9.0


def test_fail_closed_when_no_draft_clears_safety_floor(monkeypatch):
    # Every draft is unsafe (safety below SAFETY_FLOOR) -> ship nothing.
    _mock_pipeline(
        monkeypatch,
        drafts=["BAD1", "BAD2"],
        verdicts=[{"overall": 9.0, "scores": {"safety": 3}},
                  {"overall": 9.0, "scores": {"safety": 2}}],
        grades=4.0,
    )
    story, _, _ = main.generate_story("req", verbose=False)
    assert story is None


def test_tiebreak_prefers_more_readable_draft(monkeypatch):
    # Both drafts score 9.0; draft 2 has the lower (better) reading grade.
    _mock_pipeline(
        monkeypatch,
        drafts=["DRAFT1", "DRAFT2"],
        verdicts=[{"overall": 9.0, "scores": {"safety": 10}},
                  {"overall": 9.0, "scores": {"safety": 10}}],
        grades=[6.6, 5.5],
    )
    story, _, _ = main.generate_story("req", verbose=False)
    assert story == "DRAFT2"


# --------------------------------------------------------------------------- #
# Parent story card (all deterministic)
# --------------------------------------------------------------------------- #
def test_reading_level_label():
    assert main.reading_level_label(0.0) == "Kindergarten"
    assert main.reading_level_label(2.4) == "Grade 2"
    assert main.reading_level_label(5.0) == "Grade 5"


def test_read_aloud_minutes_is_at_least_one():
    assert main.read_aloud_minutes("Just a few words here.") == 1
    long_text = "word " * 400
    assert main.read_aloud_minutes(long_text) == 3       # 400 / 130 ~= 3


def test_find_refrain_detects_repeated_line():
    story = ("Goodnight, moon. The cat ran fast today in the park. "
             "Goodnight, moon.")
    assert main.find_refrain(story) == "Goodnight, moon"


def test_find_refrain_returns_none_when_no_repeat():
    assert main.find_refrain("A. B. Every sentence here is different indeed.") is None


def test_format_story_card_contains_title_and_level():
    story = "**The Sleepy Cloud**\nOnce there was a little cloud. It drifted away."
    c = {"categories": ["soothing"], "moral": "rest is good"}
    card = main.format_story_card(story, c)
    assert "The Sleepy Cloud" in card and "**" not in card   # markdown stripped
    assert "Reading level:" in card
    assert "rest is good" in card


def test_unsafe_request_is_refused_before_generation(monkeypatch):
    monkeypatch.setattr(main, "classify_request",
                        lambda *a, **k: {"safe": False, "safe_reason": "violent",
                                         "categories": ["adventure"],
                                         "category": "adventure",
                                         "must_include": []})
    story, _, _ = main.generate_story("a violent story", verbose=False)
    assert story is None
