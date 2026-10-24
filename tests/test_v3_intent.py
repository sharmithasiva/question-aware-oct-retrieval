"""Phase 1.3 parser tests: dev paraphrases, v1 regression, generic phrasings, unknown/unsupported routing.

Uses ONLY dev paraphrases and developer-written generic sentences. The locked file is never read here.
"""
import pytest

from oct_retrieval import intent as I
from oct_retrieval.evaluate_v3_dev import GOLD_V2, gold_from_v1, intent_matches, route_matches
from oct_retrieval.paraphrases import dev_by_type
from oct_retrieval.questions import HELDOUT_PARAPHRASES, QUESTION_TYPES

DEV = [(q, t) for q, texts in dev_by_type().items() for t in texts]


@pytest.mark.parametrize("qtype,text", DEV)
def test_dev_paraphrases_parse_to_gold(qtype, text):
    q = I.parse_intent(text)
    assert route_matches(q, qtype)
    if qtype != "unsupported":
        assert intent_matches(q, GOLD_V2[qtype]), q


@pytest.mark.parametrize("qtype,text", [(q, t) for q, (_, ts) in QUESTION_TYPES.items() for t in ts])
def test_v1_templates_still_parse(qtype, text):
    assert intent_matches(I.parse_intent(text), gold_from_v1(QUESTION_TYPES[qtype][0]))


@pytest.mark.parametrize("text,gold", HELDOUT_PARAPHRASES)
def test_v1_heldout_still_parse(text, gold):
    assert intent_matches(I.parse_intent(text), gold_from_v1(gold))


# --- generic phrasings written for the operations, not copied from any benchmark set ---

@pytest.mark.parametrize("text,expected", [
    ("Which visit had the largest consecutive decrease in CST?",
     dict(operation=I.CONSECUTIVE_CHANGE, target="cst", direction="decrease")),
    ("When was the biggest interval gain in visual acuity?",
     dict(operation=I.CONSECUTIVE_CHANGE, target="bcva", direction="increase")),
    ("First visit where BCVA had dropped at least 2 lines from baseline?",
     dict(operation=I.THRESHOLD_FROM_BASELINE, target="bcva", direction="decrease", threshold=10.0)),
    ("Earliest visit at which CST was 50 um lower than baseline",
     dict(operation=I.THRESHOLD_FROM_BASELINE, target="cst", direction="decrease", threshold=50.0)),
    ("Most recent visit where CST was lower than today?",
     dict(operation=I.COMPARE_TO_CURRENT, target="cst", comparator="lt", recency="last")),
    ("Last visit where vision was better than now",
     dict(operation=I.COMPARE_TO_CURRENT, target="bcva", comparator="gt", recency="last")),
    ("Show the visit from 3 months ago", dict(operation=I.TIME_OFFSET, offset_weeks=13)),
    ("Compare with two years earlier", dict(operation=I.TIME_OFFSET, offset_weeks=104)),
    ("Which visit had the thinnest retina?", dict(operation=I.EXTREMUM, target="cst", direction="min")),
])
def test_generic_operations(text, expected):
    q = I.parse_intent(text)
    assert q.task == I.RETRIEVE
    assert intent_matches(q, expected), q


@pytest.mark.parametrize("text", [
    "Tell me about this patient.",
    "What happened?",
    "Which visit had the biggest change in CST?",        # change without a reference -> must not guess
    "When did vision first improve compared with baseline?",   # no threshold -> must not guess
    "Compare CST and BCVA at the worst visit.",           # two targets
])
def test_unrecognised_or_underspecified_ask_for_clarification(text):
    q = I.parse_intent(text)
    assert q.task == I.CLARIFY and q.operation is None and q.reason


@pytest.mark.parametrize("text", [
    "When did subretinal fluid appear?",
    "Which visit shows disruption of the ellipsoid zone?",
    "When were hyperreflective foci most numerous?",
    "Has DRIL progressed?",
])
def test_image_questions_are_unsupported(text):
    assert I.parse_intent(text).task == I.UNSUPPORTED_IMAGE


@pytest.mark.parametrize("text,var", [
    ("Which visit had the most leakage?", "leakage_index"),
    ("When was the eye last injected?", "injection"),
    ("When was the DRSS level highest?", "drss_level"),
])
def test_out_of_memory_variables_are_unavailable_by_default(text, var):
    q = I.parse_intent(text)
    assert q.task == I.UNAVAILABLE and q.missing_variable == var


def test_declared_variable_becomes_a_lookup():
    q = I.parse_intent("Which visit had the most leakage?", available_variables=("cst", "bcva", "leakage_index"))
    assert q.task == I.RETRIEVE and q.operation == I.EXTREMUM and q.target == "leakage_index" and q.direction == "max"
    q2 = I.parse_intent("When was the eye last injected?", available_variables=("cst", "bcva", "injection"))
    assert q2.task == I.RETRIEVE and q2.operation == I.LAST_EVENT


def test_at_least_threshold_is_not_a_minimum():
    q = I.parse_intent("First visit with BCVA at least 5 letters above baseline")
    assert q.operation == I.THRESHOLD_FROM_BASELINE and q.threshold == 5.0


def test_best_corrected_is_not_best():
    q = I.parse_intent("Biggest visit-to-visit drop in best-corrected visual acuity")
    assert q.operation == I.CONSECUTIVE_CHANGE and q.direction == "decrease"
