import numpy as np
import pytest

from oct_retrieval.episodes import build_episodes
from oct_retrieval.questions import (HELDOUT_PARAPHRASES, QUESTION_TYPES, QuestionSpec, generate_instances,
                                     parse_question, relevant_indices)

NAN = float("nan")


@pytest.mark.parametrize("qtype", list(QUESTION_TYPES))
def test_templates_parse_to_gold(qtype):
    gold, templates = QUESTION_TYPES[qtype]
    for q in templates:
        assert parse_question(q) == gold, q


@pytest.mark.parametrize("question,gold", HELDOUT_PARAPHRASES)
def test_heldout_paraphrases(question, gold):
    assert parse_question(question) == gold


def test_brief_example_question():
    spec = parse_question("Compared with when the edema was worst, has it improved?")
    assert spec == QuestionSpec("cst", "extreme", "max", "metadata")


def test_image_question_is_not_metadata():
    assert parse_question("How has the retinal structure changed?").evidence == "image"


def test_relevant_temporal():
    v = np.array([300.0, 250.0, 280.0])
    assert relevant_indices(QuestionSpec("cst", "baseline", None, "metadata"), v, 260) == [0]
    assert relevant_indices(QuestionSpec("cst", "previous", None, "metadata"), v, 260) == [2]


def test_relevant_extreme_ties_and_nan():
    v = np.array([300.0, NAN, 300.0, 200.0])
    assert relevant_indices(QuestionSpec("cst", "extreme", "max", "metadata"), v, 0) == [0, 2]
    assert relevant_indices(QuestionSpec("cst", "extreme", "min", "metadata"), v, 0) == [3]


def test_relevant_similar():
    v = np.array([300.0, 250.0, 280.0])
    assert relevant_indices(QuestionSpec("cst", "similar", None, "metadata"), v, 270.0) == [2]
    assert relevant_indices(QuestionSpec("cst", "similar", None, "metadata"), v, NAN) == []


def test_relevant_undefined_when_all_missing():
    assert relevant_indices(QuestionSpec("bcva", "extreme", "min", "metadata"), np.array([NAN, NAN]), 1) == []


def test_instances_exclude_current_and_need_two_priors(raw_clinical):
    ep = build_episodes(raw_clinical)
    inst = generate_instances(ep)
    by_id = ep.set_index("visit_id")
    for r in inst.itertuples():
        assert r.n_history >= 2
        cur = by_id.loc[r.current_visit_id]
        for vid in r.relevant_visit_ids.split(";"):
            prior = by_id.loc[vid]
            assert vid != r.current_visit_id
            assert prior.eye_id == cur.eye_id and prior.visit_index < cur.visit_index
    # eye 2 has only 2 visits -> no instance with >=2 priors
    assert not (inst.eye_id == 2).any()


def test_worst_edema_ground_truth(raw_clinical):
    ep = build_episodes(raw_clinical)
    inst = generate_instances(ep).set_index("instance_id")
    # eye 1 CST by visit order: V1=500, V2=400, V3=350, V10=300 -> worst edema before V10 is V1
    assert inst.loc["0201GOD_V10__worst_edema", "relevant_visit_ids"] == "0201GOD_V1"
    assert inst.loc["0201GOD_V10__best_anatomy", "relevant_visit_ids"] == "0201GOD_V3"


def test_missing_values_never_ground_truth(raw_clinical):
    ep = build_episodes(raw_clinical)
    inst = generate_instances(ep)
    value_q = inst[~inst.qtype.str.startswith(("baseline", "previous"))]
    assert not value_q.relevant_visit_ids.str.contains("01-001_W4").any()
