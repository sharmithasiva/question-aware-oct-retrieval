# fresh_v1 — review sample (agreed items)

> AI-generated preliminary benchmark with limited human review. Not clinician-validated.
> USER-ONLY FILE: contains question text. The developer must not open it.

For each item record `id,decision` in `adjudication.csv` with decision in {ok, ambiguous, wrong, drop}. If more than 2 of these are `wrong`, labelling is redone and the set is not opened.

### F002

> When did BCVA first reach at least 5 letters above baseline?

| Field | Generator | Annotator |
|---|---|---|
| category | supported | supported |
| routing | retrieve | retrieve |
| intended_type | first_vision_gain_1line | first_vision_gain_1line |
| answerability | metadata_answerable | metadata_answerable |
| required_fields | ['bcva', 'visit_order'] | ['bcva', 'visit_order'] |
| parameters | {'target': 'bcva', 'direction': 'increase', 'threshold': 5, 'comparator': 'baseline'} | {'target': 'bcva', 'direction': 'increase', 'threshold': 5, 'comparator': '>='} |
| rationale | Asks for the first post-baseline visit with BCVA at least 5 letters above baseline. | First post-baseline visit with BCVA at least 5 letters above baseline. |

### F089

> biggest VA dip between consecutive visits?

| Field | Generator | Annotator |
|---|---|---|
| category | supported | supported |
| routing | retrieve | retrieve |
| intended_type | largest_bcva_drop | largest_bcva_drop |
| answerability | metadata_answerable | metadata_answerable |
| required_fields | ['bcva', 'visit_order'] | ['bcva', 'visit_order'] |
| parameters | {'target': 'bcva', 'direction': 'decrease', 'comparator': 'previous_visit'} | {'target': 'bcva', 'direction': 'decrease'} |
| rationale | Shorthand for the largest VA drop between consecutive visits. | Biggest VA dip between consecutive visits. |

### F082

> Greatest single-step rise in central thickness: which visit?

| Field | Generator | Annotator |
|---|---|---|
| category | supported | supported |
| routing | retrieve | retrieve |
| intended_type | largest_cst_rise | largest_cst_rise |
| answerability | metadata_answerable | metadata_answerable |
| required_fields | ['cst', 'visit_order'] | ['cst', 'visit_order'] |
| parameters | {'target': 'cst', 'direction': 'increase', 'comparator': 'previous_visit'} | {'target': 'cst', 'direction': 'increase'} |
| rationale | Asks for the greatest single-step CST rise. | Greatest single-step rise in central thickness. |

### F059

> Pull the latest earlier visit with a higher CST than this one.

| Field | Generator | Annotator |
|---|---|---|
| category | supported | supported |
| routing | retrieve | retrieve |
| intended_type | last_thicker_than_now | last_thicker_than_now |
| answerability | metadata_answerable | metadata_answerable |
| required_fields | ['cst', 'visit_order'] | ['cst', 'visit_order'] |
| parameters | {'target': 'cst', 'comparator': 'greater_than_current'} | {'target': 'cst', 'comparator': '>'} |
| rationale | Asks for the latest earlier visit with higher CST. | Latest earlier visit with higher CST. |

### F056

> last time VA was this good or better

| Field | Generator | Annotator |
|---|---|---|
| category | supported | supported |
| routing | retrieve | retrieve |
| intended_type | last_vision_as_good | last_vision_as_good |
| answerability | metadata_answerable | metadata_answerable |
| required_fields | ['bcva', 'visit_order'] | ['bcva', 'visit_order'] |
| parameters | {'target': 'bcva', 'comparator': 'greater_or_equal_current'} | {'target': 'bcva', 'comparator': '>='} |
| rationale | Shorthand for the last visit with VA at least as good as now. | Last time VA was this good or better. |

### F097

> Retrieve the prior visit with the largest leakage index value.

| Field | Generator | Annotator |
|---|---|---|
| category | supported | supported |
| routing | data_unavailable | data_unavailable |
| intended_type | max_leakage | max_leakage |
| answerability | needs_unavailable_data | needs_unavailable_data |
| required_fields | ['leakage_index', 'visit_order'] | ['leakage_index', 'visit_order'] |
| parameters | {'target': 'leakage_index', 'direction': 'max'} | {'target': 'leakage_index'} |
| rationale | The leakage index is not available to the retrieval system. | Largest leakage index value, hidden from the system. |

### F011

> Show me the visit from one year ago.

| Field | Generator | Annotator |
|---|---|---|
| category | supported | supported |
| routing | retrieve | retrieve |
| intended_type | one_year_ago | one_year_ago |
| answerability | metadata_answerable | metadata_answerable |
| required_fields | ['week', 'visit_order'] | ['week', 'visit_order'] |
| parameters | {'time_offset_weeks': 52} | {'time_offset_weeks': 52} |
| rationale | Asks for the visit closest to 52 weeks before the current visit. | Visit about 52 weeks before the current one. |

### F075

> Find the visit approximately twenty-six weeks before this one.

| Field | Generator | Annotator |
|---|---|---|
| category | supported | supported |
| routing | retrieve | retrieve |
| intended_type | six_months_ago | six_months_ago |
| answerability | metadata_answerable | metadata_answerable |
| required_fields | ['week', 'visit_order'] | ['week', 'visit_order'] |
| parameters | {'time_offset_weeks': 26} | {'time_offset_weeks': 26} |
| rationale | Asks for the visit about twenty-six weeks earlier. | Visit about 26 weeks earlier. |

### F017

> When was it worse?

| Field | Generator | Annotator |
|---|---|---|
| category | ambiguous_underspecified_offtopic | ambiguous_underspecified_offtopic |
| routing | needs_clarification | needs_clarification |
| intended_type | none | none |
| answerability | not_answerable_as_posed | not_answerable_as_posed |
| required_fields | ['cst', 'bcva'] | [] |
| parameters | {} | {} |
| rationale | Does not specify which measure or which reference point defines worse. | 'Worse' does not say which measure (CST or BCVA) or by how much. |

### F014

> How many anti-VEGF injections had this eye received before today?

| Field | Generator | Annotator |
|---|---|---|
| category | images_or_unavailable | images_or_unavailable |
| routing | data_unavailable | data_unavailable |
| intended_type | none | none |
| answerability | needs_unavailable_data | needs_unavailable_data |
| required_fields | ['injection'] | ['injection'] |
| parameters | {} | {} |
| rationale | Injection records are not visible to the retrieval system. | Injection records are not visible to the retrieval system. |

### F060

> Compare this eye's CST with the fellow eye at the same visit.

| Field | Generator | Annotator |
|---|---|---|
| category | plausible_unsupported | plausible_unsupported |
| routing | unsupported_operation | unsupported_operation |
| intended_type | none | none |
| answerability | metadata_answerable | metadata_answerable |
| required_fields | ['cst', 'other'] | ['cst', 'visit_order'] |
| parameters | {'target': 'cst'} | {'target': 'cst'} |
| rationale | Requires comparison with the fellow eye, outside single-eye retrieval. | Requires the fellow eye, outside single-eye history retrieval. |

### F118

> What were the CST and BCVA at the study visit one year before this one?

| Field | Generator | Annotator |
|---|---|---|
| category | supported | supported |
| routing | retrieve | retrieve |
| intended_type | one_year_ago | one_year_ago |
| answerability | metadata_answerable | metadata_answerable |
| required_fields | ['week', 'visit_order', 'cst', 'bcva'] | ['cst', 'bcva', 'week', 'visit_order'] |
| parameters | {'time_offset_weeks': 52} | {'time_offset_weeks': 52} |
| rationale | Asks for CST and BCVA at the visit one year earlier. | CST and BCVA at the visit one year earlier. |
