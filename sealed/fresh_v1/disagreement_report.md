# fresh_v1 — disagreement report

> AI-generated preliminary benchmark with limited human review. Not clinician-validated.
> USER-ONLY FILE: contains question text. The developer must not open it.

For each item record ONE decision in `adjudication.csv` as `id,decision` with decision in {generator, annotator, ambiguous, drop}.

### F065

> What's the weather going to be like tomorrow?

| Field | Generator | Annotator |
|---|---|---|
| category | ambiguous_underspecified_offtopic | ambiguous_underspecified_offtopic |
| routing | needs_clarification | unsupported_operation |
| intended_type | none | none |
| answerability | not_answerable_as_posed | not_answerable_as_posed |
| required_fields | ['other'] | [] |
| parameters | {} | {} |
| rationale | Off-topic question unrelated to the visit history. | Off-topic weather question. |

### F080

> Can you recommend a good place to park near the eye clinic?

| Field | Generator | Annotator |
|---|---|---|
| category | ambiguous_underspecified_offtopic | ambiguous_underspecified_offtopic |
| routing | needs_clarification | unsupported_operation |
| intended_type | none | none |
| answerability | not_answerable_as_posed | not_answerable_as_posed |
| required_fields | ['other'] | [] |
| parameters | {} | {} |
| rationale | Off-topic request about parking. | Off-topic parking question. |
