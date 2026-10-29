Suggested title: Add opt-in Dataset v2 confidence for selected scientific targets

A node with direct ESTIMATE(42) and tree/effective FORCED(35) currently exports
42 as exact with direct preference; four exact-family policies can also change
its certified terminal child's target from 35 to 42. Dataset v2 associates
confidence with the selected Value, keeps node exactness separately, protects
certified same-state targets, and marks heuristic family results ESTIMATE.

The five supported policies are none and the four prediction policies. All
four pv_exact_then_* policies fail closed pending a separate mathematical proof
contract. Different states' different certified values remain valid. V1 output
and interpretation are unchanged. V2 artifacts have explicit namespaces,
confidence/producer/generation identities and validated serialization. The
bounded path reuses the existing reader and SQLite TEMP family passes.

Validation: 1,872 installed-wheel CI tests pass, two CUDA skips; zero new static
diagnostics; 1,890 exact frozen-v1 comparisons; 1,050 exact v2
flat/reference/indexed JSONL triples; 840 identical policy gates. The real PR
#82 endgame and a legal 35/71 unequal-state-family witness are retained.

Dependency hold: this description is prepared locally, not published as a PR.
The implementation delta eea417271a822311e08dd8b88ee938ccb32ef2f4 requires the
reviewed unified reader/scientific-format/PR82 base described in
../morpion_dataset_target_confidence_v2.md. Its current reproducibility base
77267b64 is an explicit local merge of sibling stacks. Publishing the whole
branch against any existing source PR would duplicate unrelated changes.
Production activation and merge remain outside this change.
