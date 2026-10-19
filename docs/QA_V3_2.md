# QA record — A-BLAST MethodsX v3.2 FINAL

The package was syntax-checked and exercised end-to-end with the built-in synthetic `--quick` mode before delivery.

Verified in the smoke test:

- fixed-model rolling-origin construction with distinct origins;
- history-only STL at every origin;
- one model fit reused across all origins within a seed-level checkpoint;
- resume detection for completed dataset × horizon × model × seed groups;
- rolling-origin ablation with main BLS and Full A-BLAST reused;
- unit-free paired percentage Table 3;
- dataset-specific Fig. 2 and Fig. 3 without cross-unit error pooling;
- pre-specified central-origin Fig. 4;
- Google-Drive-compatible atomic CSV/JSON checkpoints;
- Tables 1–6 generation;
- PNG/PDF figure generation;
- final ZIP generation and clean process termination.

Synthetic smoke-test values are for software QA only and must never be used in the manuscript.
