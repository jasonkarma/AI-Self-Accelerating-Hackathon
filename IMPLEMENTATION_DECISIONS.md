# Implementation decisions

1. SPEC v1.2 wins over v1.1: model features are exactly 299-dimensional; six risk components remain separate.
2. `tau_dev = 1 - tau_risk`; qualification is `y_dev >= tau_dev`, equivalent to `Risk <= tau_risk`.
3. The bundled fixture is always marked synthetic. Real-data ingestion adds `source_is_real` before freezing.
4. AUBQC preserves the preregistered `1/30` scaling even though seven five-query checkpoints are reported.
5. Contract phase strings are mapped at the API boundary to the shorter labels in `api_contract.json`.
6. M5 policy mode is persisted in benchmark output and ledger metadata (`adaptive_tool` or `live_agent`).
