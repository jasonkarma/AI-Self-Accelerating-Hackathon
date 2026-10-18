# Implementation decisions

1. SPEC v1.2 wins over v1.1: model features are exactly 299-dimensional; six risk components remain separate.
2. `tau_dev = 1 - tau_risk`; qualification is `y_dev >= tau_dev`, equivalent to `Risk <= tau_risk`.
3. The bundled fixture is always marked synthetic. Real-data ingestion adds `source_is_real` before freezing.
4. AUBQC preserves the preregistered `1/30` scaling even though seven five-query checkpoints are reported.
5. Contract phase strings are mapped at the API boundary to the shorter labels in `api_contract.json`.
6. M5 policy mode is persisted in benchmark output and ledger metadata (`adaptive_tool` or `live_agent`).
7. A truncated-memo crash is fixed by raising the **ceiling**, not by lowering deliberation: reasoning tokens are billed against `max_tokens`, so `finish_reason="length"` on a 140-word prompt means the model never finished thinking. Affinity moved 4096 -> 8192 (the gateway was probed and accepts 16384/32768 too), and every seat now honors `<SLUG>_MAX_TOKENS` because the usable cap is a gateway property, not a model property.
8. Fallback seats inherit the primary seat's `max_tokens`/`timeout` through `AgentModel.rehomed()`; rebuilding them bare silently reset a re-homed run to the 1024 dataclass default, which is exactly the class of bug that crashed Affinity.
