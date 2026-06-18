# Evaluation Contracts

QuantCortex separates a strategy's decision stream from the engine that turns
those decisions into returns. The machine-readable contract for the reference
study is generated at `paper/results/evaluation_contract.json` and validated
against `schemas/evaluation_contract.schema.json`.

## Canonical Target Tape

The engine-neutral target tape has one row per decision, symbol, and target:

```text
decision_timestamp,symbol,target_weight
2024-01-01T00:00:00Z,A,1.0
2024-01-01T00:00:00Z,B,0.0
```

`quantcortex.backtest.conformance.validate_target_tape` requires unique
timestamp-symbol rows, a complete symbol set at every decision, finite
long-only weights, and gross exposure no greater than the declared limit.
Timestamps identify close-of-bar decisions; the reference engines execute them
on the first strictly later price bar. The versioned JSON envelope also declares
the complete symbol set and gross limit:

```json
{
  "schema_version": 1,
  "symbols": ["A", "B"],
  "max_gross": 1.0,
  "records": [
    {
      "decision_timestamp": "2024-01-01T00:00:00Z",
      "symbol": "A",
      "target_weight": 1.0
    }
  ]
}
```

`target_tape_to_payload` and `target_tape_from_payload` implement that envelope.
`schemas/canonical_target_tape.schema.json` specifies its serialized structure
and primitive constraints for other engines. Runtime validation additionally
enforces cross-record portfolio invariants: no duplicate timestamp-symbol rows,
the declared symbol universe at every decision, and the per-decision gross
limit. The paper experiment round-trips each variant through this boundary
before backtesting and publishes canonical payload hashes in
`paper/results/target_tape_hashes.json`.

The evaluation contract also records post-overlay exposure rules and the
paper-trading order-state policy: persist intent before submission, block
automatic retry after an uncertain outcome, and reject stale state revisions.
Those operational claims are supported by fault-injection tests rather than by
the historical return panel.

The fixtures under `tests/fixtures/conformance/` are deterministic and
synthetic by design. They test software semantics only and are never used for
performance claims.

## Comparator Contracts

The paper reports two distinct controls:

- The realized-exposure attribution control is an exact ex-post arithmetic
  identity. It is gross of comparator costs and is not presented as tradable.
- The target-exposure comparator is causal and event-driven. It follows the
  strategy's declared target gross exposure, uses the same next-bar timing and
  cash proxy, and pays the same proportional cost rate.

Do not merge these concepts. The first explains realized return; the second is
an implementable economic comparison under the stated model.

## Evidence Classes

Claims must identify their evidence type: exact identity, property test,
economic counterfactual, fault injection, SDK conformance, or untested live
behavior. A passing unit test cannot establish broker connectivity, data
rights, market capacity, or future performance. See the paper appendix for the
claim-to-evidence table and `docs/production-readiness.md` for external gates.
