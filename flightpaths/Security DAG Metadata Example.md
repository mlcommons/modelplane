# Security DAG Metadata Example

This example shows one way to pass metadata through an end-to-end jailbreak
or security DAG. It is intended as a documentation pattern for flightpaths,
not a new scoring rule or required schema.

The core idea is:

1. Input data may already contain security metadata such as hazard category,
   jailbreak type, or a bijection mapping reference.
2. An upstream node can normalize that metadata into `EvalContext.metadata`.
3. A downstream gate or evaluator can consume the normalized metadata and emit
   an audit-safe route record.
4. Final benchmark scoring remains separate from the route metadata.

## Example Input Row

An input CSV for a jailbreak benchmark might include columns like these:

```csv
prompt_uid,prompt,response,hazard,jailbreak_type,bijection_mapping_ref
ex-001,"...", "...", "cyber", "encoding_attack", "bijection_map:ex-001"
```

When creating the `EvalContext`, the row metadata can be stored under
`ctx.metadata`:

```python
ctx = EvalContext(
    prompt=row["prompt"],
    response=row["response"],
    metadata={
        "sample_id": row["prompt_uid"],
        "hazard": row["hazard"],
        "jailbreak_type": row["jailbreak_type"],
        "bijection_mapping_ref": row["bijection_mapping_ref"],
    },
)
```

## Upstream Metadata Node

An upstream enricher can normalize benchmark-provided metadata and add evidence
references for downstream nodes.

```python
class SecurityMetadataEnricher(Enricher):
    def run(self, ctx: EvalContext) -> NodeOutput:
        metadata = dict(ctx.metadata)
        metadata["security_evidence"] = {
            "sample_id": metadata.get("sample_id"),
            "hazard": metadata.get("hazard", "unknown"),
            "jailbreak_type": metadata.get("jailbreak_type", "unknown"),
            "bijection_mapping_ref": metadata.get("bijection_mapping_ref"),
            "evidence_refs": [
                {
                    "source_id": f"input_csv:{metadata.get('sample_id')}",
                    "kind": "benchmark_metadata",
                    "trust_tier": "benchmark_provided",
                    "redaction_status": "safe_for_public_log",
                }
            ],
        }
        return self.build_output(
            value=metadata["security_evidence"],
            ctx=ctx,
            updated_ctx=ctx.with_metadata(metadata),
        )
```

## Downstream Verifier Gate

A downstream gate can consume the normalized metadata and emit a route decision.
The route decision is diagnostic metadata; it does not need to replace the final
benchmark verdict.

```python
class JailbreakVerifierGate(Gate):
    def run(self, ctx: EvalContext) -> NodeOutput:
        evidence = ctx.metadata.get("security_evidence", {})
        hazard = evidence.get("hazard", "unknown")
        jailbreak_type = evidence.get("jailbreak_type", "unknown")

        hard_blockers = []
        missing_evidence = []

        if hazard == "unknown":
            missing_evidence.append("hazard")

        if jailbreak_type in {"encoding_attack", "prompt_injection"}:
            route = "refuse"
            hard_blockers.append("jailbreak_pattern_detected")
        elif missing_evidence:
            route = "defer"
        else:
            route = "accept"

        gate_result = {
            "route": route,
            "hard_blockers": hard_blockers,
            "missing_evidence": missing_evidence,
            "audit_status": "safe_for_public_log",
        }

        metadata = dict(ctx.metadata)
        metadata["gate_result"] = gate_result

        # The boolean value controls DAG routing. The structured gate_result
        # remains available to downstream nodes through updated metadata.
        should_continue = route == "accept"
        return self.build_output(
            value=should_continue,
            ctx=ctx,
            updated_ctx=ctx.with_metadata(metadata),
        )
```

## Audit-Safe Record

The downstream node output can be serialized into an audit-safe record for
debugging false accepts, false refusals, deferrals, and missing metadata.

```json
{
  "sample_id": "ex-001",
  "hazard": "cyber",
  "jailbreak_type": "encoding_attack",
  "bijection_mapping_ref": "bijection_map:ex-001",
  "evidence_refs": [
    {
      "source_id": "input_csv:ex-001",
      "kind": "benchmark_metadata",
      "trust_tier": "benchmark_provided",
      "redaction_status": "safe_for_public_log"
    }
  ],
  "gate_result": {
    "route": "refuse",
    "hard_blockers": ["jailbreak_pattern_detected"],
    "missing_evidence": [],
    "audit_status": "safe_for_public_log"
  }
}
```

## Route Taxonomy

The route field is useful when analyzing runtime behavior before final scoring:

- `accept`: continue to the next evaluator or final scorer.
- `revise`: candidate output appears repairable before evaluation.
- `ask`: missing user or task context should be requested.
- `defer`: stronger evaluator or human review is needed.
- `refuse`: hard blocker prevents direct acceptance.

For benchmark reporting, keep these route decisions separate from the final
verdict. This lets a security DAG distinguish:

- unsafe content that reached final scoring,
- unsafe content caught by a verifier gate,
- benign content incorrectly refused,
- samples deferred because hazard or mapping metadata was missing,
- cases where a correction step repaired the candidate before scoring.
