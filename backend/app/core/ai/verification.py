"""Deterministic verification of AI investigation results.

The LLM (or rule engine) may claim ``RESOLVED``; this module checks that claim
against the ledger records that were actually retrieved by the investigation
tools, so the "honest uncertainty" guarantee does not rely on the prompt alone.
"""
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

TOLERANCE = Decimal("0.01")

# evidence type -> (tools_payload key, reference fields, amount field)
EVIDENCE_SOURCES: Dict[str, Dict[str, Any]] = {
    "SETTLEMENT": {"key": "settlements", "refs": ("settlement_reference",), "amount": "net_amount"},
    "FEE": {"key": "fees", "refs": ("fee_reference",), "amount": "total_amount"},
    "REFUND": {"key": "refunds", "refs": ("refund_reference",), "amount": "amount"},
    "BANK": {"key": "bank_txns", "refs": ("bank_reference",), "amount": "amount"},
    "BANK_TXN": {"key": "bank_txns", "refs": ("bank_reference",), "amount": "amount"},
    "INVOICE": {"key": "invoices", "refs": ("invoice_number",), "amount": "total_amount"},
    "PAYMENT": {"key": "payment", "refs": ("transaction_id", "payment_reference", "id"), "amount": "amount"},
}


def _to_decimal(value: Any) -> Optional[Decimal]:
    if value is None or isinstance(value, bool):
        return None
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return d if d.is_finite() else None


def _records_for(tools_payload: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    data = tools_payload.get(key)
    if not data:
        return []
    if isinstance(data, dict):
        return [data]
    return [r for r in data if isinstance(r, dict)]


def verify_investigation(
    result: Dict[str, Any],
    tools_payload: Dict[str, Any],
    exception_difference: Any,
) -> Dict[str, Any]:
    """Verify a RESOLVED claim. Returns a dict with:

    passed, failed_checks (list of str), verified_amount (float: sum of cited
    records that exist and whose amount matches), checks_run.
    Non-RESOLVED results are not verified (passed=True, skipped=True).
    """
    if result.get("status") != "RESOLVED":
        return {"passed": True, "skipped": True, "failed_checks": [], "verified_amount": None}

    failed: List[str] = []
    evidence = result.get("evidence_records")
    if not isinstance(evidence, list) or not evidence:
        return {
            "passed": False,
            "skipped": False,
            "failed_checks": ["EMPTY_EVIDENCE: RESOLVED requires at least one cited evidence record"],
            "verified_amount": 0.0,
        }

    verified_total = Decimal("0")
    cited_total = Decimal("0")
    cited_total_valid = True
    seen = set()

    for i, ev in enumerate(evidence):
        if not isinstance(ev, dict):
            failed.append(f"MALFORMED_EVIDENCE: evidence #{i + 1} is not an object")
            cited_total_valid = False
            continue
        etype = str(ev.get("type", "")).upper()
        ref = ev.get("reference")
        label = f"{etype or '?'} {ref!r}"
        spec = EVIDENCE_SOURCES.get(etype)
        if spec is None or ref in (None, ""):
            failed.append(f"UNKNOWN_EVIDENCE: {label} has unsupported type or missing reference")
            cited_total_valid = False
            continue

        match = next(
            (r for r in _records_for(tools_payload, spec["key"])
             if any(r.get(f) is not None and str(r.get(f)) == str(ref) for f in spec["refs"])),
            None,
        )
        if match is None:
            failed.append(f"EVIDENCE_NOT_FOUND: {label} not present in retrieved {spec['key']}")
            cited_total_valid = False
            continue

        if (etype, str(ref)) in seen:
            failed.append(f"DUPLICATE_EVIDENCE: {label} cited more than once")
            cited_total_valid = False
            continue
        seen.add((etype, str(ref)))

        cited = _to_decimal(ev.get("amount"))
        actual = _to_decimal(match.get(spec["amount"]))
        if cited is None or actual is None:
            failed.append(f"AMOUNT_INVALID: {label} has a missing or non-numeric amount")
            cited_total_valid = False
            continue
        cited_total += cited
        if abs(cited - actual) >= TOLERANCE:
            failed.append(f"AMOUNT_MISMATCH: {label} cited {cited} but actual record amount is {actual}")
            continue
        verified_total += actual

    explained = _to_decimal(result.get("explained_amount"))
    diff = _to_decimal(exception_difference)
    if explained is None:
        failed.append("EXPLAINED_AMOUNT_INVALID: explained_amount missing or non-numeric")
    elif cited_total_valid and abs(cited_total - explained) >= TOLERANCE:
        failed.append(
            f"EVIDENCE_SUM_MISMATCH: cited evidence sums to {cited_total} but explained_amount is {explained}"
        )
    if explained is not None and diff is not None and abs(explained - abs(diff)) >= TOLERANCE:
        failed.append(
            f"DIFFERENCE_NOT_EXPLAINED: explained_amount {explained} != exception difference {abs(diff)}"
        )

    return {
        "passed": not failed,
        "skipped": False,
        "failed_checks": failed,
        "verified_amount": float(verified_total),
    }


def apply_verification(result: Dict[str, Any], verification: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``result`` downgraded to UNRESOLVED if verification failed."""
    if verification.get("passed"):
        return result
    out = dict(result)
    claimed = result.get("status")
    out["status"] = "UNRESOLVED"
    out["root_cause"] = f"UNVERIFIED_{str(result.get('root_cause') or 'CLAIM')[:80]}"
    out["explained_amount"] = verification.get("verified_amount") or 0.0
    reasons = "; ".join(verification["failed_checks"])
    out["reasoning_summary"] = (
        f"Downgraded from {claimed} to UNRESOLVED: automated evidence verification failed ({reasons}). "
        f"Original claim: {result.get('reasoning_summary', 'n/a')}"
    )
    out["recommended_action"] = "Manual review required: AI-cited evidence could not be verified against the ledger."
    return out
