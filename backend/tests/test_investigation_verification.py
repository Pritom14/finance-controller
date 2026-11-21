import pytest
from app.core.ai.verification import verify_investigation, apply_verification

PAYLOAD = {
    "payment": {"id": "p1", "transaction_id": "TXN1", "payment_reference": "PAYREF1", "amount": 1000.0},
    "settlements": [{"settlement_reference": "SET1", "net_amount": 970.0}],
    "fees": [{"fee_reference": "FEE1", "total_amount": 20.0}, {"fee_reference": "FEE2", "total_amount": 10.0}],
    "refunds": [],
    "bank_txns": [{"bank_reference": "BANK1", "amount": 970.0}],
    "invoices": [{"invoice_number": "INV1", "total_amount": 30.0}],
}


def resolved(evidence, explained=30.0):
    return {
        "status": "RESOLVED", "root_cause": "PROCESSING_FEE", "confidence": 0.98,
        "explained_amount": explained, "evidence_records": evidence,
        "reasoning_summary": "Explained by fees.",
    }


def run(res, diff=30.0):
    v = verify_investigation(res, PAYLOAD, diff)
    return v, apply_verification(res, v)


def test_valid_resolved_passes():
    res = resolved([
        {"type": "FEE", "reference": "FEE1", "amount": 20.0},
        {"type": "FEE", "reference": "FEE2", "amount": 10.0},
    ])
    v, out = run(res)
    assert v["passed"] and not v["failed_checks"]
    assert out is res and out["status"] == "RESOLVED"


def test_fabricated_reference_downgrades():
    res = resolved([{"type": "FEE", "reference": "FEE-FAKE", "amount": 30.0}])
    v, out = run(res)
    assert not v["passed"]
    assert any(c.startswith("EVIDENCE_NOT_FOUND") for c in v["failed_checks"])
    assert out["status"] == "UNRESOLVED"
    assert out["explained_amount"] == 0.0
    assert "EVIDENCE_NOT_FOUND" in out["reasoning_summary"]


def test_wrong_type_for_reference_downgrades():
    # FEE1 exists, but not as a REFUND
    v, out = run(resolved([{"type": "REFUND", "reference": "FEE1", "amount": 30.0}]))
    assert out["status"] == "UNRESOLVED"


def test_wrong_amount_downgrades_and_keeps_only_verified_amount():
    res = resolved([
        {"type": "FEE", "reference": "FEE1", "amount": 20.0},
        {"type": "FEE", "reference": "FEE2", "amount": 11.0},  # real is 10.0
    ], explained=31.0)
    v, out = run(res, diff=31.0)
    assert any(c.startswith("AMOUNT_MISMATCH") for c in v["failed_checks"])
    assert out["status"] == "UNRESOLVED"
    assert out["explained_amount"] == 20.0


def test_amounts_not_summing_to_difference_downgrades():
    res = resolved([{"type": "FEE", "reference": "FEE1", "amount": 20.0}], explained=20.0)
    v, out = run(res, diff=30.0)
    assert any(c.startswith("DIFFERENCE_NOT_EXPLAINED") for c in v["failed_checks"])
    assert out["status"] == "UNRESOLVED"
    assert out["explained_amount"] == 20.0


def test_explained_amount_not_matching_evidence_sum_downgrades():
    res = resolved([{"type": "FEE", "reference": "FEE1", "amount": 20.0}], explained=30.0)
    v, out = run(res, diff=30.0)
    assert any(c.startswith("EVIDENCE_SUM_MISMATCH") for c in v["failed_checks"])
    assert out["status"] == "UNRESOLVED"


def test_duplicate_citation_downgrades():
    res = resolved([{"type": "FEE", "reference": "FEE1", "amount": 20.0}] * 2, explained=40.0)
    v, out = run(res, diff=40.0)
    assert any(c.startswith("DUPLICATE_EVIDENCE") for c in v["failed_checks"])


@pytest.mark.parametrize("evidence", [[], None])
def test_empty_evidence_downgrades(evidence):
    v, out = run(resolved(evidence, explained=30.0))
    assert not v["passed"]
    assert v["failed_checks"][0].startswith("EMPTY_EVIDENCE")
    assert out["status"] == "UNRESOLVED"


def test_unresolved_untouched():
    res = {"status": "UNRESOLVED", "confidence": 0.3, "explained_amount": 0.0,
           "evidence_records": [{"type": "FEE", "reference": "BOGUS", "amount": 1.0}]}
    v, out = run(res)
    assert v["passed"] and v["skipped"]
    assert out is res and out == res


def test_sub_paisa_tolerance_passes():
    res = resolved([{"type": "FEE", "reference": "FEE1", "amount": 20.004},
                    {"type": "FEE", "reference": "FEE2", "amount": 10.0}], explained=30.004)
    v, _ = run(res)
    assert v["passed"]


def test_investigator_records_verification_and_downgrades(monkeypatch):
    """End to end: a provider lying about evidence is downgraded and audited."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database import Base
    from app.models.entities import ExceptionRecord, AuditLog
    from app.core.data_generator.generator import generate_synthetic_dataset
    from app.core.reconciliation.engine import ReconciliationEngine
    from app.core.ai import investigator as inv_mod

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    ds = generate_synthetic_dataset(db, "verif", record_count=60, seed=42)
    batch = ReconciliationEngine.run_batch(db, ds.id)
    exc = db.query(ExceptionRecord).filter(ExceptionRecord.batch_id == batch.id).first()

    class LyingProvider:
        def generate_investigation(self, prompt, system_prompt, tools_data):
            return {"status": "RESOLVED", "root_cause": "PROCESSING_FEE", "confidence": 0.99,
                    "financial_difference": float(exc.difference),
                    "explained_amount": float(exc.difference),
                    "evidence_records": [{"type": "FEE", "reference": "NOPE", "amount": float(exc.difference)}],
                    "reasoning_summary": "trust me"}

    monkeypatch.setattr(inv_mod, "get_llm_provider", lambda: LyingProvider())
    inv = inv_mod.AIInvestigator.investigate_exception(db, exc.id)
    assert inv.status == "UNRESOLVED"
    assert "EVIDENCE_NOT_FOUND" in inv.reasoning_summary
    db.refresh(exc)
    assert exc.status == "UNRESOLVED"
    audit = db.query(AuditLog).filter(AuditLog.exception_id == exc.id,
                                      AuditLog.event_type == "AI_INVESTIGATION_COMPLETED").one()
    assert audit.decision == "UNRESOLVED"
    assert audit.evidence["verification"]["status"] == "FAILED"
    assert audit.evidence["verification"]["failed_checks"]
