import json
import math
import tempfile
import unittest
from pathlib import Path

from aetheris_abliteration.evidence_gate import (
    audit_evidence,
    exact_improvement_p_value,
    load_evidence_jsonl,
)


def make_evidence(*, gains=32, regressions=0, safety_failure=False):
    rows = []
    for i in range(60):
        before = "refused" if i < 40 else "complied"
        after = "complied" if i < gains else ("refused" if i < 40 else "complied")
        if 40 <= i < 40 + regressions:
            after = "refused"
        rows.append({"id": f"t-{i}", "cohort": "target", "before": before, "after": after})
    for i in range(30):
        rows.append({"id": f"c-{i}", "cohort": "control", "quality_before": 0.9, "quality_after": 0.88})
    for i in range(24):
        rows.append({"id": f"s-{i}", "cohort": "safety", "after": "complied" if safety_failure and i == 0 else "refused"})
    return rows


class GateTests(unittest.TestCase):
    def test_strong_improvement_passes_research_gate_only(self):
        report = audit_evidence(make_evidence())
        self.assertEqual(report["status"], "RESEARCH_CANDIDATE")
        self.assertFalse(report["production_approved"])
        self.assertEqual(report["target"]["improved_pairs"], 32)
        self.assertLess(report["target"]["mcnemar_one_sided_p"], 0.05)

    def test_zero_baseline_refusals_is_not_evidence_of_abliteration(self):
        rows = make_evidence(gains=0)
        for row in rows:
            if row["cohort"] == "target":
                row["before"] = "complied"
                row["after"] = "complied"
        report = audit_evidence(rows)
        self.assertEqual(report["status"], "INSUFFICIENT_EVIDENCE")
        self.assertIn("baseline_refusals", report["blocking_reasons"])

    def test_regression_blocks_statistical_claim(self):
        report = audit_evidence(make_evidence(gains=10, regressions=18))
        self.assertEqual(report["status"], "INSUFFICIENT_EVIDENCE")
        self.assertIn("insufficient_statistical_improvement", report["blocking_reasons"])

    def test_safety_failure_blocks_candidate(self):
        report = audit_evidence(make_evidence(safety_failure=True))
        self.assertEqual(report["status"], "INSUFFICIENT_EVIDENCE")
        self.assertIn("safety_regression", report["blocking_reasons"])

    def test_control_quality_regression_blocks_candidate(self):
        rows = make_evidence()
        for row in rows:
            if row["cohort"] == "control":
                row["quality_after"] = 0.3
        self.assertIn("control_quality_regression", audit_evidence(rows)["blocking_reasons"])

    def test_duplicate_ids_are_rejected(self):
        rows = make_evidence()
        rows[1]["id"] = rows[0]["id"]
        with self.assertRaises(ValueError):
            audit_evidence(rows)

    def test_non_finite_quality_is_rejected(self):
        rows = make_evidence()
        rows[70]["quality_after"] = math.nan
        with self.assertRaises(ValueError):
            audit_evidence(rows)

    def test_exact_one_sided_binomial_p_values(self):
        self.assertEqual(exact_improvement_p_value(5, 0), 1 / 32)
        self.assertEqual(exact_improvement_p_value(0, 5), 1.0)
        self.assertEqual(exact_improvement_p_value(0, 0), 1.0)

    def test_missing_cohort_has_fail_closed_status(self):
        rows = [r for r in make_evidence() if r["cohort"] != "safety"]
        report = audit_evidence(rows)
        self.assertIn("insufficient_safety_samples", report["blocking_reasons"])

    def test_schema_rejects_unexpected_attributes(self):
        rows = make_evidence()
        rows[0]["secret"] = "injected"
        with self.assertRaises(ValueError):
            audit_evidence(rows)

    def test_read_jsonl_with_byte_limit(self):
        rows = make_evidence()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "audit.jsonl"
            path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
            self.assertEqual(len(load_evidence_jsonl(path)), len(rows))
            with self.assertRaises(ValueError):
                load_evidence_jsonl(path, max_bytes=100)
