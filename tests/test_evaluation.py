"""
Tests for evaluation.py — the retrieval-quality harness.

Fully offline: a stub retriever (``question -> [doc_id]``) drives the metrics, so
no FAISS index, embeddings, or API key are needed. Expected aggregate values are
hand-computed from a 3-question gold set over the ranking [A, B, C]:

  q1 rel {A}   -> recall 1.0,  prec 1/3, hit 1, rr 1.0, ap 1.0
  q2 rel {B,D} -> recall 0.5,  prec 1/3, hit 1, rr 0.5, ap 0.25
  q3 rel {Z}   -> recall 0.0,  prec 0.0, hit 0, rr 0.0, ap 0.0
"""
import os
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evaluation as ev  # noqa: E402


def _doc(source, page):
    return types.SimpleNamespace(metadata={"source": source, "page": page})


class TestDocIds(unittest.TestCase):
    def test_dedupe_preserves_rank_order(self):
        docs = [_doc("a.pdf", 3), _doc("a.pdf", 3), _doc("b.pdf", 1)]
        self.assertEqual(ev.ids_from_docs(docs), ["a.pdf::p3", "b.pdf::p1"])

    def test_missing_metadata_is_tolerated(self):
        self.assertEqual(ev.ids_from_docs([types.SimpleNamespace(metadata={})]),
                         ["document::p?"])


class TestPointwiseMetrics(unittest.TestCase):
    def test_recall_respects_k_cutoff(self):
        self.assertEqual(ev.recall_at_k(["B", "A"], {"A"}, 1), 0.0)  # A ranked 2nd
        self.assertEqual(ev.recall_at_k(["B", "A"], {"A"}, 2), 1.0)

    def test_recall_with_multiple_relevant(self):
        self.assertEqual(ev.recall_at_k(["A", "B", "C"], {"B", "D"}, 3), 0.5)

    def test_precision_is_over_returned_items(self):
        self.assertAlmostEqual(ev.precision_at_k(["A", "B", "C"], {"B"}, 3), 1 / 3)

    def test_hit_at_k(self):
        self.assertEqual(ev.hit_at_k(["A", "B"], {"B"}, 2), 1.0)
        self.assertEqual(ev.hit_at_k(["A", "B"], {"Z"}, 2), 0.0)

    def test_reciprocal_rank(self):
        self.assertEqual(ev.reciprocal_rank(["A", "B", "C"], {"B"}), 0.5)
        self.assertEqual(ev.reciprocal_rank(["A", "B"], {"Z"}), 0.0)

    def test_average_precision(self):
        # relevant at ranks 1 and 3 -> (1/1 + 2/3)/2
        self.assertAlmostEqual(ev.average_precision(["A", "B", "C"], {"A", "C"}),
                               (1.0 + 2 / 3) / 2)

    def test_empty_relevant_set_is_zero(self):
        self.assertEqual(ev.recall_at_k(["A"], set(), 3), 0.0)
        self.assertEqual(ev.average_precision(["A"], set()), 0.0)


class TestAggregateEvaluate(unittest.TestCase):
    def setUp(self):
        ranking = ["A", "B", "C"]
        self.gold = [
            {"question": "q1", "relevant": ["A"]},
            {"question": "q2", "relevant": ["B", "D"]},
            {"question": "q3", "relevant": ["Z"]},
        ]
        self.result = ev.evaluate(lambda q: ranking, self.gold, k=3)

    def test_aggregate_values(self):
        r = self.result
        self.assertEqual(r["questions"], 3)
        self.assertAlmostEqual(r["recall@k"], 0.5)
        self.assertAlmostEqual(r["precision@k"], 0.2222, places=4)
        self.assertAlmostEqual(r["hit_rate@k"], 0.6667, places=4)
        self.assertAlmostEqual(r["mrr"], 0.5)
        self.assertAlmostEqual(r["map"], 0.4167, places=4)

    def test_per_question_breakdown_present(self):
        self.assertEqual(len(self.result["per_question"]), 3)

    def test_report_renders(self):
        text = ev.format_report(self.result)
        self.assertIn("Retrieval Evaluation", text)
        self.assertIn("citation accuracy", text)


if __name__ == "__main__":
    unittest.main()
