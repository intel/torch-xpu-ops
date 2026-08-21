#!/usr/bin/env python3
import unittest

from alignment_triage import (
    UNIT_MARKER,
    filed_body,
    find_draft,
    has_run_note,
    has_unit,
    parse_draft,
    render_draft,
    render_run_note,
)
from xpu_alignment_publish import run_note

TITLE = "[xpu-alignment] Fix GELU erf-tail catastrophic cancellation"


def draft(unit_id: str, title: str = TITLE, body: str = "Details.") -> str:
    return render_draft(unit_id, title, body, "12345", "2026-08-18")


def comment(unit_id: str, **kwargs: str) -> dict:
    return {"id": 1, "body": draft(unit_id, **kwargs)}


class DraftLookupTests(unittest.TestCase):
    def test_finds_the_comment_carrying_the_marker(self) -> None:
        comments = [{"id": 7, "body": "chatter"}, comment("candidate-1")]
        self.assertEqual(find_draft(comments, "candidate-1")["id"], 1)

    def test_unknown_unit_id_fails(self) -> None:
        with self.assertRaises(SystemExit):
            find_draft([comment("candidate-1")], "candidate-2")

    def test_duplicate_markers_fail(self) -> None:
        with self.assertRaises(SystemExit):
            find_draft([comment("candidate-1"), comment("candidate-1")], "candidate-1")

    def test_marker_is_not_matched_by_a_longer_id(self) -> None:
        with self.assertRaises(SystemExit):
            find_draft([comment("candidate-10")], "candidate-1")

    def test_has_unit_detects_an_already_queued_draft(self) -> None:
        self.assertTrue(has_unit([comment("candidate-1")], "candidate-1"))
        self.assertFalse(has_unit([comment("candidate-1")], "candidate-2"))

    def test_has_unit_detects_an_automatically_filed_unit(self) -> None:
        note = filed_body(
            draft("candidate-1"), "candidate-1", "https://github.com/intel/torch-xpu-ops/issues/42"
        )
        self.assertTrue(has_unit([{"id": 3, "body": note}], "candidate-1"))


class DraftParsingTests(unittest.TestCase):
    def test_splits_title_and_body(self) -> None:
        title, body = parse_draft(draft("candidate-1"), "candidate-1")
        self.assertEqual(title, TITLE)
        self.assertEqual(body, "Details.")

    def test_preserves_a_multiline_body_verbatim(self) -> None:
        body_text = "## Repro\n\n```python\nimport torch\n```\n\n### Notes\nmore"
        title, body = parse_draft(draft("candidate-1", body=body_text), "candidate-1")
        self.assertEqual(title, TITLE)
        self.assertEqual(body, body_text)

    def test_foreign_title_prefix_fails(self) -> None:
        with self.assertRaises(SystemExit):
            parse_draft(draft("candidate-1", title="Fix everything"), "candidate-1")

    def test_missing_title_line_fails(self) -> None:
        with self.assertRaises(SystemExit):
            parse_draft(f"{UNIT_MARKER.format(unit_id='candidate-1')}\nno heading\n", "candidate-1")

    def test_empty_body_fails(self) -> None:
        with self.assertRaises(SystemExit):
            parse_draft(draft("candidate-1", body=""), "candidate-1")

    def test_an_already_filed_draft_cannot_be_filed_twice(self) -> None:
        filed = filed_body(
            draft("candidate-1"), "candidate-1", "https://github.com/intel/torch-xpu-ops/issues/42"
        )
        with self.assertRaises(SystemExit):
            parse_draft(filed, "candidate-1")


class FiledMarkerTests(unittest.TestCase):
    def test_records_the_issue_number_and_keeps_the_draft(self) -> None:
        updated = filed_body(
            draft("candidate-1"), "candidate-1", "https://github.com/intel/torch-xpu-ops/issues/42"
        )
        self.assertIn("<!-- alignment-unit-filed: #42 -->", updated)
        self.assertIn(UNIT_MARKER.format(unit_id="candidate-1"), updated)
        self.assertIn(f"### {TITLE}", updated)

    def test_marks_only_the_first_occurrence(self) -> None:
        body = draft("candidate-1") + draft("candidate-1")
        updated = filed_body(body, "candidate-1", "https://github.com/x/y/issues/9")
        self.assertEqual(updated.count("<!-- alignment-unit-filed: #9 -->"), 1)


class RunNoteTests(unittest.TestCase):
    def test_mentions_the_maintainers_and_is_not_mistaken_for_a_draft(self) -> None:
        body = render_run_note("12345", "2026-08-18", "Needs a human", ["- x"], "@a @b")
        self.assertIn("@a @b", body)
        with self.assertRaises(SystemExit):
            find_draft([{"id": 1, "body": body}], "candidate-1")

    def test_a_rerun_does_not_ping_twice(self) -> None:
        body = render_run_note("12345", "2026-08-18", "Needs a human", ["- x"], "@a")
        self.assertTrue(has_run_note([{"id": 1, "body": body}], "12345"))
        self.assertFalse(has_run_note([{"id": 1, "body": body}], "12346"))

    def test_a_quiet_day_pings_nobody(self) -> None:
        self.assertIsNone(run_note({"decision": "none", "needs_attention": False}, [], []))

    def test_an_unattended_filing_is_announced(self) -> None:
        decision = {"decision": "file-one", "needs_attention": False}
        payloads = [{"unit_id": "c1", "title": TITLE}]
        url = "https://github.com/intel/torch-xpu-ops/issues/42"
        headline, lines = run_note(decision, payloads, [("c1", url)])
        self.assertIn("filed automatically", headline)
        self.assertTrue(any(url in line for line in lines))

    def test_two_candidates_are_all_listed(self) -> None:
        decision = {"decision": "triage", "needs_attention": False}
        payloads = [
            {"unit_id": "c1", "title": TITLE},
            {"unit_id": "c2", "title": "[xpu-alignment] Other"},
        ]
        headline, lines = run_note(decision, payloads, [])
        self.assertIn("2 XPU alignment candidates", headline)
        self.assertTrue(any("`c1`" in line for line in lines))
        self.assertTrue(any("`c2`" in line for line in lines))

    def test_a_blocked_run_still_reaches_a_human(self) -> None:
        decision = {
            "decision": "blocked",
            "needs_attention": True,
            "blockers": ["review-blocked:reports/reviewer_manifest.json"],
        }
        headline, lines = run_note(decision, [], [])
        self.assertIn("needs attention", headline)
        self.assertTrue(any("review-blocked" in line for line in lines))


if __name__ == "__main__":
    unittest.main(verbosity=2)
