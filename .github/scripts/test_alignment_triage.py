#!/usr/bin/env python3
import unittest

from alignment_triage import (
    UNIT_MARKER,
    filed_body,
    find_draft,
    has_unit,
    parse_draft,
    render_draft,
    render_filed_note,
)

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
        note = render_filed_note(
            "candidate-1", "https://github.com/intel/torch-xpu-ops/issues/42", "1", "2026-08-18"
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

    def test_an_automatically_filed_note_cannot_be_filed_again(self) -> None:
        note = render_filed_note(
            "candidate-1", "https://github.com/intel/torch-xpu-ops/issues/42", "1", "2026-08-18"
        )
        with self.assertRaises(SystemExit):
            parse_draft(note, "candidate-1")


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
