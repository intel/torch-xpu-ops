#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import xpu_alignment_publish as publish


def payload(unit_id: str = "issue-123") -> dict:
    return {
        "unit_id": unit_id,
        "title": f"[xpu-alignment] Fix {unit_id} on XPU",
        "body": "Reviewed evidence.",
        "labels": ["ai_generated"],
    }


class PublisherTests(unittest.TestCase):
    def invoke(
        self, decision: dict, *, comments: list[dict] | None = None
    ) -> tuple[mock.Mock, mock.Mock, mock.Mock]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "decision.json"
        path.write_text(json.dumps({"schema_version": 1, **decision}) + "\n")
        post = mock.Mock(side_effect=range(1, 20))
        create = mock.Mock(return_value="https://github.com/intel/torch-xpu-ops/issues/99")
        update = mock.Mock()
        arguments = [
            "xpu_alignment_publish.py",
            "--repo",
            "intel/torch-xpu-ops",
            "--triage-issue",
            "5018",
            "--notify",
            "@owner",
            "--decision",
            str(path),
        ]
        with (
            mock.patch.object(sys, "argv", arguments),
            mock.patch.object(publish, "list_comments", return_value=comments or []),
            mock.patch.object(publish, "post_comment", post),
            mock.patch.object(publish, "create_issue", create),
            mock.patch.object(publish, "update_comment", update),
        ):
            self.assertEqual(publish.main(), 0)
        return post, create, update

    def test_dry_run_posts_draft_and_summary_without_filing_or_notification(self) -> None:
        post, create, update = self.invoke(
            {
                "run_id": "42-1",
                "scan_date": "2026-08-20",
                "mode": "dry-run",
                "decision": "dry-run",
                "would_decision": "file-one",
                "needs_attention": False,
                "attention_reasons": [],
                "blockers": [],
                "payloads": [payload()],
            }
        )

        self.assertEqual(post.call_count, 2)
        self.assertIn("alignment-dry-run-unit", post.call_args_list[0].args[2])
        self.assertIn("[DRY RUN]", post.call_args_list[1].args[2])
        self.assertNotIn("@owner", post.call_args_list[1].args[2])
        create.assert_not_called()
        update.assert_not_called()

    def test_dry_run_reposts_even_when_the_same_dry_marker_exists(self) -> None:
        existing = [
            {"body": "<!-- alignment-dry-run-unit: 42-1:issue-123 -->\n[DRY RUN]"}
        ]
        post, create, _ = self.invoke(
            {
                "run_id": "42-1",
                "scan_date": "2026-08-20",
                "mode": "dry-run",
                "decision": "dry-run",
                "would_decision": "file-one",
                "needs_attention": False,
                "payloads": [payload()],
            },
            comments=existing,
        )

        self.assertEqual(post.call_count, 2)
        create.assert_not_called()

    def test_scheduled_single_candidate_is_filed_and_notified(self) -> None:
        post, create, update = self.invoke(
            {
                "run_id": "42-1",
                "scan_date": "2026-08-20",
                "mode": "schedule",
                "decision": "file-one",
                "would_decision": "file-one",
                "needs_attention": False,
                "payloads": [payload()],
            }
        )

        self.assertEqual(post.call_count, 2)
        self.assertIn("alignment-unit: issue-123", post.call_args_list[0].args[2])
        self.assertIn("@owner", post.call_args_list[1].args[2])
        create.assert_called_once()
        update.assert_called_once()

    def test_scheduled_candidate_is_filed_when_other_units_are_blocked(self) -> None:
        post, create, update = self.invoke(
            {
                "run_id": "42-1",
                "scan_date": "2026-08-20",
                "mode": "schedule",
                "decision": "file-one",
                "would_decision": "file-one",
                "needs_attention": True,
                "attention_reasons": [],
                "blockers": ["scan-blocked-result:issue-456:blocked-env"],
                "payloads": [payload("issue-123")],
            }
        )

        self.assertEqual(post.call_count, 2)
        create.assert_called_once()
        update.assert_called_once()
        summary = post.call_args_list[-1].args[2]
        self.assertIn("published or queued", summary)
        self.assertIn("blocked units were excluded", summary)
        self.assertIn("@owner", summary)

    def test_scheduled_quiet_day_posts_summary_without_notification(self) -> None:
        post, create, _ = self.invoke(
            {
                "run_id": "42-1",
                "scan_date": "2026-08-20",
                "mode": "schedule",
                "decision": "none",
                "would_decision": "none",
                "needs_attention": False,
                "payloads": [],
            }
        )

        self.assertEqual(post.call_count, 1)
        self.assertIn("run complete", post.call_args.args[2])
        self.assertNotIn("@owner", post.call_args.args[2])
        create.assert_not_called()

    def test_scheduled_multiple_candidates_queue_drafts_and_notify(self) -> None:
        post, create, _ = self.invoke(
            {
                "run_id": "42-1",
                "scan_date": "2026-08-20",
                "mode": "schedule",
                "decision": "triage",
                "would_decision": "triage",
                "needs_attention": False,
                "payloads": [payload("issue-123"), payload("issue-456")],
            }
        )

        self.assertEqual(post.call_count, 3)
        self.assertIn("@owner", post.call_args_list[-1].args[2])
        create.assert_not_called()

    def test_blocked_schedule_posts_only_a_notifying_summary(self) -> None:
        post, create, _ = self.invoke(
            {
                "run_id": "42-1",
                "scan_date": "2026-08-20",
                "mode": "schedule",
                "decision": "blocked",
                "would_decision": "blocked",
                "needs_attention": True,
                "attention_reasons": [],
                "blockers": ["runner-log-digest-mismatch:issue-123"],
                "payloads": [],
            }
        )

        self.assertEqual(post.call_count, 1)
        self.assertIn("runner-log-digest-mismatch", post.call_args.args[2])
        self.assertIn("@owner", post.call_args.args[2])
        create.assert_not_called()

    def test_partial_schedule_posts_idempotent_diagnostics_and_notifies(self) -> None:
        decision = {
            "run_id": "42-1",
            "scan_date": "2026-08-20",
            "mode": "schedule",
            "decision": "diagnostic",
            "would_decision": "diagnostic",
            "collection_status": "partial",
            "collection_progress": {
                "pull_requests_created": {
                    "pages_completed": 3,
                    "items_fetched": 250,
                    "last_cursor": "cursor-3",
                    "rate_remaining": 0,
                    "rate_reset_at": "2026-08-21T02:15:00Z",
                    "status": "partial",
                    "error": "rate limit did not recover",
                }
            },
            "needs_attention": True,
            "attention_reasons": [],
            "blockers": ["pull_requests_created:rate-limit"],
            "payloads": [payload()],
        }

        post, create, update = self.invoke(decision)

        self.assertEqual(post.call_count, 2)
        draft = post.call_args_list[0].args[2]
        summary = post.call_args_list[1].args[2]
        self.assertIn("alignment-diagnostic-unit", draft)
        self.assertNotIn("<!-- alignment-unit: issue-123 -->", draft)
        self.assertIn("[INCOMPLETE SCAN]", draft)
        self.assertIn("3 page(s), 250 item(s)", summary)
        self.assertIn("@owner", summary)
        create.assert_not_called()
        update.assert_not_called()

        existing = [
            {"id": 71, "body": draft},
            {"id": 72, "body": summary},
        ]
        retry_post, retry_create, retry_update = self.invoke(decision, comments=existing)
        retry_post.assert_not_called()
        retry_create.assert_not_called()
        retry_update.assert_not_called()

    def test_partial_dry_run_reposts_diagnostics_without_notification(self) -> None:
        decision = {
            "run_id": "42-1",
            "scan_date": "2026-08-20",
            "mode": "dry-run",
            "decision": "dry-run-diagnostic",
            "would_decision": "diagnostic",
            "collection_status": "partial",
            "collection_progress": {},
            "needs_attention": True,
            "attention_reasons": [],
            "blockers": ["issues_created:http-429"],
            "payloads": [payload()],
        }
        existing = [
            {
                "id": 71,
                "body": "<!-- alignment-dry-run-diagnostic-unit: 42-1:issue-123 -->",
            }
        ]

        post, create, update = self.invoke(decision, comments=existing)

        self.assertEqual(post.call_count, 2)
        self.assertIn("alignment-dry-run-diagnostic-unit", post.call_args_list[0].args[2])
        summary = post.call_args_list[1].args[2]
        self.assertIn("[DRY RUN][INCOMPLETE SCAN]", summary)
        self.assertNotIn("@owner", summary)
        create.assert_not_called()
        update.assert_not_called()

    def test_scheduled_retry_resumes_an_unfiled_formal_draft(self) -> None:
        existing = [
            {
                "id": 77,
                "body": publish.render_draft(
                    "issue-123",
                    "[xpu-alignment] Fix issue-123 on XPU",
                    "Reviewed evidence.",
                    "42-1",
                    "2026-08-20",
                ),
            }
        ]

        post, create, update = self.invoke(
            {
                "run_id": "42-1",
                "scan_date": "2026-08-20",
                "mode": "schedule",
                "decision": "file-one",
                "would_decision": "file-one",
                "needs_attention": False,
                "payloads": [payload()],
            },
            comments=existing,
        )

        create.assert_called_once()
        update.assert_called_once()
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
