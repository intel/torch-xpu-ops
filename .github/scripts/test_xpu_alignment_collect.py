#!/usr/bin/env python3

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from xpu_alignment_collect import (
    CollectionError,
    FetchError,
    GitHubGraphQL,
    collect,
    validate_collection,
)


class FakeGitHub:
    def __init__(self, pages: dict[str, list[dict]]) -> None:
        self.pages = pages
        self.positions = dict.fromkeys(pages, 0)

    def snapshot(self, repository: str) -> dict:
        return {"default_branch": "main", "default_branch_head": "a" * 40}

    def page(
        self,
        repository: str,
        source: str,
        cursor: str | None,
        snapshot: dict,
        scan_window: dict,
    ) -> dict:
        position = self.positions[source]
        self.positions[source] += 1
        result = self.pages[source][position]
        if isinstance(result, Exception):
            raise result
        return result


def page(nodes: list[dict], *, next_cursor: str | None = None) -> dict:
    return {
        "nodes": nodes,
        "page_info": {"has_next_page": next_cursor is not None, "end_cursor": next_cursor},
        "rate": {"remaining": 900, "reset_at": "2026-08-21T03:00:00Z"},
        "raw": {"data": {"nodes": nodes, "next": next_cursor}},
    }


class CollectorTests(unittest.TestCase):
    def test_collects_every_page_and_deduplicates_pr_events(self) -> None:
        github = FakeGitHub(
            {
                "issues-created": [
                    page(
                        [
                            {
                                "id": "issue-123",
                                "kind": "issue",
                                "title": "Issue",
                                "url": "https://github.com/pytorch/pytorch/issues/123",
                                "event_at": "2026-08-20T20:00:00Z",
                            }
                        ],
                        next_cursor="issues-page-2",
                    ),
                    page([]),
                ],
                "prs-created": [
                    page(
                        [
                            {
                                "id": "pr-456",
                                "kind": "pr",
                                "title": "PR",
                                "url": "https://github.com/pytorch/pytorch/pull/456",
                                "event_at": "2026-08-20T10:00:00Z",
                            }
                        ]
                    )
                ],
                "prs-merged": [
                    page(
                        [
                            {
                                "id": "pr-456",
                                "kind": "pr",
                                "title": "PR",
                                "url": "https://github.com/pytorch/pytorch/pull/456",
                                "event_at": "2026-08-20T22:00:00Z",
                                "order_at": "2026-08-20T22:00:00Z",
                            }
                        ]
                    )
                ],
                "default-branch-commits": [
                    page(
                        [
                            {
                                "id": "commit-" + "b" * 40,
                                "kind": "commit",
                                "title": "Commit",
                                "url": "https://github.com/pytorch/pytorch/commit/" + "b" * 40,
                                "event_at": "2026-08-20T12:00:00Z",
                            }
                        ]
                    )
                ],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            result = collect(
                "pytorch/pytorch",
                {"start": "2026-08-20T00:00:00Z", "end": "2026-08-21T00:00:00Z"},
                Path(directory),
                github,
            )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["observed_count"], 4)
        self.assertEqual(result["unique_count"], 3)
        self.assertEqual(result["sources"][0]["pages_completed"], 2)
        pr = next(item for item in result["inventory"] if item["id"] == "pr-456")
        self.assertEqual(pr["events"], [
            {"type": "created", "at": "2026-08-20T10:00:00Z"},
            {"type": "merged", "at": "2026-08-20T22:00:00Z"},
        ])

    def test_stops_at_the_time_boundary_and_uses_a_half_open_window(self) -> None:
        def node(unit_id: str, at: str) -> dict:
            return {
                "id": unit_id,
                "kind": "issue",
                "title": unit_id,
                "url": f"https://github.com/pytorch/pytorch/issues/{unit_id}",
                "event_at": at,
            }

        github = FakeGitHub(
            {
                "issues-created": [
                    page(
                        [
                            node("issue-after", "2026-08-21T00:00:00Z"),
                            node("issue-end", "2026-08-20T23:59:59Z"),
                            node("issue-start", "2026-08-20T00:00:00Z"),
                            node("issue-before", "2026-08-19T23:59:59Z"),
                        ],
                        next_cursor="unused",
                    )
                ],
                "prs-created": [page([])],
                "prs-merged": [page([])],
                "default-branch-commits": [page([])],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            result = collect(
                "pytorch/pytorch",
                {"start": "2026-08-20T00:00:00Z", "end": "2026-08-21T00:00:00Z"},
                Path(directory),
                github,
            )

        self.assertEqual(
            [item["id"] for item in result["inventory"]],
            ["issue-end", "issue-start"],
        )
        self.assertEqual(github.positions["issues-created"], 1)

    def test_preserves_progress_when_a_source_is_rate_limited(self) -> None:
        github = FakeGitHub(
            {
                "issues-created": [
                    page(
                        [
                            {
                                "id": "issue-123",
                                "kind": "issue",
                                "title": "Issue",
                                "url": "https://github.com/pytorch/pytorch/issues/123",
                                "event_at": "2026-08-20T20:00:00Z",
                            }
                        ],
                        next_cursor="issues-page-2",
                    ),
                    FetchError(
                        "rate-limit",
                        "GitHub rate limit did not reset within 600 seconds",
                        remaining=0,
                        reset_at="2026-08-21T03:00:00Z",
                    ),
                ],
                "prs-created": [page([])],
                "prs-merged": [page([])],
                "default-branch-commits": [page([])],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = collect(
                "pytorch/pytorch",
                {"start": "2026-08-20T00:00:00Z", "end": "2026-08-21T00:00:00Z"},
                root,
                github,
            )

            self.assertTrue((root / "collection/pages/issues-created/page_0001.json").is_file())
            _, persisted, persisted_inventory = validate_collection(
                root,
                {"start": "2026-08-20T00:00:00Z", "end": "2026-08-21T00:00:00Z"},
            )

        self.assertEqual(result["status"], "partial")
        self.assertEqual(persisted["status"], "partial")
        self.assertEqual(set(persisted_inventory), {"issue-123"})
        self.assertEqual(result["unique_count"], 1)
        progress = result["sources"][0]
        self.assertEqual(progress["pages_completed"], 1)
        self.assertEqual(progress["items_fetched"], 1)
        self.assertEqual(progress["last_cursor"], "issues-page-2")
        self.assertEqual(progress["rate_remaining"], 0)
        self.assertEqual(progress["error"]["kind"], "rate-limit")
        self.assertEqual(result["blockers"], ["issues-created:rate-limit"])

    def test_rejects_a_tampered_raw_page(self) -> None:
        github = FakeGitHub(
            {
                "issues-created": [page([])],
                "prs-created": [page([])],
                "prs-merged": [page([])],
                "default-branch-commits": [page([])],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collect(
                "pytorch/pytorch",
                {"start": "2026-08-20T00:00:00Z", "end": "2026-08-21T00:00:00Z"},
                root,
                github,
            )
            raw_page = root / "collection/pages/issues-created/page_0001.json"
            raw_page.write_text("{}\n")

            with self.assertRaisesRegex(CollectionError, "digest"):
                validate_collection(
                    root,
                    {"start": "2026-08-20T00:00:00Z", "end": "2026-08-21T00:00:00Z"},
                )

    def test_rejects_malformed_inventory_metadata(self) -> None:
        github = FakeGitHub(
            {
                "issues-created": [
                    page(
                        [
                            {
                                "id": "issue-123",
                                "kind": "issue",
                                "title": "Issue",
                                "url": "https://github.com/pytorch/pytorch/issues/123",
                                "event_at": "2026-08-20T20:00:00Z",
                            }
                        ]
                    )
                ],
                "prs-created": [page([])],
                "prs-merged": [page([])],
                "default-branch-commits": [page([])],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collect(
                "pytorch/pytorch",
                {"start": "2026-08-20T00:00:00Z", "end": "2026-08-21T00:00:00Z"},
                root,
                github,
            )
            manifest = root / "collection/collection.json"
            value = json.loads(manifest.read_text())
            value["inventory"][0]["url"] = "https://example.invalid/123"
            manifest.write_text(json.dumps(value) + "\n")

            with self.assertRaisesRegex(CollectionError, "metadata"):
                validate_collection(
                    root,
                    {"start": "2026-08-20T00:00:00Z", "end": "2026-08-21T00:00:00Z"},
                )

    def test_repeated_cursor_marks_the_checkpoint_malformed(self) -> None:
        github = FakeGitHub(
            {
                "issues-created": [
                    page([], next_cursor="cursor-1"),
                    page([], next_cursor="cursor-1"),
                ],
                "prs-created": [page([])],
                "prs-merged": [page([])],
                "default-branch-commits": [page([])],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "invalid cursor"):
                collect(
                    "pytorch/pytorch",
                    {
                        "start": "2026-08-20T00:00:00Z",
                        "end": "2026-08-21T00:00:00Z",
                    },
                    root,
                    github,
                )
            manifest = root / "collection/collection.json"
            self.assertTrue(manifest.is_file())
            self.assertEqual(json.loads(manifest.read_text())["status"], "malformed")
            with self.assertRaisesRegex(CollectionError, "status"):
                validate_collection(
                    root,
                    {"start": "2026-08-20T00:00:00Z", "end": "2026-08-21T00:00:00Z"},
                )

    def test_non_adjacent_repeated_cursor_marks_the_checkpoint_malformed(self) -> None:
        github = FakeGitHub(
            {
                "issues-created": [
                    page([], next_cursor="cursor-a"),
                    page([], next_cursor="cursor-b"),
                    page([], next_cursor="cursor-a"),
                    page([]),
                ],
                "prs-created": [page([])],
                "prs-merged": [page([])],
                "default-branch-commits": [page([])],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "invalid cursor"):
                collect(
                    "pytorch/pytorch",
                    {
                        "start": "2026-08-20T00:00:00Z",
                        "end": "2026-08-21T00:00:00Z",
                    },
                    root,
                    github,
                )

            manifest = root / "collection/collection.json"
            self.assertEqual(json.loads(manifest.read_text())["status"], "malformed")

    def test_validator_rejects_a_non_adjacent_cursor_cycle(self) -> None:
        github = FakeGitHub(
            {
                "issues-created": [
                    page([], next_cursor="cursor-a"),
                    page([], next_cursor="cursor-b"),
                    page([], next_cursor="cursor-c"),
                    page([]),
                ],
                "prs-created": [page([])],
                "prs-merged": [page([])],
                "default-branch-commits": [page([])],
            }
        )
        scan_window = {
            "start": "2026-08-20T00:00:00Z",
            "end": "2026-08-21T00:00:00Z",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collect("pytorch/pytorch", scan_window, root, github)
            manifest = root / "collection/collection.json"
            payload = json.loads(manifest.read_text())
            pages = payload["sources"][0]["pages"]
            pages[2]["next_cursor"] = "cursor-a"
            pages[3]["cursor"] = "cursor-a"
            manifest.write_text(json.dumps(payload) + "\n")

            with self.assertRaisesRegex(CollectionError, "cursor chain"):
                validate_collection(root, scan_window)

    def test_collects_more_than_one_thousand_candidates_without_search(self) -> None:
        issue_pages = []
        for page_number in range(12):
            nodes = [
                {
                    "id": f"issue-{page_number * 100 + offset}",
                    "kind": "issue",
                    "title": "Issue",
                    "url": (
                        "https://github.com/pytorch/pytorch/issues/"
                        f"{page_number * 100 + offset}"
                    ),
                    "event_at": "2026-08-20T12:00:00Z",
                }
                for offset in range(100)
            ]
            next_cursor = f"page-{page_number + 2}" if page_number < 11 else None
            issue_pages.append(page(nodes, next_cursor=next_cursor))
        github = FakeGitHub(
            {
                "issues-created": issue_pages,
                "prs-created": [page([])],
                "prs-merged": [page([])],
                "default-branch-commits": [page([])],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            result = collect(
                "pytorch/pytorch",
                {"start": "2026-08-20T00:00:00Z", "end": "2026-08-21T00:00:00Z"},
                Path(directory),
                github,
            )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["unique_count"], 1200)
        self.assertEqual(result["sources"][0]["pages_completed"], 12)


class GraphQLAdapterTests(unittest.TestCase):
    def test_retries_a_graphql_rate_limit_response(self) -> None:
        limited_payload = {
            "data": {"rateLimit": {"remaining": 0, "resetAt": None}},
            "errors": [{"type": "RATE_LIMITED", "message": "quota exhausted"}],
        }
        success_payload = {
            "data": {
                "repository": {
                    "defaultBranchRef": {"name": "main", "target": {"oid": "a" * 40}}
                },
                "rateLimit": {"remaining": 999, "resetAt": "2026-08-21T03:00:00Z"},
            }
        }
        limited = io.BytesIO(json.dumps(limited_payload).encode())
        limited.__enter__ = mock.Mock(return_value=limited)  # type: ignore[attr-defined]
        limited.__exit__ = mock.Mock(return_value=False)  # type: ignore[attr-defined]
        success = io.BytesIO(json.dumps(success_payload).encode())
        success.__enter__ = mock.Mock(return_value=success)  # type: ignore[attr-defined]
        success.__exit__ = mock.Mock(return_value=False)  # type: ignore[attr-defined]
        with (
            mock.patch("urllib.request.urlopen", side_effect=[limited, success]),
            mock.patch("time.sleep") as sleep,
        ):
            snapshot = GitHubGraphQL("token").snapshot("pytorch/pytorch")

        self.assertEqual(snapshot["default_branch_head"], "a" * 40)
        sleep.assert_called_once_with(60)

    def test_retries_a_server_error_before_reading_the_snapshot(self) -> None:
        payload = {
            "data": {
                "repository": {
                    "defaultBranchRef": {"name": "main", "target": {"oid": "a" * 40}}
                },
                "rateLimit": {"remaining": 999, "resetAt": "2026-08-21T03:00:00Z"},
            }
        }
        response = io.BytesIO(json.dumps(payload).encode())
        response.__enter__ = mock.Mock(return_value=response)  # type: ignore[attr-defined]
        response.__exit__ = mock.Mock(return_value=False)  # type: ignore[attr-defined]
        server_error = urllib.error.HTTPError(
            "https://api.github.com/graphql", 500, "server error", {}, None
        )
        with (
            mock.patch("urllib.request.urlopen", side_effect=[server_error, response]),
            mock.patch("time.sleep") as sleep,
        ):
            snapshot = GitHubGraphQL("token").snapshot("pytorch/pytorch")

        self.assertEqual(snapshot["default_branch_head"], "a" * 40)
        sleep.assert_called_once_with(1)

    def test_honors_retry_after_for_http_429(self) -> None:
        payload = {
            "data": {
                "repository": {
                    "defaultBranchRef": {"name": "main", "target": {"oid": "a" * 40}}
                },
                "rateLimit": {"remaining": 998, "resetAt": "2026-08-21T03:00:00Z"},
            }
        }
        response = io.BytesIO(json.dumps(payload).encode())
        response.__enter__ = mock.Mock(return_value=response)  # type: ignore[attr-defined]
        response.__exit__ = mock.Mock(return_value=False)  # type: ignore[attr-defined]
        rate_limit = urllib.error.HTTPError(
            "https://api.github.com/graphql",
            429,
            "rate limited",
            {"retry-after": "3", "x-ratelimit-remaining": "0"},
            None,
        )
        with (
            mock.patch("urllib.request.urlopen", side_effect=[rate_limit, response]),
            mock.patch("time.sleep") as sleep,
        ):
            snapshot = GitHubGraphQL("token").snapshot("pytorch/pytorch")

        self.assertEqual(snapshot["default_branch_head"], "a" * 40)
        sleep.assert_called_once_with(3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
