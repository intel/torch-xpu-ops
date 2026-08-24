#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Collect an auditable upstream inventory for XPU alignment scans."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol


SOURCES = (
    "issues-created",
    "prs-created",
    "prs-merged",
    "default-branch-commits",
)
EVENT_TYPES = {
    "issues-created": "created",
    "prs-created": "created",
    "prs-merged": "merged",
    "default-branch-commits": "committed",
}


class FetchError(RuntimeError):
    def __init__(
        self,
        kind: str,
        message: str,
        *,
        remaining: int | None = None,
        reset_at: str | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.remaining = remaining
        self.reset_at = reset_at


class CollectionError(ValueError):
    pass


class GitHub(Protocol):
    def snapshot(self, repository: str) -> dict[str, object]: ...

    def page(
        self,
        repository: str,
        source: str,
        cursor: str | None,
        snapshot: dict[str, object],
        scan_window: dict[str, str],
    ) -> dict[str, object]: ...


class GitHubGraphQL:
    def __init__(
        self,
        token: str,
        *,
        wait_budget_seconds: int = 600,
        overall_budget_seconds: int = 28 * 60,
    ) -> None:
        self.token = token
        started = time.monotonic()
        self.wait_deadline = started + wait_budget_seconds
        self.overall_deadline = started + overall_budget_seconds

    def _request(self, query: str, variables: dict[str, object]) -> dict[str, object]:
        body = json.dumps({"query": query, "variables": variables}).encode()
        request = urllib.request.Request(
            "https://api.github.com/graphql",
            data=body,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "User-Agent": "torch-xpu-ops-xpu-alignment",
            },
        )
        server_attempt = 0
        while True:
            remaining_runtime = self.overall_deadline - time.monotonic()
            if remaining_runtime <= 0:
                raise FetchError(
                    "collector-timeout",
                    "collector reached its internal runtime budget",
                )
            try:
                with urllib.request.urlopen(
                    request, timeout=min(60, max(1, remaining_runtime))
                ) as response:
                    payload = json.load(response)
                if not isinstance(payload, dict):
                    raise ValueError("GitHub GraphQL response is not an object")
                errors = payload.get("errors")
                if errors:
                    data = payload.get("data")
                    rate = data.get("rateLimit") if isinstance(data, dict) else {}
                    rate = rate if isinstance(rate, dict) else {}
                    limited = any(
                        isinstance(error, dict) and error.get("type") == "RATE_LIMITED"
                        for error in errors
                    )
                    if not limited:
                        raise FetchError(
                            "graphql-error", json.dumps(errors, sort_keys=True)
                        )
                    reset_at = rate.get("resetAt")
                    delay = 60
                    if isinstance(reset_at, str):
                        try:
                            reset_time = datetime.strptime(
                                reset_at, "%Y-%m-%dT%H:%M:%SZ"
                            ).replace(tzinfo=timezone.utc)
                            delay = max(1, int(reset_time.timestamp() - time.time()))
                        except ValueError:
                            reset_at = None
                    if time.monotonic() + delay > min(
                        self.wait_deadline, self.overall_deadline
                    ):
                        raise FetchError(
                            "rate-limit",
                            "GitHub GraphQL rate limit did not reset within the wait budget",
                            remaining=rate.get("remaining"),
                            reset_at=reset_at,
                        )
                    time.sleep(delay)
                    continue
                return payload
            except urllib.error.HTTPError as error:
                if 500 <= error.code < 600:
                    if server_attempt >= 4:
                        raise FetchError("server-error", str(error)) from error
                    delay = 2**server_attempt
                    server_attempt += 1
                    if time.monotonic() + delay > min(
                        self.wait_deadline, self.overall_deadline
                    ):
                        raise FetchError("server-error", str(error)) from error
                    time.sleep(delay)
                    continue
                remaining = error.headers.get("x-ratelimit-remaining")
                reset = error.headers.get("x-ratelimit-reset")
                retry_after = error.headers.get("retry-after")
                limited = (
                    error.code in {403, 429}
                    or retry_after is not None
                    or remaining == "0"
                )
                if not limited:
                    raise FetchError(f"github-http-{error.code}", str(error)) from error
                now = time.time()
                if retry_after is not None:
                    delay = max(1, int(retry_after))
                elif reset is not None:
                    delay = max(1, int(reset) - int(now))
                else:
                    delay = 60
                reset_at = (
                    datetime.fromtimestamp(int(reset), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    if reset is not None
                    else None
                )
                if time.monotonic() + delay > min(
                    self.wait_deadline, self.overall_deadline
                ):
                    raise FetchError(
                        "rate-limit",
                        "GitHub rate limit did not reset within the wait budget",
                        remaining=int(remaining) if remaining is not None else None,
                        reset_at=reset_at,
                    ) from error
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError) as error:
                if server_attempt >= 4:
                    raise FetchError("network-error", str(error)) from error
                delay = 2**server_attempt
                server_attempt += 1
                if time.monotonic() + delay > min(
                    self.wait_deadline, self.overall_deadline
                ):
                    raise FetchError("network-error", str(error)) from error
                time.sleep(delay)

    def snapshot(self, repository: str) -> dict[str, object]:
        owner, name = repository.split("/", 1)
        payload = self._request(
            """
            query($owner: String!, $name: String!) {
              repository(owner: $owner, name: $name) {
                defaultBranchRef { name target { oid } }
              }
              rateLimit { remaining resetAt }
            }
            """,
            {"owner": owner, "name": name},
        )
        branch = payload.get("data", {}).get("repository", {}).get("defaultBranchRef", {})
        name_value = branch.get("name") if isinstance(branch, dict) else None
        target = branch.get("target") if isinstance(branch, dict) else None
        head = target.get("oid") if isinstance(target, dict) else None
        if not isinstance(name_value, str) or not isinstance(head, str):
            raise ValueError("GitHub response has no default branch snapshot")
        return {"default_branch": name_value, "default_branch_head": head}

    def page(
        self,
        repository: str,
        source: str,
        cursor: str | None,
        snapshot: dict[str, object],
        scan_window: dict[str, str],
    ) -> dict[str, object]:
        owner, name = repository.split("/", 1)
        variables: dict[str, object] = {"owner": owner, "name": name, "cursor": cursor}
        if source == "issues-created":
            connection = "issues"
            query = """
                query($owner: String!, $name: String!, $cursor: String) {
                  repository(owner: $owner, name: $name) {
                    result: issues(first: 100, after: $cursor,
                      orderBy: {field: CREATED_AT, direction: DESC}) {
                      nodes { number title url createdAt }
                      pageInfo { hasNextPage endCursor }
                    }
                  }
                  rateLimit { remaining resetAt }
                }
            """
        elif source in {"prs-created", "prs-merged"}:
            connection = "pullRequests"
            states = ", states: MERGED" if source == "prs-merged" else ""
            order = "UPDATED_AT" if source == "prs-merged" else "CREATED_AT"
            query = f"""
                query($owner: String!, $name: String!, $cursor: String) {{
                  repository(owner: $owner, name: $name) {{
                    result: pullRequests(first: 100, after: $cursor{states},
                      orderBy: {{field: {order}, direction: DESC}}) {{
                      nodes {{ number title url createdAt mergedAt updatedAt }}
                      pageInfo {{ hasNextPage endCursor }}
                    }}
                  }}
                  rateLimit {{ remaining resetAt }}
                }}
            """
        elif source == "default-branch-commits":
            connection = "history"
            variables.update(
                {
                    "head": snapshot["default_branch_head"],
                    "start": scan_window["start"],
                    "end": scan_window["end"],
                }
            )
            query = """
                query($owner: String!, $name: String!, $cursor: String,
                      $head: GitObjectID!, $start: GitTimestamp!, $end: GitTimestamp!) {
                  repository(owner: $owner, name: $name) {
                    object(oid: $head) {
                      ... on Commit {
                        result: history(first: 100, after: $cursor, since: $start, until: $end) {
                          nodes { oid messageHeadline committedDate url }
                          pageInfo { hasNextPage endCursor }
                        }
                      }
                    }
                  }
                  rateLimit { remaining resetAt }
                }
            """
        else:
            raise ValueError(f"unsupported collection source: {source}")
        payload = self._request(query, variables)
        data = payload.get("data")
        repository_data = data.get("repository") if isinstance(data, dict) else None
        if source == "default-branch-commits":
            object_data = (
                repository_data.get("object")
                if isinstance(repository_data, dict)
                else None
            )
            result = object_data.get("result") if isinstance(object_data, dict) else None
        else:
            result = repository_data.get("result") if isinstance(repository_data, dict) else None
        rate = data.get("rateLimit") if isinstance(data, dict) else None
        if not isinstance(result, dict) or not isinstance(rate, dict):
            raise ValueError(f"{connection}: malformed GitHub response")
        raw_nodes = result.get("nodes")
        page_info = result.get("pageInfo")
        if not isinstance(raw_nodes, list) or not isinstance(page_info, dict):
            raise ValueError(f"{connection}: malformed GitHub connection")
        nodes = []
        for node in raw_nodes:
            if not isinstance(node, dict):
                raise ValueError(f"{connection}: malformed GitHub node")
            if source == "issues-created":
                number = node.get("number")
                event_at, order_at, kind = node.get("createdAt"), None, "issue"
            elif source == "prs-created":
                number = node.get("number")
                event_at, order_at, kind = node.get("createdAt"), None, "pr"
            elif source == "prs-merged":
                number = node.get("number")
                event_at, order_at, kind = (
                    node.get("mergedAt"),
                    node.get("updatedAt"),
                    "pr",
                )
                if event_at is None:
                    raise ValueError("merged pull request has no mergedAt")
            else:
                number = node.get("oid")
                event_at, order_at, kind = node.get("committedDate"), None, "commit"
            prefix = "commit" if kind == "commit" else kind
            nodes.append(
                {
                    "id": f"{prefix}-{number}",
                    "kind": kind,
                    "title": node.get("messageHeadline") if kind == "commit" else node.get("title"),
                    "url": node.get("url"),
                    "event_at": event_at,
                    **({"order_at": order_at} if order_at is not None else {}),
                }
            )
        return {
            "nodes": nodes,
            "page_info": {
                "has_next_page": page_info.get("hasNextPage"),
                "end_cursor": page_info.get("endCursor"),
            },
            "rate": {"remaining": rate.get("remaining"), "reset_at": rate.get("resetAt")},
            "raw": payload,
        }


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timestamp(value: object) -> datetime:
    return datetime.strptime(str(value), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _collection_manifest(
    repository: str,
    scan_window: dict[str, str],
    snapshot: dict[str, object],
    collected_at: str,
    source_states: dict[str, dict[str, object]],
    inventory: dict[str, dict[str, object]],
) -> dict[str, object]:
    normalized: list[dict[str, object]] = []
    observed_count = 0
    for original in sorted(inventory.values(), key=lambda item: str(item["id"])):
        item = {**original}
        raw_events = original.get("events")
        events = (
            sorted(raw_events, key=lambda event: (str(event["at"]), str(event["type"])))
            if isinstance(raw_events, list)
            else []
        )
        item["events"] = events
        observed_count += len(events)
        normalized.append(item)
    sources = [source_states[name] for name in SOURCES]
    blockers = [
        f"{source['source']}:{source['error']['kind']}"
        for source in sources
        if source["status"] == "partial"
    ]
    return {
        "schema_version": 1,
        "status": "partial" if blockers else "complete",
        "repository": repository,
        "scan_window": scan_window,
        "snapshot": {"collected_at": collected_at, **snapshot},
        "sources": sources,
        "observed_count": observed_count,
        "unique_count": len(normalized),
        "inventory": normalized,
        "blockers": blockers,
    }


def collect(
    repository: str,
    scan_window: dict[str, str],
    output: Path,
    github: GitHub,
) -> dict[str, object]:
    snapshot = github.snapshot(repository)
    collected_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    start = _timestamp(scan_window["start"])
    end = _timestamp(scan_window["end"])
    inventory: dict[str, dict[str, object]] = {}
    source_states = {
        source: {
            "source": source,
            "status": "partial",
            "pages_completed": 0,
            "items_fetched": 0,
            "last_cursor": None,
            "boundary_reached": False,
            "rate_remaining": None,
            "rate_reset_at": None,
            "error": {
                "kind": "not-started",
                "message": "source collection has not started",
            },
            "pages": [],
        }
        for source in SOURCES
    }

    def checkpoint() -> dict[str, object]:
        result = _collection_manifest(
            repository,
            scan_window,
            snapshot,
            collected_at,
            source_states,
            inventory,
        )
        _write_json(output / "collection/collection.json", result)
        return result

    checkpoint()
    for source in SOURCES:
        cursor: str | None = None
        seen_cursors: set[str] = set()
        pages: list[dict[str, object]] = []
        fetched = 0
        rate: dict[str, object] = {"remaining": None, "reset_at": None}
        state = source_states[source]
        state["error"] = {
            "kind": "in-progress",
            "message": "source collection is in progress",
        }
        checkpoint()
        while True:
            if cursor is not None:
                seen_cursors.add(cursor)
            try:
                response = github.page(repository, source, cursor, snapshot, scan_window)
                nodes = response["nodes"]
                page_info = response["page_info"]
                rate = response["rate"]
                if (
                    not isinstance(nodes, list)
                    or not isinstance(page_info, dict)
                    or not isinstance(page_info.get("has_next_page"), bool)
                    or not isinstance(rate, dict)
                    or "raw" not in response
                ):
                    raise ValueError(f"{source}: malformed page")
                for node in nodes:
                    if not isinstance(node, dict):
                        raise ValueError(f"{source}: malformed node")
                    _timestamp(node["event_at"])
                    _timestamp(node.get("order_at", node["event_at"]))
            except FetchError as error:
                state.update(
                    {
                        "status": "partial",
                        "pages_completed": len(pages),
                        "items_fetched": fetched,
                        "last_cursor": cursor,
                        "boundary_reached": False,
                        "rate_remaining": (
                            error.remaining
                            if error.remaining is not None
                            else rate.get("remaining")
                        ),
                        "rate_reset_at": error.reset_at or rate.get("reset_at"),
                        "error": {"kind": error.kind, "message": str(error)},
                        "pages": pages,
                    }
                )
                checkpoint()
                break
            except (KeyError, TypeError, ValueError) as error:
                state.update(
                    {
                        "status": "partial",
                        "pages_completed": len(pages),
                        "items_fetched": fetched,
                        "last_cursor": cursor,
                        "boundary_reached": False,
                        "error": {"kind": "malformed", "message": str(error)},
                        "pages": pages,
                    }
                )
                malformed = _collection_manifest(
                    repository,
                    scan_window,
                    snapshot,
                    collected_at,
                    source_states,
                    inventory,
                )
                malformed["status"] = "malformed"
                _write_json(output / "collection/collection.json", malformed)
                raise ValueError(f"{source}: malformed response: {error}") from error
            page_path = Path("collection/pages") / source / f"page_{len(pages) + 1:04d}.json"
            digest = _write_json(output / page_path, response["raw"])
            next_cursor = page_info.get("end_cursor")
            pages.append(
                {
                    "path": str(page_path),
                    "sha256": digest,
                    "cursor": cursor,
                    "next_cursor": next_cursor,
                    "count": len(nodes),
                }
            )
            fetched += len(nodes)
            reached_boundary = False
            for node in nodes:
                event_time = _timestamp(node["event_at"])
                order_time = _timestamp(node.get("order_at", node["event_at"]))
                reached_boundary = reached_boundary or order_time < start
                if not start <= event_time < end:
                    continue
                unit_id = str(node["id"])
                event = {"type": EVENT_TYPES[source], "at": node["event_at"]}
                if unit_id not in inventory:
                    inventory[unit_id] = {
                        "id": unit_id,
                        "kind": node["kind"],
                        "title": node["title"],
                        "url": node["url"],
                        "events": [],
                    }
                events = inventory[unit_id]["events"]
                if isinstance(events, list) and event not in events:
                    events.append(event)
            state.update(
                {
                    "pages_completed": len(pages),
                    "items_fetched": fetched,
                    "last_cursor": next_cursor,
                    "rate_remaining": rate.get("remaining"),
                    "rate_reset_at": rate.get("reset_at"),
                    "pages": pages,
                }
            )
            checkpoint()
            needs_next_cursor = not reached_boundary and page_info.get("has_next_page")
            repeated_cursor = (
                isinstance(next_cursor, str) and next_cursor in seen_cursors
            )
            if repeated_cursor or (
                needs_next_cursor
                and (not isinstance(next_cursor, str) or not next_cursor)
            ):
                state["error"] = {
                    "kind": "malformed",
                    "message": "cursor did not advance",
                }
                malformed = _collection_manifest(
                    repository,
                    scan_window,
                    snapshot,
                    collected_at,
                    source_states,
                    inventory,
                )
                malformed["status"] = "malformed"
                _write_json(output / "collection/collection.json", malformed)
                raise ValueError(f"{source}: invalid cursor")
            if reached_boundary or not page_info.get("has_next_page"):
                state.update(
                    {
                        "status": "complete",
                        "boundary_reached": True,
                        "error": None,
                    }
                )
                checkpoint()
                break
            cursor = next_cursor
    return checkpoint()


def validate_collection(
    root: Path, expected_window: dict[str, str]
) -> tuple[Path, dict[str, object], dict[str, dict[str, object]]]:
    path = root / "collection/collection.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CollectionError(f"collection manifest is unreadable: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise CollectionError("collection manifest is not a v1 object")
    status = manifest.get("status")
    if status not in {"complete", "partial"}:
        raise CollectionError("collection status is invalid")
    if manifest.get("repository") != "pytorch/pytorch":
        raise CollectionError("collection repository is invalid")
    if manifest.get("scan_window") != expected_window:
        raise CollectionError("collection scan window does not match")
    snapshot = manifest.get("snapshot")
    if not isinstance(snapshot, dict) or not all(
        isinstance(snapshot.get(field), str) and snapshot[field]
        for field in ("collected_at", "default_branch", "default_branch_head")
    ):
        raise CollectionError("collection snapshot is invalid")
    try:
        _timestamp(snapshot["collected_at"])
    except ValueError as error:
        raise CollectionError("collection snapshot timestamp is invalid") from error
    head = snapshot["default_branch_head"]
    if len(head) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in head
    ):
        raise CollectionError("collection snapshot head is invalid")
    raw_sources = manifest.get("sources")
    if not isinstance(raw_sources, list):
        raise CollectionError("collection sources are invalid")
    sources: dict[str, dict[str, object]] = {}
    for source in raw_sources:
        if not isinstance(source, dict):
            raise CollectionError("collection source is not an object")
        name = source.get("source")
        if name not in SOURCES or name in sources:
            raise CollectionError(f"collection source is invalid: {name}")
        source_status = source.get("status")
        if source_status not in {"complete", "partial"}:
            raise CollectionError(f"{name}: invalid status")
        pages = source.get("pages")
        pages_completed = source.get("pages_completed")
        if (
            not isinstance(pages, list)
            or not isinstance(pages_completed, int)
            or isinstance(pages_completed, bool)
            or pages_completed != len(pages)
        ):
            raise CollectionError(f"{name}: page count mismatch")
        if source_status == "complete" and not pages:
            raise CollectionError(f"{name}: complete source has no raw page")
        cursor: str | None = None
        seen_cursors: set[str] = set()
        fetched = 0
        for index, page in enumerate(pages):
            if not isinstance(page, dict) or page.get("cursor") != cursor:
                raise CollectionError(f"{name}: cursor chain is invalid at page {index + 1}")
            if cursor is not None:
                if cursor in seen_cursors:
                    raise CollectionError(
                        f"{name}: cursor chain repeats at page {index + 1}"
                    )
                seen_cursors.add(cursor)
            relative = Path(str(page.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise CollectionError(f"{name}: page path is unsafe")
            raw_path = root / relative
            try:
                raw_path.resolve(strict=True).relative_to(root.resolve(strict=True))
            except (OSError, ValueError) as error:
                raise CollectionError(f"{name}: raw page is missing") from error
            if raw_path.is_symlink() or not raw_path.is_file():
                raise CollectionError(f"{name}: raw page is not a regular file")
            if page.get("sha256") != sha256(raw_path):
                raise CollectionError(f"{name}: raw page digest mismatch")
            count = page.get("count")
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise CollectionError(f"{name}: invalid page item count")
            fetched += count
            next_cursor = page.get("next_cursor")
            if next_cursor is not None and not isinstance(next_cursor, str):
                raise CollectionError(f"{name}: invalid next cursor")
            if isinstance(next_cursor, str) and next_cursor in seen_cursors:
                raise CollectionError(
                    f"{name}: cursor chain repeats at page {index + 1}"
                )
            cursor = next_cursor
        items_fetched = source.get("items_fetched")
        if (
            not isinstance(items_fetched, int)
            or isinstance(items_fetched, bool)
            or items_fetched != fetched
            or source.get("last_cursor") != cursor
        ):
            raise CollectionError(f"{name}: progress does not match pages")
        remaining = source.get("rate_remaining")
        reset_at = source.get("rate_reset_at")
        if remaining is not None and (
            not isinstance(remaining, int)
            or isinstance(remaining, bool)
            or remaining < 0
        ):
            raise CollectionError(f"{name}: rate remaining is invalid")
        if reset_at is not None:
            try:
                _timestamp(reset_at)
            except ValueError as error:
                raise CollectionError(f"{name}: rate reset is invalid") from error
        error = source.get("error")
        if source_status == "complete" and (
            source.get("boundary_reached") is not True or error is not None
        ):
            raise CollectionError(f"{name}: complete source has incomplete progress")
        if source_status == "partial" and (
            source.get("boundary_reached") is not False
            or not isinstance(error, dict)
            or not str(error.get("kind", "")).strip()
            or not str(error.get("message", "")).strip()
        ):
            raise CollectionError(f"{name}: partial source has invalid progress")
        sources[str(name)] = source
    if set(sources) != set(SOURCES):
        raise CollectionError("collection source coverage is incomplete")
    blockers = manifest.get("blockers")
    if not isinstance(blockers, list):
        raise CollectionError("collection blockers are invalid")
    any_partial = any(source["status"] == "partial" for source in sources.values())
    expected_blockers = [
        f"{name}:{source['error']['kind']}"
        for name, source in sources.items()
        if source["status"] == "partial"
    ]
    if (
        (status == "partial") != any_partial
        or blockers != expected_blockers
    ):
        raise CollectionError("collection status does not match source progress")
    raw_inventory = manifest.get("inventory")
    if not isinstance(raw_inventory, list):
        raise CollectionError("collection inventory is invalid")
    inventory: dict[str, dict[str, object]] = {}
    observed = 0
    start, end = _timestamp(expected_window["start"]), _timestamp(expected_window["end"])
    for item in raw_inventory:
        if not isinstance(item, dict):
            raise CollectionError("collection inventory item is invalid")
        unit_id, kind, events = item.get("id"), item.get("kind"), item.get("events")
        if not isinstance(unit_id, str) or not unit_id or unit_id in inventory:
            raise CollectionError(f"collection inventory id is invalid: {unit_id}")
        if kind not in {"issue", "pr", "commit"} or not isinstance(events, list) or not events:
            raise CollectionError(f"{unit_id}: collection inventory shape is invalid")
        if kind == "issue":
            valid_id = unit_id.startswith("issue-") and unit_id[6:].isdigit()
            expected_url = f"https://github.com/pytorch/pytorch/issues/{unit_id[6:]}"
            allowed_events = {"created"}
        elif kind == "pr":
            valid_id = unit_id.startswith("pr-") and unit_id[3:].isdigit()
            expected_url = f"https://github.com/pytorch/pytorch/pull/{unit_id[3:]}"
            allowed_events = {"created", "merged"}
        else:
            commit_id = unit_id.removeprefix("commit-")
            valid_id = (
                unit_id.startswith("commit-")
                and len(commit_id) in {40, 64}
                and all(character in "0123456789abcdef" for character in commit_id)
            )
            expected_url = f"https://github.com/pytorch/pytorch/commit/{commit_id}"
            allowed_events = {"committed"}
        if (
            not valid_id
            or not isinstance(item.get("title"), str)
            or not item["title"].strip()
            or item.get("url") != expected_url
        ):
            raise CollectionError(f"{unit_id}: collection inventory metadata is invalid")
        seen_events: set[tuple[str, str]] = set()
        for event in events:
            if not isinstance(event, dict) or event.get("type") not in allowed_events:
                raise CollectionError(f"{unit_id}: collection event metadata is invalid")
            event_key = (str(event["type"]), str(event.get("at")))
            if event_key in seen_events:
                raise CollectionError(f"{unit_id}: collection event is duplicated")
            seen_events.add(event_key)
            try:
                event_time = _timestamp(event.get("at"))
            except ValueError as error:
                raise CollectionError(
                    f"{unit_id}: collection event timestamp is invalid"
                ) from error
            if not start <= event_time < end:
                raise CollectionError(f"{unit_id}: collection event is outside the scan window")
            observed += 1
        inventory[unit_id] = item
    if manifest.get("observed_count") != observed or manifest.get("unique_count") != len(inventory):
        raise CollectionError("collection inventory counts do not match")
    return path, manifest, inventory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default="pytorch/pytorch")
    parser.add_argument("--scan-date", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        day = date.fromisoformat(args.scan_date)
        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if not token:
            raise ValueError("GH_TOKEN or GITHUB_TOKEN is required")
        window = {
            "start": f"{day.isoformat()}T00:00:00Z",
            "end": f"{(day + timedelta(days=1)).isoformat()}T00:00:00Z",
        }
        result = collect(args.repository, window, args.output, GitHubGraphQL(token))
        manifest = args.output / "collection/collection.json"
    except (FetchError, OSError, ValueError) as error:
        print(f"collection failed: {error}")
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "observed_count": result["observed_count"],
                "manifest": str(manifest),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
