#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Offline tests for bot_nightly_watch. `gh` is never called.

Covers only where a mistake gives a plausible wrong answer instead of a crash:
a duplicated copy starts a second fix job on a shared GPU runner, and a dropped
one leaves a test disabled forever.
"""

import importlib.util
import json
import os
import sys
import unittest

_spec = importlib.util.spec_from_file_location(
    "bot_nightly_watch",
    os.path.join(os.path.dirname(__file__), "bot_nightly_watch.py"))
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)

DISABLE = ("Commit `abc123`\nDisable issues:\n"
           "- https://github.com/pytorch/pytorch/issues/196247\n")
RERUN = "Commit `def456`\nRerun: unparseable job log\n"


def comment(cid, body):
    return {"id": cid, "body": body, "created_at": "2026-09-08T00:00:00Z"}


class TestAnnouncements(unittest.TestCase):
    """#364 is a log, not a queue: most comments are reruns, and only some
    carry `Disable issues:`."""

    def feed(self, *comments):
        original = w.gh
        self.addCleanup(lambda: setattr(w, "gh", original))
        w.gh = lambda *a: "\n".join(json.dumps(c) for c in comments)

    def test_only_disable_comments_are_copied(self):
        self.feed(comment(1, RERUN), comment(2, DISABLE))
        found, _ = w.announcements(0)
        self.assertEqual([c["id"] for c in found], [2])

    def test_the_cursor_passes_reruns_too(self):
        """It has to cover comments that announced nothing, or they are re-read
        on every run forever."""
        self.feed(comment(1, DISABLE), comment(2, RERUN))
        found, newest = w.announcements(0)
        self.assertEqual(([c["id"] for c in found], newest), ([1], 2))

    def test_nothing_before_the_cursor_comes_back(self):
        self.feed(comment(1, DISABLE), comment(2, RERUN))
        self.assertEqual(w.announcements(1), ([], 2))

    def test_oldest_first_whatever_order_they_arrive_in(self):
        """--limit cuts the tail, and the oldest disable has been blocking CI
        the longest."""
        self.feed(comment(9, DISABLE), comment(2, DISABLE))
        found, newest = w.announcements(0)
        self.assertEqual(([c["id"] for c in found], newest), ([2, 9], 9))

    def test_an_empty_log_does_not_move_the_cursor_backwards(self):
        self.feed()
        self.assertEqual(w.announcements(42), ([], 42))


class TestUnseen(unittest.TestCase):
    """#364 re-announces a test every night it stays disabled: 16 of the first
    36 announcements were pure repeats, and each one would start another fix
    job for an issue already being worked."""

    def disable(self, cid, *issues):
        links = "".join(
            f"- https://github.com/pytorch/pytorch/issues/{i}\n" for i in issues)
        return comment(cid, f"Commit `abc123`\nDisable issues:\n{links}")

    def test_a_repeat_is_dropped(self):
        found = [self.disable(1, "111")]
        self.assertEqual(w.unseen(found, {"111"}), [])

    def test_a_new_issue_is_kept(self):
        found = [self.disable(1, "222")]
        self.assertEqual([c["id"] for c in w.unseen(found, {"111"})], [1])

    def test_a_repeat_within_one_run_is_dropped(self):
        """`done` has to grow as we go, or a burst copies the same issue
        twice in a single run."""
        found = [self.disable(1, "111"), self.disable(2, "111")]
        self.assertEqual([c["id"] for c in w.unseen(found, set())], [1])

    def test_partly_new_is_copied_whole(self):
        """Rare (3 of 36) and deliberately not split: the old issue gets
        re-fixed."""
        found = [self.disable(1, "111", "222")]
        self.assertEqual([c["id"] for c in w.unseen(found, {"111"})], [1])

    def test_the_caller_set_is_left_alone(self):
        done = {"111"}
        w.unseen([self.disable(1, "222")], done)
        self.assertEqual(done, {"111"})


class TestCommandComment(unittest.TestCase):
    def test_the_command_is_the_first_line(self):
        """bot.yml anchors on `^@torchxpubot <cmd>`. Anything above it -- even
        a blank line -- and the command is never seen."""
        self.assertTrue(
            w.command_comment(comment(5, DISABLE)).startswith("@torchxpubot fix\n"))

    def test_the_announcement_is_copied_verbatim_and_linked(self):
        out = w.command_comment(comment(5, DISABLE))
        self.assertIn(DISABLE.strip(), out)
        self.assertIn("issuecomment-5", out)

    def test_an_empty_body_does_not_crash(self):
        w.command_comment(comment(5, None))


class TestCursor(unittest.TestCase):
    """State lives in the tracking issue body, visible to anyone reading it."""

    def test_the_marker_round_trips(self):
        self.assertEqual(
            w.MARKER_RE.search("text\n\n<!-- last-364-comment: 12345 -->\n").group(1),
            "12345")

    def test_an_absent_marker_is_not_an_error(self):
        self.assertIsNone(w.MARKER_RE.search("no marker here"))

    def test_rewriting_does_not_stack_markers(self):
        body = "text\n<!-- last-364-comment: 1 -->\n"
        out = f"{w.MARKER_RE.sub('', body).rstrip()}\n\n<!-- last-364-comment: 2 -->\n"
        self.assertEqual(w.MARKER_RE.findall(out), ["2"])
        self.assertIn("text", out)


class TestLimit(unittest.TestCase):
    """The dispatch form takes `limit` as free text. Rejecting it before the
    first API call is what makes this testable without touching `gh`."""

    def run_main(self, *argv):
        """`gh` is replaced, not merely expected to go unreached. A bad limit
        must be refused before any request, and a regression here would
        otherwise post real `@torchxpubot fix` comments from whatever token is
        in the environment."""
        original = w.gh
        w.gh = lambda *a: self.fail("gh was called for a rejected --limit")
        self.addCleanup(lambda: setattr(w, "gh", original))
        old = sys.argv
        sys.argv = ["bot_nightly_watch.py", *argv]
        self.addCleanup(lambda: setattr(sys, "argv", old))
        return w.main()

    def test_zero_is_rejected(self):
        """`0` reads like "no cap" but slices to nothing, and the cursor then
        indexes an empty list."""
        with self.assertRaises(SystemExit):
            self.run_main("--limit", "0")

    def test_a_negative_limit_is_rejected(self):
        """Worse than a crash: `-1` copies every announcement but the newest
        and writes a plausible-looking cursor."""
        with self.assertRaises(SystemExit):
            self.run_main("--limit", "-1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
