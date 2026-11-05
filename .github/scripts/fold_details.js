// Copyright 2026 Intel Corporation
// Licensed under the Apache License, Version 2.0

// Hoist a bot comment's VERDICT line and fold the rest behind a summary.
//
// The line is prescribed by each skill's own Output Format section, not only by
// the workflow prompt: a prompt alone loses to the skill body, which is the more
// specific instruction.
//
// The search covers the first few lines rather than only the first, because
// models routinely open with a sentence of preamble -- the bot's own review on
// the PR that introduced this began with one, which would have left every
// comment unfolded.
//
// A model that omits the line entirely degrades to an unfolded comment rather
// than hiding everything behind a summary that says nothing, and the caller is
// told so the fallback is distinguishable from success in the run log.

const VERDICT_RE = /^\**VERDICT:\**/;

// Far enough to clear a preamble, close enough that a VERDICT buried in the
// body is treated as prose rather than promoted out of context.
const SEARCH_LINES = 6;

function foldDetails(text, onMissingVerdict) {
  const lines = String(text).trim().split("\n");
  const at = lines.findIndex((l) => VERDICT_RE.test(l.trim()));
  if (at === -1 || at >= SEARCH_LINES) {
    if (onMissingVerdict) {
      onMissingVerdict(
        "no VERDICT line near the top of the model output; posting it unfolded",
      );
    }
    return text;
  }
  const verdict = lines[at].trim();
  // Close only the gap the hoisted line leaves behind. A global collapse would
  // also rewrite blank runs inside the body, and log excerpts and stack traces
  // in ut-check output depend on those.
  const before = lines.slice(0, at);
  const after = lines.slice(at + 1);
  while (before.length && !before[before.length - 1].trim()) before.pop();
  while (after.length && !after[0].trim()) after.shift();
  const detail = [...before, ...(before.length && after.length ? [""] : []), ...after]
    .join("\n")
    .trim();
  if (!detail) return verdict;
  return `${verdict}\n\n<details><summary>Details</summary>\n\n${detail}\n\n</details>`;
}

module.exports = { foldDetails };
