// Copyright 2026 Intel Corporation
// Licensed under the Apache License, Version 2.0

// Keep the model's VERDICT line visible and fold the rest of a bot comment.
//
// The VERDICT line is prescribed by each skill's own Output Format section, not
// only by the workflow prompt: a prompt alone loses to the skill body, which is
// the more specific instruction. A model that still omits it degrades to an
// unfolded comment rather than hiding everything behind a summary that says
// nothing, and the caller is told so it does not look like the fold worked.

const VERDICT_RE = /^\**VERDICT:\**/;

function foldDetails(text, onMissingVerdict) {
  const [first, ...rest] = String(text).trim().split("\n");
  if (!VERDICT_RE.test(first.trim())) {
    if (onMissingVerdict) {
      onMissingVerdict("no VERDICT line in the model output; posting it unfolded");
    }
    return text;
  }
  const detail = rest.join("\n").trim();
  if (!detail) return first.trim();
  return `${first.trim()}\n\n<details><summary>Details</summary>\n\n${detail}\n\n</details>`;
}

module.exports = { foldDetails };
