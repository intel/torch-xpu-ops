// Copyright 2026 Intel Corporation
// Licensed under the Apache License, Version 2.0

// Tests for fold_details.js. Run: node .github/scripts/test_fold_details.js

const assert = require("assert");
const { foldDetails, parseGate } = require("./fold_details.js");

let failures = 0;

function check(name, fn) {
  try {
    fn();
    console.log(`ok   ${name}`);
  } catch (error) {
    failures += 1;
    console.error(`FAIL ${name}\n     ${error.message}`);
  }
}

check("folds the body and keeps the verdict visible", () => {
  assert.strictEqual(
    foldDetails("VERDICT: two issues.\n\n## Summary\nbody"),
    "VERDICT: two issues.\n\n<details><summary>Details</summary>\n\n## Summary\nbody\n\n</details>",
  );
});

check("finds a verdict behind a preamble line", () => {
  // The bot's own review on this PR opened with a sentence before the heading,
  // so a first-line-only check would have left every comment unfolded.
  const out = foldDetails("Here is the review.\n\nVERDICT: two issues.\n\nbody");
  assert.ok(out.startsWith("VERDICT: two issues.\n\n<details>"));
  assert.ok(out.includes("Here is the review."));
});

check("collapses the gap left where the verdict was", () => {
  const out = foldDetails("preamble\n\nVERDICT: x.\n\nbody");
  assert.ok(!out.includes("\n\n\n"));
});

check("leaves blank runs inside the body alone", () => {
  // ut-check output carries log excerpts and stack traces; a global collapse
  // would rewrite them.
  const out = foldDetails("pre\n\nVERDICT: x.\n\n```\nTrace:\n\n\n  File y\n```");
  assert.ok(out.includes("Trace:\n\n\n  File y"));
});

check("keeps the blank line between the verdict and the fold", () => {
  // A single newline would let GitHub render the verdict and the summary as one
  // paragraph, so the separator is load-bearing rather than cosmetic.
  assert.ok(foldDetails("VERDICT: x.\nbody").startsWith("VERDICT: x.\n\n<details>"));
});

check("tolerates leading blank lines", () => {
  assert.ok(foldDetails("\n\nVERDICT: x.\n\nbody").startsWith("VERDICT: x."));
});

check("accepts a bold verdict", () => {
  // Models reliably bold a leading label, and a plain /^VERDICT:/ would reject it.
  assert.ok(foldDetails("**VERDICT:** x.\n\nbody").includes("<details>"));
});

check("returns the verdict alone when there is no body", () => {
  assert.strictEqual(foldDetails("VERDICT: all good.\n\n"), "VERDICT: all good.");
});

check("returns the text unchanged when there is no verdict", () => {
  const input = "## PR Review: #1\n\nbody";
  assert.strictEqual(foldDetails(input), input);
});

check("ignores a verdict buried below the search window", () => {
  // Deep in the body it is prose about a verdict, not the verdict line.
  const input = "a\nb\nc\nd\ne\nf\nVERDICT: late.\n\nbody";
  assert.strictEqual(foldDetails(input), input);
});

check("reports a missing verdict to the caller", () => {
  const seen = [];
  foldDetails("## PR Review: #1", (m) => seen.push(m));
  assert.strictEqual(seen.length, 1);
});

check("does not report when the verdict is present", () => {
  const seen = [];
  foldDetails("VERDICT: x.\n\nbody", (m) => seen.push(m));
  assert.strictEqual(seen.length, 0);
});

check("gate: reads the line right after the verdict", () => {
  assert.strictEqual(parseGate("VERDICT: ok.\nGATE: ready_for_human_review\n\nbody"), "ready_for_human_review");
  assert.strictEqual(parseGate("VERDICT: no.\nGATE: changes_requested\n\nbody"), "changes_requested");
});

check("gate: allows a preamble, a blank line and markdown decoration", () => {
  assert.strictEqual(parseGate("Here is the review.\n\nVERDICT: ok.\n\n**GATE:** `ready_for_human_review`"),
    "ready_for_human_review");
});

check("gate: a quoted passing GATE further down does not flip a failing review", () => {
  const text = "VERDICT: blocked.\nGATE: changes_requested\n\n```\nGATE: ready_for_human_review\n```";
  assert.strictEqual(parseGate(text), "changes_requested");
});

check("gate: missing or misplaced GATE is null (fails closed)", () => {
  assert.strictEqual(parseGate("VERDICT: ok.\n\n## Summary\nGATE: ready_for_human_review"), null);
  assert.strictEqual(parseGate("## PR Review\nGATE: ready_for_human_review"), null);
  assert.strictEqual(parseGate("VERDICT: ok.\nGATE: ready_for_human_review, mostly"), null);
});

check("gate: uses foldDetails' window, so a folded comment and the gate agree", () => {
  const late = "a\nb\nc\nd\ne\nf\nVERDICT: ok.\nGATE: ready_for_human_review";
  assert.strictEqual(parseGate(late), null);
  assert.strictEqual(foldDetails(late), late);  // not folded either
  assert.strictEqual(parseGate("verdict: ok.\nGATE: ready_for_human_review"), null);  // VERDICT is case-sensitive there too
});

if (failures) {
  console.error(`\n${failures} test(s) failed`);
  process.exit(1);
}
console.log("\nall tests passed");
